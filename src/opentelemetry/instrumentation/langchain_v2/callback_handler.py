import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from langchain_core.callbacks import (
    BaseCallbackHandler,
)

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.context.context import Context
from opentelemetry.trace import SpanKind, set_span_in_context
from opentelemetry.trace.span import Span
from opentelemetry.util.types import AttributeValue
from uuid import UUID

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY


from opentelemetry.instrumentation.langchain_v2.span_attributes import Span_Attributes, GenAIOperationValues
from opentelemetry.instrumentation.langchain_v2.utils import dont_throw, CallbackFilteredJSONEncoder
from opentelemetry.trace.status import Status, StatusCode

# below dataclass stolen from openLLMetry
@dataclass
class SpanHolder:
    span: Span
    context: Context
    children: list[UUID]
    start_time: float = field(default_factory=time.time)
    request_model: Optional[str] = None
    
def _set_request_params(span, kwargs, span_holder: SpanHolder):
    for model_tag in ("model", "model_id", "model_name"):
        if (model := kwargs.get(model_tag)) is not None:
            span_holder.request_model = model
            break
        elif (
            model := (kwargs.get("invocation_params") or {}).get(model_tag)
        ) is not None:
            span_holder.request_model = model
            break
    else:
        model = "unknown"

    _set_span_attribute(span, Span_Attributes.GEN_AI_REQUEST_MODEL, model)
    # response is not available for LLM requests (as opposed to chat)
    _set_span_attribute(span, Span_Attributes.GEN_AI_RESPONSE_MODEL, model)

    if "invocation_params" in kwargs:
        params = (
            kwargs["invocation_params"].get("params") or kwargs["invocation_params"]
        )
    else:
        params = kwargs
    
    _set_span_attribute(
        span,
        Span_Attributes.GEN_AI_REQUEST_MAX_TOKENS,
        params.get("max_tokens") or params.get("max_new_tokens"),
    )
    
    _set_span_attribute(
        span, Span_Attributes.GEN_AI_REQUEST_TEMPERATURE, params.get("temperature")
    )
    
    _set_span_attribute(span, Span_Attributes.GEN_AI_REQUEST_TOP_P, params.get("top_p"))

    tools = kwargs.get("invocation_params", {}).get("tools", [])
    for i, tool in enumerate(tools):
        tool_function = tool.get("function", tool)
        _set_span_attribute(
            span,
            f"{Span_Attributes.GEN_AI_TOOL_NAME}.{i}",
            tool_function.get("name"),
        )

        _set_span_attribute(
            span,
            f"{Span_Attributes.GEN_AI_TOOL_DESCRIPTION}.{i}",
            tool_function.get("description"),
        )
        
        _set_span_attribute(
            span,
            f"{Span_Attributes.GEN_AI_TOOL_TYPE}.{i}",
            json.dumps(tool_function.get("parameters", tool.get("input_schema"))),
        )


def _set_span_attribute(span: Span, name: str, value: AttributeValue):
    if value is not None and value != "":
        span.set_attribute(name, value)


def _set_llm_request(
    span: Span,
    serialized: dict[str, Any],
    prompts: list[str],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    _set_request_params(span, kwargs, span_holder)
        
def _set_chat_response(span: Span, response: LLMResult) -> None:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cache_read_tokens = 0

    for generations in response.generations:
        for generation in generations:
            if (
                hasattr(generation, "message")
                and hasattr(generation.message, "usage_metadata")
                and generation.message.usage_metadata is not None
            ):
                input_tokens += (
                    generation.message.usage_metadata.get("input_tokens")
                    or generation.message.usage_metadata.get("prompt_tokens")
                    or 0
                )
                output_tokens += (
                    generation.message.usage_metadata.get("output_tokens")
                    or generation.message.usage_metadata.get("completion_tokens")
                    or 0
                )
                total_tokens = input_tokens + output_tokens

                if generation.message.usage_metadata.get("input_token_details"):
                    input_token_details = generation.message.usage_metadata.get("input_token_details", {})
                    cache_read_tokens += input_token_details.get("cache_read", 0)


    if input_tokens > 0 or output_tokens > 0 or total_tokens > 0 or cache_read_tokens > 0:
        _set_span_attribute(
            span,
            Span_Attributes.GEN_AI_USAGE_INPUT_TOKENS,
            input_tokens,
        )
        _set_span_attribute(
            span,
            Span_Attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            output_tokens,
        )
        
def _sanitize_metadata_value(value: Any) -> Any:
    """Convert metadata values to OpenTelemetry-compatible types."""
    if value is None:
        return None
    if isinstance(value, (bool, str, bytes, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(_sanitize_metadata_value(v)) for v in value]
    return str(value)



def _set_chat_request(
    span: Span,
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    kwargs: Any,
    span_holder: SpanHolder,
) -> None:
    _set_request_params(span, kwargs, span_holder)

    should_trace = context_api.get_value("override_enable_content_tracing") or True
    if should_trace:
        
        for i, function in enumerate(
            kwargs.get("invocation_params", {}).get("functions", [])
        ):   
            _set_span_attribute(span, f"{Span_Attributes.GEN_AI_TOOL_NAME}.{i}", function.get("name"))
            _set_span_attribute(span, f"{Span_Attributes.GEN_AI_TOOL_DESCRIPTION}.{i}", function.get("description"))
            _set_span_attribute(span, f"{Span_Attributes.GEN_AI_TOOL_TYPE}.{i}", json.dumps(function.get("parameters")))

class OpenTelemetryCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer):
        super().__init__()
        self.tracer = tracer
        self.span_mapping: dict[UUID, SpanHolder] = {}
    
    def _get_span(self, run_id: UUID) -> Span:
        return self.span_mapping[run_id].span

    def _end_span(self, span: Span, run_id: UUID) -> None:
        for child_id in self.span_mapping[run_id].children:
            child_span = self.span_mapping[child_id].span
            if child_span.end_time is None:  # avoid warning on ended spans
                child_span.end()
        span.end()
        
    def _create_span(
            self,
            run_id: UUID,
            parent_run_id: Optional[UUID],
            span_name: str,
            kind: SpanKind = SpanKind.INTERNAL,
            metadata: Optional[dict[str, Any]] = None,
        ) -> Span:
            if metadata is not None:
                current_association_properties = (
                    context_api.get_value("association_properties") or {}
                )
                sanitized_metadata = {
                    k: _sanitize_metadata_value(v)
                    for k, v in metadata.items()
                    if v is not None
                }
                context_api.attach(
                    context_api.set_value(
                        "association_properties",
                        {**current_association_properties, **sanitized_metadata},
                    )
                )

            if parent_run_id is not None and parent_run_id in self.span_mapping:
                span = self.tracer.start_span(
                    span_name,
                    context=set_span_in_context(self.span_mapping[parent_run_id].span),
                    kind=kind,
                )
            else:
                span = self.tracer.start_span(span_name, kind=kind)


            self.span_mapping[run_id] = SpanHolder(
                span, None, []
            )

            if parent_run_id is not None and parent_run_id in self.span_mapping:
                self.span_mapping[parent_run_id].children.append(run_id)

            return span

    def _create_task_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        kind: SpanKind,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        # operation_type = kind.value if kind else "INTERNAL"
        # span_name = f"{operation_type} {name}"
    
        # span_name = f"{name}.{kind.value}"
        span_name = f"{name}.{kind}"
        span = self._create_span(
            run_id,
            parent_run_id,
            span_name,
            metadata=metadata,
        )
        return span
        
        
    def _create_llm_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        operation_name: GenAIOperationValues,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:    
        span = self._create_span(
            run_id,
            parent_run_id,
            # f"{name}.{operation_name.value}",
            f"{operation_name.value} {name}",
            kind=SpanKind.CLIENT,
            metadata=metadata,
        )
   
        _set_span_attribute(span, Span_Attributes.GEN_AI_SYSTEM, "Langchain")
        _set_span_attribute(span, Span_Attributes.GEN_AI_OPERATION_NAME, operation_name.value)
        
        return span
    
    
    @staticmethod
    def _get_name_from_callback(
        serialized: dict[str, Any],
        _tags: Optional[list[str]] = None,
        _metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Get the name to be used for the span. Based on heuristic. Can be extended."""
        if serialized and "kwargs" in serialized and serialized["kwargs"].get("name"):
            return serialized["kwargs"]["name"]
        if kwargs.get("name"):
            return kwargs["name"]
        if serialized.get("name"):
            return serialized["name"]
        if "id" in serialized:
            return serialized["id"][-1]

        return "unknown"
    
    def _handle_error(
        self,
        error: BaseException,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Common error handling logic for all components."""
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        span = self._get_span(run_id)
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(error)
        self._end_span(span, run_id)

    
    @dont_throw
    def on_chat_model_start(self, serialized, messages, *, run_id, tags, parent_run_id, metadata, **kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return
        
        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        span = self._create_llm_span(
            run_id, parent_run_id, name, GenAIOperationValues.CHAT, metadata=metadata
        )
        _set_chat_request(span, serialized, messages, kwargs, self.span_mapping[run_id])
    
    
    @dont_throw
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, **kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        span = self._create_llm_span(
            run_id, parent_run_id, name, GenAIOperationValues.TEXT_COMPLETION
        )
        
        _set_llm_request(span, serialized, prompts, kwargs, self.span_mapping[run_id])

        
    @dont_throw
    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        span = self._get_span(run_id)

        model_name = None
        if response.llm_output is not None:
            model_name = response.llm_output.get(
                "model_name"
            ) or response.llm_output.get("model_id")
            if model_name is not None:
                _set_span_attribute(span, Span_Attributes.GEN_AI_RESPONSE_MODEL, model_name)

                if self.span_mapping[run_id].request_model is None:
                    _set_span_attribute(span, Span_Attributes.GEN_AI_REQUEST_MODEL, model_name)
            id = response.llm_output.get("id")
            if id is not None and id != "":
                _set_span_attribute(span, Span_Attributes.GEN_AI_RESPONSE_ID, id)

        token_usage = (response.llm_output or {}).get("token_usage") or (
            response.llm_output or {}
        ).get("usage")
        if token_usage is not None:
            prompt_tokens = (
                token_usage.get("prompt_tokens")
                or token_usage.get("input_token_count")
                or token_usage.get("input_tokens")
            )
            completion_tokens = (
                token_usage.get("completion_tokens")
                or token_usage.get("generated_token_count")
                or token_usage.get("output_tokens")
            )
            
            _set_span_attribute(
                span, Span_Attributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
            )
        
            _set_span_attribute(
                span, Span_Attributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
            )
        span.update_name(f"{model_name}") if model_name else span.update_name(span.name)
        _set_chat_response(span, response)
        self._end_span(span, run_id)

    @dont_throw
    def on_llm_error(self, error, *, run_id, parent_run_id, **kwargs):
        self._handle_error(error, run_id, parent_run_id, **kwargs)

    @dont_throw
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id, tags, metadata, **kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = self._get_name_from_callback(serialized, **kwargs)
        kind = SpanKind.INTERNAL

        span = self._create_task_span(
            run_id,
            parent_run_id,
            name,
            kind,
            metadata,
        )
        should_trace = context_api.get_value("override_enable_content_tracing") or True
        if should_trace:
            _set_span_attribute(
                span,
                "app.entity.input", # gpt generated alternative naming scheme
                json.dumps(
                    {
                        "inputs": inputs,
                        "tags": tags,
                        "metadata": metadata,
                        "kwargs": kwargs,
                    },
                    cls=CallbackFilteredJSONEncoder,
                ),
            )

    @dont_throw
    def on_chain_end(self, outputs, *, run_id, parent_run_id, **kwargs):   
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        span_holder = self.span_mapping[run_id]
        span = span_holder.span
        should_trace = context_api.get_value("override_enable_content_tracing") or True
        if should_trace:
            _set_span_attribute(
                span,
                "app.entity.output",
                json.dumps(
                    {"outputs": outputs, "kwargs": kwargs},
                    cls=CallbackFilteredJSONEncoder,
                ),
            )

        self._end_span(span, run_id)
        _set_request_params(span, kwargs, self.span_mapping[run_id])

    @dont_throw
    def on_chain_error(self, error, run_id, parent_run_id, tags, **kwargs):
        self._handle_error(error, run_id, parent_run_id, **kwargs)
        
    @dont_throw
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id, tags, metadata, inputs, **kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = self._get_name_from_callback(serialized, kwargs=kwargs)

        span = self._create_task_span(
            run_id,
            parent_run_id,
            name,
            SpanKind.INTERNAL,
        )
        
        should_trace = context_api.get_value("override_enable_content_tracing") or True
        if should_trace:
            _set_span_attribute(
                span,
                "app.entity.input",
                json.dumps(
                    {
                        "input_str": input_str,
                        "tags": tags,
                        "metadata": metadata,
                        "inputs": inputs,
                        "kwargs": kwargs,
                    },
                    cls=CallbackFilteredJSONEncoder,
                ),
            )
    
    @dont_throw
    def on_tool_end(self, output, *, run_id, parent_run_id, **kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        span = self._get_span(run_id)
        should_trace = context_api.get_value("override_enable_content_tracing") or True
        if should_trace:
            _set_span_attribute(
                span,
                "app.entity.output",
                json.dumps(
                    {"output": output, "kwargs": kwargs},
                    cls=CallbackFilteredJSONEncoder,
                ),
            )
        self._end_span(span, run_id)
    
    @dont_throw
    def on_tool_error(self, error, run_id, parent_run_id, **kwargs):
        self._handle_error(error, run_id, parent_run_id, **kwargs)
    
    
    def on_agent_action(self, action, run_id, parent_run_idone, **kwargs):
        pass
    
    def on_agent_finish(self, finish, run_id, parent_run_id, **kwargs):
        pass

    def on_agent_error(self, error, *, run_id, parent_run_id, **kwargs):
        self._handle_error(error, run_id, parent_run_id, **kwargs)
    
    
    def get_parent_span(self, parent_run_id: Optional[str] = None):
        if parent_run_id is None:
            return None
        return self.span_mapping[parent_run_id]