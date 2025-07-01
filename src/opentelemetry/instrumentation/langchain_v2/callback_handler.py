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


# below dataclass stolen from openLLMetry
@dataclass
class SpanHolder:
    span: Span
    token: Any # potentially can remove token *high
    context: Context
    children: list[UUID]
    workflow_name: str # potentially can remove *high
    entity_name: str 
    entity_path: str # potentially can remove *low
    start_time: float = field(default_factory=time.time)
    request_model: Optional[str] = None
    
def _message_type_to_role(message_type: str) -> str:
    if message_type == "human":
        return "user"
    elif message_type == "system":
        return "system"
    elif message_type == "ai":
        return "assistant"
    elif message_type == "tool":
        return "tool"
    else:
        return "unknown"
    
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

    should_trace = context_api.get_value("override_enable_content_tracing") or True
    if should_trace:
        for i, msg in enumerate(prompts):
            
            #  /////// Below span attributes are labeled as "DEPRECATED": Deprecated, use Event API to report prompt contents.
            
            # _set_span_attribute(
            #     span,
            #     f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
            #     "user",
            # )
            # _set_span_attribute(
            #     span,
            #     f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
            #     msg,
            # )
            pass
        
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
            
        i = 0
        for message in messages:
            for msg in message:
                # _set_span_attribute(
                #     span,
                #     f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                #     _message_type_to_role(msg.type),
                # ) # !!!!!!!deprecated
                
                tool_calls = (
                    msg.tool_calls
                    if hasattr(msg, "tool_calls")
                    else msg.additional_kwargs.get("tool_calls")
                )

                if tool_calls:
                #     _set_chat_tool_calls(span, f"{SpanAttributes.LLM_PROMPTS}.{i}", tool_calls)
                # !!!!!!!!deprecated
                    pass 
                
                else:
                    content = (
                        msg.content
                        if isinstance(msg.content, str)
                        else json.dumps(msg.content, cls=CallbackFilteredJSONEncoder)
                    )
                    # _set_span_attribute(
                    #     span,
                    #     f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    #     content,
                    # ) #!!!!!!!deprecated

                # if msg.type == "tool" and hasattr(msg, "tool_call_id"):
                #     _set_span_attribute(
                #         span,
                #         f"{SpanAttributes.LLM_PROMPTS}.{i}.tool_call_id",
                #         msg.tool_call_id,
                #     ) # !!!!!!!deprecated

                i += 1


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
            workflow_name: str = "",
            entity_name: str = "",
            entity_path: str = "",
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

            # ////////// not OTel, cant find usage so potentially remove later
            token = context_api.attach(
                context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
            )

            self.span_mapping[run_id] = SpanHolder(
                span, token, None, [], workflow_name, entity_name, entity_path
            )

            if parent_run_id is not None and parent_run_id in self.span_mapping:
                self.span_mapping[parent_run_id].children.append(run_id)

            return span

    def _create_task_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        # kind: TraceloopSpanKindValues,
        kind: SpanKind,
        workflow_name: str,
        entity_name: str = "",
        entity_path: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        span_name = f"{name}.{kind.value}"
        span = self._create_span(
            run_id,
            parent_run_id,
            span_name,
            workflow_name=workflow_name,
            entity_name=entity_name,
            entity_path=entity_path,
            metadata=metadata,
        )

        # _set_span_attribute(span, SpanAttributes.TRACELOOP_SPAN_KIND, kind.value)
        # _set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        
         # Replace Traceloop attributes with standard semantic conventions
        _set_span_attribute(span, "app.workflow.name", workflow_name)
        _set_span_attribute(span, "app.entity.name", entity_name)
        
        # Add span kind as an attribute (since we're using the SpanKind enum)
        if kind == SpanKind.INTERNAL:
            _set_span_attribute(span, "app.component.type", "workflow" if parent_run_id is None else "task")

        return span
        
        
    def _create_llm_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        operation_name: GenAIOperationValues,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        workflow_name = self.get_workflow_name(parent_run_id) 
        entity_path = self.get_entity_path(parent_run_id) 

        span = self._create_span(
            run_id,
            parent_run_id,
            f"{name}.{operation_name.value}",
            kind=SpanKind.CLIENT,
            workflow_name=workflow_name,
            entity_path=entity_path,
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
            total_tokens = token_usage.get("total_tokens") or (
                prompt_tokens + completion_tokens
            )
            
            _set_span_attribute(
                span, Span_Attributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
            )
        
            _set_span_attribute(
                span, Span_Attributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
            )
        
            # Record token usage metrics
            if prompt_tokens > 0:
                self.token_histogram.record( # potentially can remove token_histogram
                    prompt_tokens,
                    attributes={
                        Span_Attributes.GEN_AI_SYSTEM: "Langchain",
                        Span_Attributes.GEN_AI_TOKEN_TYPE: "input",
                        Span_Attributes.GEN_AI_RESPONSE_MODEL: model_name or "unknown",
                    },
                )

            if completion_tokens > 0:
                self.token_histogram.record( # potentially can remove token_histogram
                    completion_tokens,
                    attributes={
                        Span_Attributes.GEN_AI_SYSTEM: "Langchain",
                        Span_Attributes.GEN_AI_TOKEN_TYPE: "output",
                        Span_Attributes.GEN_AI_RESPONSE_MODEL: model_name or "unknown",
                    },
                )

        _set_chat_response(span, response)
        self._end_span(span, run_id)

        # Record duration
        duration = time.time() - self.span_mapping[run_id].start_time
        self.duration_histogram.record(
            duration,
            attributes={
                Span_Attributes.GEN_AI_SYSTEM: "Langchain",
                Span_Attributes.GEN_AI_RESPONSE_MODEL: model_name or "unknown",
            },
        )
    
    @dont_throw
    def on_llm_error(self, error, *, run_id, parent_run_id, **kwargs):
        self._handle_error(error, run_id, parent_run_id, **kwargs)

    @dont_throw
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id, tags, metadata, **kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        workflow_name = ""
        entity_path = ""

        name = self._get_name_from_callback(serialized, **kwargs)
        # kind = (
        #     TraceloopSpanKindValues.WORKFLOW
        #     if parent_run_id is None or parent_run_id not in self.spans
        #     else TraceloopSpanKindValues.TASK
        # )
        
        is_top_level = parent_run_id is None or parent_run_id not in self.span_mapping
        
        
        kind = SpanKind.INTERNAL

        if is_top_level:
            workflow_name = name
        else:
            workflow_name = self.get_workflow_name(parent_run_id)
            entity_path = self.get_entity_path(parent_run_id)

        span = self._create_task_span(
            run_id,
            parent_run_id,
            name,
            kind,
            workflow_name,
            name,
            entity_path,
            metadata,
        )
        should_trace = context_api.get_value("override_enable_content_tracing") or True
        if should_trace:
            _set_span_attribute(
                span,
                # SpanAttributes.TRACELOOP_ENTITY_INPUT,
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
                # SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                "app.entity.output",
                json.dumps(
                    {"outputs": outputs, "kwargs": kwargs},
                    cls=CallbackFilteredJSONEncoder,
                ),
            )

        self._end_span(span, run_id)
        if parent_run_id is None:
            context_api.attach(
                context_api.set_value(
                    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, False # another instance of something we need to address
                )
            )
    
    @dont_throw
    def on_chain_error(self, error, run_id, parent_run_id, tags, **kwargs):
        self._handle_error(error, run_id, parent_run_id, **kwargs)
        
    @dont_throw
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id, tags, metadata, inputs, **kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        workflow_name = self.get_workflow_name(parent_run_id)
        entity_path = self.get_entity_path(parent_run_id)

        span = self._create_task_span(
            run_id,
            parent_run_id,
            name,
            # TraceloopSpanKindValues.TOOL,
            SpanKind.INTERNAL,
            workflow_name,
            name,
            entity_path,
        )
        # if should_send_prompts():
        should_trace = context_api.get_value("override_enable_content_tracing") or True
        if should_trace:
            _set_span_attribute(
                span,
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
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
                # SpanAttributes.TRACELOOP_ENTITY_OUTPUT, # investigate this, do we even need it?
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


    def get_workflow_name(self, parent_run_id: str):
        parent_span = self.get_parent_span(parent_run_id)

        if parent_span is None:
            return ""

        return parent_span.workflow_name
        
    def get_entity_path(self, parent_run_id: str):
        parent_span = self.get_parent_span(parent_run_id)

        if parent_span is None:
            return ""
        elif (
            parent_span.entity_path == ""
            and parent_span.entity_name == parent_span.workflow_name
        ):
            return ""
        elif parent_span.entity_path == "":
            return f"{parent_span.entity_name}"
        else:
            return f"{parent_span.entity_path}.{parent_span.entity_name}"