import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks import (
    BaseCallbackHandler,
)

from opentelemetry.context.context import Context
from opentelemetry.trace import SpanKind, set_span_in_context, Tracer
from opentelemetry.trace.span import Span
from opentelemetry.util.types import AttributeValue
from uuid import UUID

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
    TraceloopSpanKindValues,
)


from opentelemetry.instrumentation.langchain_v2.span_attributes import Span_Attributes, GenAIOperationValues
from src.opentelemetry.instrumentation.langchain_v2.utils import dont_throw


# below dataclass stolen from openLLMetry
@dataclass
class SpanHolder:
    span: Span
    token: Any # potentially can remove token
    context: Context
    children: list[UUID]
    workflow_name: str
    entity_name: str
    entity_path: str
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
        # ///////////////////////// what are we doing about this indexing?
        # ///////////////////////// Current implementation: gen_ai.tool.name.index
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
        
        ########## replaced SpanAttributes.LLM_REQUEST_FUNCTIONS.{i}.parameters with this
        ########## Not a 1 to 1 replacement
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

            # ////////// below 2 are not OTel
            # _set_span_attribute(span, SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflow_name)
            # _set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path)

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
                
    def on_chat_model_start(self, serialized, messages, run_id, parent_run_id, **kwargs):
        pass
    
    
    @dont_throw
    def on_llm_start(self, serialized, prompts, run_id, parent_run_id, **kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = self._get_name_from_callback(serialized, kwargs=kwargs)
        span = self._create_llm_span(
            run_id, parent_run_id, name, GenAIOperationValues.TEXT_COMPLETION
        )
        
        _set_llm_request(span, serialized, prompts, kwargs, self.span_mapping[run_id])

        
        
                
    # def on_llm_new_token(self, token, **kwargs):
    #     pass
    
    def on_llm_end(self, response, run_id, parent_run_id, **kwargs):
        pass
    
    def on_llm_error(self, error, run_id, parent_run_id, **kwargs):
        pass

    def on_chain_start(self, serialized, inputs, run_id, parent_run_id, **kwargs):
        pass 
    
    def on_chain_end(self, outputs, run_id, parent_run_id, **kwargs):   
        pass
    
    def on_chain_error(self, error, run_id, parent_run_id, tags, **kwargs):
        pass
    
    def on_tool_start(self, serialized, input_str, run_id, parent_run_id, **kwargs):
        pass
    
    def on_tool_end(self, output, run_id, parent_run_id, **kwargs):
        pass
    
    def on_tool_error(self, error, run_id, parent_run_id, **kwargs):
        pass
    
    def on_agent_action(self, action, run_id, parent_run_idone, **kwargs):
        pass
    
    def on_agent_finish(self, finish, run_id, parent_run_id, **kwargs):
        pass

    def on_agent_error(self, error, run_id, parent_run_id, **kwargs):
        pass
    
    
    
    
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