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


# below dataclass stolen from openLLMetry
@dataclass
class SpanHolder:
    span: Span
    token: Any
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

    _set_span_attribute(span, Span_Attributes.GENAI_REQUEST_MODEL, model)
    # response is not available for LLM requests (as opposed to chat)
    _set_span_attribute(span, Span_Attributes.GENAI_RESPONSE_MODEL, model)

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
        # _set_span_attribute(
        #     span,
        #     f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.name",
        #     tool_function.get("name"),
        # )
        
        # ///////////////////////// STOP HERE: what are we doing about this indexing?
        _set_span_attribute(
            span,
            f"{Span_Attributes.GEN_AI_TOOL_CALL_ID}.{i}.name",
            tool_function.get("name"),
        )
        
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.description",
            tool_function.get("description"),
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.parameters",
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

    if should_send_prompts():
        for i, msg in enumerate(prompts):
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                "user",
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                msg,
            )

class OpenTelemetryCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer):
        super().__init__()
        self.tracer = tracer
        self.span_mapping: dict[UUID, SpanHolder] = {}
    
    def _get_span(self, run_id: UUID) -> Span:
        return self.spans[run_id].span

    def _end_span(self, span: Span, run_id: UUID) -> None:
        for child_id in self.spans[run_id].children:
            child_span = self.spans[child_id].span
            if child_span.end_time is None:  # avoid warning on ended spans
                child_span.end()
        span.end()
        
        
    def _create_llm_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        operation_name: GenAIOperationValues,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        workflow_name = self.get_workflow_name(parent_run_id) # implement
        entity_path = self.get_entity_path(parent_run_id) # implement

        span = self._create_span(  # implement _create_span
            run_id,
            parent_run_id,
            f"{name}.{operation_name.value}",
            kind=SpanKind.CLIENT,
            workflow_name=workflow_name,
            entity_path=entity_path,
            metadata=metadata,
        )
        _set_span_attribute(span, Span_Attributes.GENAI_SYSTEM, "Langchain")
        _set_span_attribute(span, GenAIOperationValues.GENAI_OPERATION_NAME, operation_name.value)
        
        return span
    
    def on_chat_model_start(self, serialized, messages, run_id, parent_run_id, **kwargs):
        pass
    
    def on_llm_start(self, serialized, prompts, run_id, parent_run_id, **kwargs):
        # if parent_run_id and parent_run_id in self.span_mapping:
        #     parent_span = self.span_mapping[parent_run_id].span
            
        #     with self.tracer.start_as_current_span() as span:
        #     # span = self.tracer.start_as_current_span(name, context=set_span_in_context(parent_span))
        #         pass
        # else:
        #     with self.tracer.start_as_current_span() as span:
        #         self.span_mapping[run_id] = SpanInfo(span=span, prompts=prompts)
        #         span.set_attribute("run_id", run_id)
        #         if parent_run_id:
        #             span.set_attribute("parent_run_id", parent_run_id)
        #         # Additional span setup as needed
        #         self._set_span_attributes(span, serialized)
        
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = self._get_name_from_callback(serialized, kwargs=kwargs) # implement
        span = self._create_llm_span(
            run_id, parent_run_id, name, GenAIOperationValues.TEXT_COMPLETION
        )
        
        _set_llm_request(span, serialized, prompts, kwargs, self.spans[run_id])

        
        
                
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