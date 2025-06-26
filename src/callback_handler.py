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
    
    
    
    
    # for callback handler methods: https://python.langchain.com/api_reference/langchain/callbacks/langchain.callbacks.streaming_aiter.AsyncIteratorCallbackHandler.html#langchain.callbacks.streaming_aiter.AsyncIteratorCallbackHandler
    
class OpenTelemetryCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer):
        super().__init__()
        self.tracer = tracer
        self.spans: dict[UUID, SpanHolder] = {}
        
    
    def on_chat_model_start(self, serialized, messages, **kwargs):
        pass
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        pass
    
    def on_llm_new_token(self, token, **kwargs):
        pass
    
    def on_llm_end(self, response, **kwargs):
        pass
    
    def on_llm_error(self, error, **kwargs):
        pass

    def on_chain_start(self, serialized, inputs, **kwargs):
        pass 
    
    def on_chain_end(self, outputs, **kwargs):   
        pass
    
    def on_chain_error(self, error, run_id, parent_run_id, tags, **kwargs):
        pass
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        pass
    
    def on_tool_end(self, output, **kwargs):
        pass
    
    def on_tool_error(self, error, **kwargs):
        pass