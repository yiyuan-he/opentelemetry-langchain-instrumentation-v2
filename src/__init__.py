import logging
from typing import Collection
from opentelemetry.instrumentation.langchain.config import Config
from wrapt import wrap_function_wrapper

from opentelemetry.trace import get_tracer

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.instrumentation.langchain.utils import is_package_available


from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)
from opentelemetry.trace.propagation import set_span_in_context

from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)

from opentelemetry.metrics import get_meter
from opentelemetry.semconv_ai import Meters



logger = logging.getLogger(__name__)

_instruments = ("langchain-core > 0.1.0", )


class LangChainInstrumentor(BaseInstrumentor):
    """Main entry point for instrumentation"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.handler = None
        self.wrapped = None
        # Config.exception_logger = exception_logger
        # self.disable_trace_context_propagation = disable_trace_context_propagation


    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments


    def _instrument(self, **kwargs):
        # Inject callback handler into LangChain
        # pass
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        
        self.handler = TraceloopCallbackHandler(tracer_provider=tracer_provider)
        
        self._wrapped = [("langchain_core.callbacks", "BaseCallbackManager.__init__")]
        
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(self.handler),
        )
        
        # below code is for context propagation
        
        # if not self.disable_trace_context_propagation:
        #     self._wrap_openai_functions_for_tracing(self.handler)
    
    def _uninstrument(self, **kwargs):
        # Remove callback handler
        
        unwrap("langchain_core.callbacks", "BaseCallbackManager.__init__")
        if hasattr(self, "_wrapped"):
            for module, name in self._wrapped:
                unwrap(module, name)
        self.handler = None
    
    
    
class _BaseCallbackManagerInitWrapper:
    def __init__(self, callback_manager: "TraceloopCallbackHandler"):
        self._callback_manager = callback_manager

    def __call__(
        self,
        wrapped,
        instance,
        args,
        kwargs,
    ) -> None:
        wrapped(*args, **kwargs)
        for handler in instance.inheritable_handlers:
            if isinstance(handler, type(self._callback_manager)):
                break
        else:
            instance.add_handler(self._callback_manager, True)