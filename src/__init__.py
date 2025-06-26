from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
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

from .callback_handler import OpenTelemetryCallbackHandler


__all__ = ["OpenTelemetryCallbackHandler"]

_instruments = ("langchain >= 0.1.0",)

class LangChainInstrumentor(BaseInstrumentor):
    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        otelCallbackHandler = OpenTelemetryCallbackHandler(tracer)
        
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(otelCallbackHandler),
        )
    def _uninstrument(self, **kwargs):
        unwrap("langchain_core.callbacks", "BaseCallbackManager.__init__")
        if hasattr(self, "_wrapped"):
            for module, name in self._wrapped:
                unwrap(module, name)
        self.handler = None
    
    
class _BaseCallbackManagerInitWrapper:
    def __init__(self, callback_manager: "OpenTelemetryCallbackHandler"):
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