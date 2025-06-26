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

_instruments = ("langchain >= 0.1.0",)

class LangChainInstrumentor(BaseInstrumentor):
  
  

    def _instrument(self, **kwargs):
        # pass
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        otelCallbackHandler = OpenTelemetryCallbackHandler(tracer)
        traceloopCallbackHandler = TraceloopCallbackHandler(
            tracer)
    def _uninstrument(self, **kwargs):
        pass