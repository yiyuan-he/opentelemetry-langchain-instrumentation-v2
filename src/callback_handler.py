from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

_instruments = ("langchain >= 0.1.0",)

class LangChainInstrumentor(BaseInstrumentor):
  pass