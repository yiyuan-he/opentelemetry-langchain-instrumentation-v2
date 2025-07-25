import unittest
from unittest.mock import Mock, patch
import uuid
from langchain_core.outputs import LLMResult, Generation

from opentelemetry.trace import SpanKind, Status, StatusCode

from opentelemetry.instrumentation.langchain_v2.callback_handler import (
    OpenTelemetryCallbackHandler,
    SpanHolder,
    _set_request_params,
    _set_span_attribute,
    _sanitize_metadata_value
)
from opentelemetry.instrumentation.langchain_v2 import (
    LangChainInstrumentor,
    _BaseCallbackManagerInitWrapper,
    _instruments
)
from opentelemetry.instrumentation.langchain_v2.span_attributes import Span_Attributes, GenAIOperationValues

from opentelemetry.instrumentation.langchain_v2.callback_handler import _set_request_params

class TestOpenTelemetryHelperFunctions(unittest.TestCase):
    """Test the helper functions in the callback handler module."""
    
    def test_set_span_attribute(self):
        mock_span = Mock()
        
        _set_span_attribute(mock_span, "test.attribute", "test_value")
        mock_span.set_attribute.assert_called_once_with("test.attribute", "test_value")
        
        mock_span.reset_mock()

        _set_span_attribute(mock_span, "test.attribute", None)
        mock_span.set_attribute.assert_not_called()

        _set_span_attribute(mock_span, "test.attribute", "")
        mock_span.set_attribute.assert_not_called()
    
    def test_sanitize_metadata_value(self):
        self.assertEqual(_sanitize_metadata_value(None), None)
        self.assertEqual(_sanitize_metadata_value(True), True)
        self.assertEqual(_sanitize_metadata_value("string"), "string")
        self.assertEqual(_sanitize_metadata_value(123), 123)
        self.assertEqual(_sanitize_metadata_value(1.23), 1.23)
        
        self.assertEqual(_sanitize_metadata_value([1, "two", 3.0]), ["1", "two", "3.0"])
        self.assertEqual(_sanitize_metadata_value((1, "two", 3.0)), ["1", "two", "3.0"])
        
        class TestClass:
            def __str__(self):
                return "test_class"
        
        self.assertEqual(_sanitize_metadata_value(TestClass()), "test_class")

    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler._set_span_attribute")
    def test_set_request_params(self, mock_set_span_attribute):
        mock_span = Mock()
        mock_span_holder = Mock(spec=SpanHolder)
        
        kwargs = {"model_id": "gpt-4", "temperature": 0.7, "max_tokens": 100, "top_p": 0.9}
        _set_request_params(mock_span, kwargs, mock_span_holder)
        
        self.assertEqual(mock_span_holder.request_model, "gpt-4")
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_REQUEST_MODEL, "gpt-4")
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_RESPONSE_MODEL, "gpt-4")
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_REQUEST_TEMPERATURE, 0.7)
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_REQUEST_MAX_TOKENS, 100)
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_REQUEST_TOP_P, 0.9)
        
        mock_set_span_attribute.reset_mock()
        mock_span_holder.reset_mock()
        
        kwargs = {
            "invocation_params": {
                "model_id": "gpt-3.5-turbo",
                "temperature": 0.5,
                "max_tokens": 50
            }
        }
        _set_request_params(mock_span, kwargs, mock_span_holder)
        
        self.assertEqual(mock_span_holder.request_model, "gpt-3.5-turbo")
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_REQUEST_MODEL, "gpt-3.5-turbo")


class TestOpenTelemetryCallbackHandler(unittest.TestCase):
    """Test the OpenTelemetryCallbackHandler class."""
    
    def setUp(self):
        self.mock_tracer = Mock()
        self.mock_span = Mock()
        self.mock_tracer.start_span.return_value = self.mock_span
        self.handler = OpenTelemetryCallbackHandler(self.mock_tracer)
        self.run_id = uuid.uuid4()
        self.parent_run_id = uuid.uuid4()
    
    def test_init(self):
        """Test the initialization of the handler."""
        handler = OpenTelemetryCallbackHandler(self.mock_tracer)
        self.assertEqual(handler.tracer, self.mock_tracer)
        self.assertEqual(handler.span_mapping, {})
    
    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler.context_api")
    def test_create_span(self, mock_context_api):
        """Test the _create_span method."""
        mock_context_api.get_value.return_value = {}
        mock_context_api.set_value.return_value = {}
        mock_context_api.attach.return_value = None
        
        span = self.handler._create_span(
            run_id=self.run_id,
            parent_run_id=None,
            span_name="test_span",
            kind=SpanKind.INTERNAL,
            metadata={"key": "value"}
        )
        
        self.mock_tracer.start_span.assert_called_once_with("test_span", kind=SpanKind.INTERNAL)
        self.assertEqual(span, self.mock_span)
        self.assertIn(self.run_id, self.handler.span_mapping)
        
        self.mock_tracer.reset_mock()
        
        parent_span = Mock()
        self.handler.span_mapping[self.parent_run_id] = SpanHolder(
            parent_span, [], time.time(), "model-id"
        )
        
    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler.context_api")
    def test_on_llm_start_and_end(self, mock_context_api):
        """Test the on_llm_start and on_llm_end methods together."""
        mock_context_api.get_value.return_value = False
        serialized = {"name": "test_llm"}
        prompts = ["Hello, world!"]
        kwargs = {
            "invocation_params": {
                "model_id": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 100
            }
        }
        
        original_create_llm_span = self.handler._create_llm_span
        self.handler._create_llm_span = Mock(return_value=self.mock_span)
        
        self.handler.on_llm_start(
            serialized=serialized,
            prompts=prompts,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs
        )
        
        self.handler._create_llm_span.assert_called_once_with(
            self.run_id, 
            self.parent_run_id, 
            "gpt-4",  
            GenAIOperationValues.TEXT_COMPLETION,
            kwargs
        )
            
        self.handler.span_mapping[self.run_id] = SpanHolder(
            self.mock_span, [], time.time(), "gpt-4"
        )
        
        llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20
            },
            "model_name": "gpt-4",
            "id": "response-123"
        }
        generations = [[Generation(text="This is a test response")]]
        response = LLMResult(generations=generations, llm_output=llm_output)
     
        with patch("opentelemetry.instrumentation.langchain_v2.callback_handler._set_span_attribute") as mock_set_attribute:
            with patch.object(self.handler, '_end_span') as mock_end_span:
                self.handler.on_llm_end(
                    response=response,
                    run_id=self.run_id,
                    parent_run_id=self.parent_run_id
                )
                
                print("\nAll calls to mock_set_attribute:")
                for i, call in enumerate(mock_set_attribute.call_args_list):
                    args, kwargs = call
                    print(f"Call {i+1}:", args, kwargs)
                    
                mock_set_attribute.assert_any_call(self.mock_span, Span_Attributes.GEN_AI_RESPONSE_MODEL, "gpt-4")
                mock_set_attribute.assert_any_call(self.mock_span, Span_Attributes.GEN_AI_RESPONSE_ID, "response-123")
                mock_set_attribute.assert_any_call(self.mock_span, Span_Attributes.GEN_AI_USAGE_INPUT_TOKENS, 10)
                mock_set_attribute.assert_any_call(self.mock_span, Span_Attributes.GEN_AI_USAGE_OUTPUT_TOKENS, 20)
                
                mock_end_span.assert_not_called()
        
        self.handler._create_llm_span = original_create_llm_span
    
    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler.context_api")
    def test_on_llm_error(self, mock_context_api):
        """Test the on_llm_error method."""
        mock_context_api.get_value.return_value = False
        self.handler.span_mapping[self.run_id] = SpanHolder(
            self.mock_span, [], time.time(), "gpt-4"
        )
        error = ValueError("Test error")
        
        self.handler._handle_error(
            error=error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id
        )

        self.mock_span.set_status.assert_called_once()
        args, _ = self.mock_span.set_status.call_args
        self.assertEqual(args[0].status_code, StatusCode.ERROR)


        self.mock_span.record_exception.assert_called_once_with(error)
        self.mock_span.end.assert_called_once()

    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler.context_api")
    def test_on_chain_start_end(self, mock_context_api):
        """Test the on_chain_start and on_chain_end methods."""
        mock_context_api.get_value.return_value = False
        serialized = {"name": "test_chain"}
        inputs = {"query": "What is the capital of France?"}
        
        with patch.object(self.handler, '_create_span', return_value=self.mock_span) as mock_create_span:
            self.handler.on_chain_start(
                serialized=serialized,
                inputs=inputs,
                run_id=self.run_id,
                parent_run_id=self.parent_run_id
            )
            
            mock_create_span.assert_called_once()
            self.mock_span.set_attribute.assert_called_once_with("chain.input", str(inputs))
        
        outputs = {"result": "Paris"}
        self.handler.span_mapping[self.run_id] = SpanHolder(
            self.mock_span, [], time.time(), "gpt-4"
        )
        
        with patch.object(self.handler, '_end_span') as mock_end_span:
            self.handler.on_chain_end(
                outputs=outputs,
                run_id=self.run_id,
                parent_run_id=self.parent_run_id
            )

            self.mock_span.set_attribute.assert_called_with("chain.output", str(outputs))
            mock_end_span.assert_called_once_with(self.mock_span, self.run_id)

    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler.context_api")
    def test_on_tool_start_end(self, mock_context_api):
        """Test the on_tool_start and on_tool_end methods."""
        mock_context_api.get_value.return_value = False
        serialized = {
            "name": "test_tool", 
            "id": "tool-123",
            "description": "A test tool"
        }
        input_str = "What is 2 + 2?"
        
        with patch.object(self.handler, '_create_span', return_value=self.mock_span) as mock_create_span:
            with patch.object(self.handler, '_get_name_from_callback', return_value="test_tool") as mock_get_name:
                self.handler.on_tool_start(
                    serialized=serialized,
                    input_str=input_str,
                    run_id=self.run_id,
                    parent_run_id=self.parent_run_id
                )
                
                mock_create_span.assert_called_once()
                mock_get_name.assert_called_once()

                
                self.mock_span.set_attribute.assert_any_call("tool.input", input_str)
                self.mock_span.set_attribute.assert_any_call(Span_Attributes.GEN_AI_TOOL_CALL_ID, "tool-123")
                self.mock_span.set_attribute.assert_any_call(Span_Attributes.GEN_AI_TOOL_DESCRIPTION, "A test tool")
                self.mock_span.set_attribute.assert_any_call(Span_Attributes.GEN_AI_TOOL_NAME, "test_tool")
    
        
        output = "The answer is 4"
        
        self.handler.span_mapping[self.run_id] = SpanHolder(
            self.mock_span, [], time.time(), "gpt-4"
        )
        
        with patch.object(self.handler, '_end_span') as mock_end_span:
            with patch.object(self.handler, '_get_span', return_value=self.mock_span) as mock_get_span:
                self.handler.on_tool_end(
                    output=output,
                    run_id=self.run_id,
                    parent_run_id=self.parent_run_id
                )
                
                mock_get_span.assert_called_once_with(self.run_id)
                self.mock_span.set_attribute.assert_any_call("tool.output", str(output))
                mock_end_span.assert_called_once_with(self.mock_span, self.run_id)
            
            
class TestLangChainInstrumentor(unittest.TestCase):
    """Test the LangChainInstrumentor class."""
    
    def setUp(self):
        self.instrumentor = LangChainInstrumentor()
    
    def test_instrumentation_dependencies(self):
        """Test that instrumentation_dependencies returns the correct dependencies."""
        result = self.instrumentor.instrumentation_dependencies()
        self.assertEqual(result, _instruments)
        self.assertEqual(result, ("langchain >= 0.1.0",))
    
    @patch("opentelemetry.instrumentation.langchain_v2.get_tracer")
    @patch("opentelemetry.instrumentation.langchain_v2.wrap_function_wrapper")
    def test_instrument(self, mock_wrap, mock_get_tracer):
        """Test the _instrument method."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        tracer_provider = Mock()
        
        self.instrumentor._instrument(tracer_provider=tracer_provider)
        
        mock_get_tracer.assert_called_once()
        mock_wrap.assert_called_once()
        
        module = mock_wrap.call_args[1]["module"]
        name = mock_wrap.call_args[1]["name"]
        wrapper = mock_wrap.call_args[1]["wrapper"]
        
        self.assertEqual(module, "langchain_core.callbacks")
        self.assertEqual(name, "BaseCallbackManager.__init__")
        self.assertIsInstance(wrapper, _BaseCallbackManagerInitWrapper)
        self.assertIsInstance(wrapper.callback_handler, OpenTelemetryCallbackHandler)
    
    @patch("opentelemetry.instrumentation.langchain_v2.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test the _uninstrument method."""
        self.instrumentor._wrapped = [
            ("module1", "function1"),
            ("module2", "function2")
        ]
        self.instrumentor.handler = Mock()
        
        self.instrumentor._uninstrument()

        mock_unwrap.assert_any_call("langchain_core.callbacks", "BaseCallbackManager.__init__")
        mock_unwrap.assert_any_call("module1", "function1")
        mock_unwrap.assert_any_call("module2", "function2")
        self.assertIsNone(self.instrumentor.handler)


class TestBaseCallbackManagerInitWrapper(unittest.TestCase):
    """Test the _BaseCallbackManagerInitWrapper class."""
    
    def test_init_wrapper_add_handler(self):
        """Test that the wrapper adds the handler to the callback manager."""
        mock_handler = Mock(spec=OpenTelemetryCallbackHandler)
        
        wrapper_instance = _BaseCallbackManagerInitWrapper(mock_handler)
        
        original_func = Mock()
        instance = Mock()
        instance.inheritable_handlers = []
        
        wrapper_instance(original_func, instance, [], {})
        
        original_func.assert_called_once_with() 
        instance.add_handler.assert_called_once_with(mock_handler, True)
    
    def test_init_wrapper_handler_already_exists(self):
        """Test that the wrapper doesn't add a duplicate handler."""
        mock_handler = Mock(spec=OpenTelemetryCallbackHandler)
        
        wrapper_instance = _BaseCallbackManagerInitWrapper(mock_handler)
        
        original_func = Mock()
        instance = Mock()
        
        mock_tracer = Mock()
        existing_handler = OpenTelemetryCallbackHandler(mock_tracer) 
        instance.inheritable_handlers = [existing_handler]
        
        wrapper_instance(original_func, instance, [], {})
        
        original_func.assert_called_once_with() 
        instance.add_handler.assert_not_called()


if __name__ == "__main__":
    import time 
    unittest.main()
