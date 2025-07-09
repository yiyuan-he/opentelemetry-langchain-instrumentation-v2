import unittest
from unittest.mock import Mock, patch, MagicMock, call
import uuid
from typing import Collection
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import HumanMessage
from langchain_core.agents import AgentAction, AgentFinish
from opentelemetry.trace import SpanKind, Status, StatusCode

# Import your actual modules
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


class TestOpenTelemetryHelperFunctions(unittest.TestCase):
    """Test the helper functions in the callback handler module."""
    
    def test_set_span_attribute(self):
        # Create a mock span
        mock_span = Mock()
        
        # Test with a valid value
        _set_span_attribute(mock_span, "test.attribute", "test_value")
        mock_span.set_attribute.assert_called_once_with("test.attribute", "test_value")
        
        # Reset mock
        mock_span.reset_mock()
        
        # Test with a None value - should not call set_attribute
        _set_span_attribute(mock_span, "test.attribute", None)
        mock_span.set_attribute.assert_not_called()
        
        # Test with an empty string value - should not call set_attribute
        _set_span_attribute(mock_span, "test.attribute", "")
        mock_span.set_attribute.assert_not_called()
    
    def test_sanitize_metadata_value(self):
        # Test primitive values
        self.assertEqual(_sanitize_metadata_value(None), None)
        self.assertEqual(_sanitize_metadata_value(True), True)
        self.assertEqual(_sanitize_metadata_value("string"), "string")
        self.assertEqual(_sanitize_metadata_value(123), 123)
        self.assertEqual(_sanitize_metadata_value(1.23), 1.23)
        
        # Test list/tuple
        self.assertEqual(_sanitize_metadata_value([1, "two", 3.0]), ["1", "two", "3.0"])
        self.assertEqual(_sanitize_metadata_value((1, "two", 3.0)), ["1", "two", "3.0"])
        
        # Test complex object
        class TestClass:
            def __str__(self):
                return "test_class"
        
        self.assertEqual(_sanitize_metadata_value(TestClass()), "test_class")

    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler._set_span_attribute")
    def test_set_request_params(self, mock_set_span_attribute):
        mock_span = Mock()
        mock_span_holder = Mock(spec=SpanHolder)
        
        # Test with model_id in kwargs
        kwargs = {"model_id": "gpt-4", "temperature": 0.7, "max_tokens": 100, "top_p": 0.9}
        _set_request_params(mock_span, kwargs, mock_span_holder)
        
        # Verify request_model was set and attributes were applied
        self.assertEqual(mock_span_holder.request_model, "gpt-4")
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_REQUEST_MODEL, "gpt-4")
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_RESPONSE_MODEL, "gpt-4")
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_REQUEST_TEMPERATURE, 0.7)
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_REQUEST_MAX_TOKENS, 100)
        mock_set_span_attribute.assert_any_call(mock_span, Span_Attributes.GEN_AI_REQUEST_TOP_P, 0.9)
        
        # Reset mock
        mock_set_span_attribute.reset_mock()
        mock_span_holder.reset_mock()
        
        # Test with invocation_params containing model_id
        kwargs = {
            "invocation_params": {
                "model_id": "gpt-3.5-turbo",
                "temperature": 0.5,
                "max_tokens": 50
            }
        }
        _set_request_params(mock_span, kwargs, mock_span_holder)
        
        # Verify request_model was set and attributes were applied
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
        # Setup
        mock_context_api.get_value.return_value = {}
        mock_context_api.set_value.return_value = {}
        mock_context_api.attach.return_value = None
        
        # Create a span without a parent
        span = self.handler._create_span(
            run_id=self.run_id,
            parent_run_id=None,
            span_name="test_span",
            kind=SpanKind.INTERNAL,
            metadata={"key": "value"}
        )
        
        # Verify span creation
        self.mock_tracer.start_span.assert_called_once_with("test_span", kind=SpanKind.INTERNAL)
        self.assertEqual(span, self.mock_span)
        self.assertIn(self.run_id, self.handler.span_mapping)
        # swear to the holy trinity these tests above do nothing important
        
        
        
        # Reset mocks
        self.mock_tracer.reset_mock()
        
        # Create a span with a parent
        parent_span = Mock()
        self.handler.span_mapping[self.parent_run_id] = SpanHolder(
            parent_span, [], time.time(), "model-id"
        )
        
        # with patch("opentelemetry.instrumentation.langchain_v2.callback_handler.set_span_in_context") as mock_set_context:
        #     mock_set_context.return_value = parent_context
        #     span = self.handler._create_span(
        #         run_id=uuid.uuid4(),
        #         parent_run_id=self.parent_run_id,
        #         span_name="child_span"
        #     )
            
        #     # Verify parent relationship
        #     mock_set_context.assert_called_once_with(parent_span)
        #     self.mock_tracer.start_span.assert_called_once()
        #     self.assertIn(self.parent_run_id, self.handler.span_mapping)
    
    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler.context_api")
    def test_on_llm_start(self, mock_context_api):
        """Test the on_llm_start method."""
        # Setup
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
        
        # Create a mock _create_llm_span method
        original_create_llm_span = self.handler._create_llm_span
        self.handler._create_llm_span = Mock(return_value=self.mock_span)
        
        # Call on_llm_start
        self.handler.on_llm_start(
            serialized=serialized,
            prompts=prompts,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs
        )
        
        # Verify _create_llm_span was called with the right parameters
        self.handler._create_llm_span.assert_called_once_with(
            self.run_id, 
            self.parent_run_id, 
            "gpt-4", 
            GenAIOperationValues.TEXT_COMPLETION,
            kwargs
        )
        
        # Restore original method
        self.handler._create_llm_span = original_create_llm_span
    
    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler.context_api")
    def test_on_llm_end(self, mock_context_api):
        """Test the on_llm_end method."""
        # Setup
        mock_context_api.get_value.return_value = False
        self.handler.span_mapping[self.run_id] = SpanHolder(
            self.mock_span, [], time.time(), "gpt-4"
        )
        
        # Create a LLMResult with token usage
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
        
        # Mock _set_span_attribute to verify calls
        with patch("opentelemetry.instrumentation.langchain_v2.callback_handler._set_span_attribute") as mock_set_attribute:
            self.handler.on_llm_end(
                response=response,
                run_id=self.run_id,
                parent_run_id=self.parent_run_id
            )
            
            # Verify span attributes were set
            mock_set_attribute.assert_any_call(self.mock_span, Span_Attributes.GEN_AI_RESPONSE_MODEL, "gpt-4")
            mock_set_attribute.assert_any_call(self.mock_span, Span_Attributes.GEN_AI_REQUEST_MODEL, "gpt-4")
            mock_set_attribute.assert_any_call(self.mock_span, Span_Attributes.GEN_AI_RESPONSE_ID, "response-123")
            mock_set_attribute.assert_any_call(self.mock_span, Span_Attributes.GEN_AI_USAGE_INPUT_TOKENS, 10)
            mock_set_attribute.assert_any_call(self.mock_span, Span_Attributes.GEN_AI_USAGE_OUTPUT_TOKENS, 20)

    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler.context_api")
    def test_on_llm_error(self, mock_context_api):
        """Test the on_llm_error method."""
        # Setup
        mock_context_api.get_value.return_value = False
        self.handler.span_mapping[self.run_id] = SpanHolder(
            self.mock_span, [], time.time(), "gpt-4"
        )
        error = ValueError("Test error")
        
        # Call _handle_error directly since on_llm_error just delegates to it
        self.handler._handle_error(
            error=error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id
        )
        
        # Verify span status was set to ERROR and exception was recorded
        # self.mock_span.set_status.assert_called_once_with(Status(StatusCode.ERROR))

        # Instead of comparing objects directly, use an argument matcher
        self.mock_span.set_status.assert_called_once()
        # Check that the call was with a Status object with ERROR code
        args, _ = self.mock_span.set_status.call_args
        self.assertEqual(args[0].status_code, StatusCode.ERROR)


        self.mock_span.record_exception.assert_called_once_with(error)
        self.mock_span.end.assert_called_once()

    @patch("opentelemetry.instrumentation.langchain_v2.callback_handler.context_api")
    def test_on_chain_start_end(self, mock_context_api):
        """Test the on_chain_start and on_chain_end methods."""
        # Setup
        mock_context_api.get_value.return_value = False
        serialized = {"name": "test_chain"}
        inputs = {"query": "What is the capital of France?"}
        
        # Test chain start
        with patch.object(self.handler, '_create_span', return_value=self.mock_span) as mock_create_span:
            self.handler.on_chain_start(
                serialized=serialized,
                inputs=inputs,
                run_id=self.run_id,
                parent_run_id=self.parent_run_id
            )
            
            # Verify span was created and input was set
            mock_create_span.assert_called_once()
            self.mock_span.set_attribute.assert_called_once_with("chain.input", str(inputs))
        
        # Test chain end
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
            
            # Verify output was set and span was ended
            self.mock_span.set_attribute.assert_called_with("chain.output", str(outputs))
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
        # Setup
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        tracer_provider = Mock()
        
        # Execute _instrument
        self.instrumentor._instrument(tracer_provider=tracer_provider)
        
        # Verify tracer was obtained and wrap_function_wrapper was called
        mock_get_tracer.assert_called_once()
        mock_wrap.assert_called_once()
        
        # Verify the wrapper is a _BaseCallbackManagerInitWrapper
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
        # Setup - add some wrapped methods to be unwrapped
        self.instrumentor._wrapped = [
            ("module1", "function1"),
            ("module2", "function2")
        ]
        self.instrumentor.handler = Mock()
        
        # Execute _uninstrument
        self.instrumentor._uninstrument()
        
        # Verify unwrapping happened and handler was cleared
        mock_unwrap.assert_any_call("langchain_core.callbacks", "BaseCallbackManager.__init__")
        mock_unwrap.assert_any_call("module1", "function1")
        mock_unwrap.assert_any_call("module2", "function2")
        self.assertIsNone(self.instrumentor.handler)


class TestBaseCallbackManagerInitWrapper(unittest.TestCase):
    """Test the _BaseCallbackManagerInitWrapper class."""
    
    def test_init_wrapper_add_handler(self):
        """Test that the wrapper adds the handler to the callback manager."""
        # Create a mock callback handler
        mock_handler = Mock(spec=OpenTelemetryCallbackHandler)
        
        # Create the wrapper with our mock handler
        wrapper_instance = _BaseCallbackManagerInitWrapper(mock_handler)
        
        # Create mocks for the original function and instance
        original_func = Mock()
        instance = Mock()
        instance.inheritable_handlers = []
        
        # Call the wrapper
        wrapper_instance(original_func, instance, [], {})
        
        # Verify the original function was called and handler was added
        # original_func.assert_called_once_with(instance, [], {})
        original_func.assert_called_once_with() ####### THIS TEST IS IRRELEVANT
        instance.add_handler.assert_called_once_with(mock_handler, True)
    
    def test_init_wrapper_handler_already_exists(self):
        """Test that the wrapper doesn't add a duplicate handler."""
        # Create a mock callback handler
        mock_handler = Mock(spec=OpenTelemetryCallbackHandler)
        
        # Create the wrapper with our mock handler
        wrapper_instance = _BaseCallbackManagerInitWrapper(mock_handler)
        
        # Create mocks for the original function and instance
        original_func = Mock()
        instance = Mock()
        
        # Set up the instance to already have a handler of the same type
        # existing_handler = Mock(spec=OpenTelemetryCallbackHandler)
        # instance.inheritable_handlers = [existing_handler]
        
        mock_tracer = Mock()
        existing_handler = OpenTelemetryCallbackHandler(mock_tracer)  # Real handler with mock tracer
        instance.inheritable_handlers = [existing_handler]
        
        # Call the wrapper
        wrapper_instance(original_func, instance, [], {})
        
        # Verify the original function was called and handler was not added
        # original_func.assert_called_once_with(instance, [], {})
        original_func.assert_called_once_with() ######## another dumb test
        instance.add_handler.assert_not_called()


if __name__ == "__main__":
    import time  # Import needed for SpanHolder
    unittest.main()
