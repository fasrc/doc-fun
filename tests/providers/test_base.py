"""
Tests for base provider classes and abstractions.
"""

import pytest
from unittest.mock import Mock
from doc_generator.providers.base import LLMProvider, CompletionRequest, CompletionResponse


class TestCompletionRequest:
    """Test CompletionRequest dataclass."""
    
    def test_creation_with_required_fields(self):
        """Test creating CompletionRequest with required fields."""
        messages = [{"role": "user", "content": "Hello"}]
        request = CompletionRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.7
        )
        
        assert request.messages == messages
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens is None
    
    def test_creation_with_optional_fields(self):
        """Test creating CompletionRequest with optional fields."""
        messages = [{"role": "user", "content": "Hello"}]
        request = CompletionRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
        
        assert request.max_tokens == 1000


class TestCompletionResponse:
    """Test CompletionResponse dataclass."""
    
    def test_creation_with_required_fields(self):
        """Test creating CompletionResponse with required fields."""
        response = CompletionResponse(
            content="Hello world",
            model="gpt-4",
            provider="openai"
        )
        
        assert response.content == "Hello world"
        assert response.model == "gpt-4" 
        assert response.provider == "openai"
        assert response.usage is None
    
    def test_creation_with_usage_data(self):
        """Test creating CompletionResponse with usage information."""
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        
        response = CompletionResponse(
            content="Hello world",
            model="gpt-4",
            provider="openai",
            usage=usage
        )
        
        assert response.usage == usage


class TestLLMProvider:
    """Test LLMProvider abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LLMProvider()
    
    def test_concrete_implementation_required_methods(self):
        """Test that concrete implementations must implement required methods."""
        
        class IncompleteProvider(LLMProvider):
            def get_provider_name(self):
                return "incomplete"
            # Missing other required methods
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider()
    
    def test_concrete_implementation_works(self):
        """Test that properly implemented provider works."""
        
        class TestProvider(LLMProvider):
            def generate_completion(self, request):
                return CompletionResponse(
                    content="test response",
                    model=request.model,
                    provider="test"
                )
            
            def get_available_models(self):
                return ["test-model-1", "test-model-2"]
            
            def is_available(self):
                return True
            
            def get_provider_name(self):
                return "test"
        
        provider = TestProvider(api_key="test-key", custom_param="value")
        
        assert provider.api_key == "test-key"
        assert provider.custom_param == "value"
        assert provider.get_provider_name() == "test"
        assert provider.is_available() == True
        assert provider.get_available_models() == ["test-model-1", "test-model-2"]
    
    def test_validate_model_method(self):
        """Test the validate_model method."""
        
        class TestProvider(LLMProvider):
            def generate_completion(self, request):
                pass
            
            def get_available_models(self):
                return ["model-1", "model-2"]
            
            def is_available(self):
                return True
            
            def get_provider_name(self):
                return "test"
        
        provider = TestProvider()
        
        assert provider.validate_model("model-1") == True
        assert provider.validate_model("model-2") == True
        assert provider.validate_model("model-3") == False
    
    def test_estimate_cost_default_implementation(self):
        """Test that default estimate_cost returns None."""
        
        class TestProvider(LLMProvider):
            def generate_completion(self, request):
                pass
            
            def get_available_models(self):
                return []
            
            def is_available(self):
                return True
            
            def get_provider_name(self):
                return "test"
        
        provider = TestProvider()
        request = CompletionRequest([], "model", 0.7)
        response = CompletionResponse("content", "model", "test")
        
        assert provider.estimate_cost(request, response) is None