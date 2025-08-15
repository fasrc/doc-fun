"""
Tests for OpenAI provider implementation.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from doc_generator.providers.openai_provider import OpenAIProvider
from doc_generator.providers.base import CompletionRequest, CompletionResponse


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""
    
    def test_initialization_with_api_key(self):
        """Test provider initialization with explicit API key."""
        provider = OpenAIProvider(api_key="test-key")
        
        assert provider.api_key == "test-key"
        assert provider.client is not None
        assert provider.get_provider_name() == "openai"
    
    def test_initialization_from_environment(self):
        """Test provider initialization from environment variable."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            provider = OpenAIProvider()
            
            assert provider.api_key == "env-key"
    
    def test_initialization_no_api_key(self):
        """Test provider initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()
            
            assert provider.api_key is None
            assert provider.client is None
    
    def test_is_available(self):
        """Test is_available method."""
        # With API key
        provider = OpenAIProvider(api_key="test-key")
        assert provider.is_available() == True
        
        # Without API key
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()
            assert provider.is_available() == False
    
    def test_get_available_models(self):
        """Test get_available_models method."""
        provider = OpenAIProvider(api_key="test-key")
        models = provider.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'gpt-4' in models
        assert 'gpt-4o-mini' in models
        assert 'gpt-3.5-turbo' in models
    
    def test_validate_model(self):
        """Test model validation."""
        provider = OpenAIProvider(api_key="test-key")
        
        assert provider.validate_model('gpt-4') == True
        assert provider.validate_model('gpt-4o-mini') == True
        assert provider.validate_model('invalid-model') == False
    
    @patch('doc_generator.providers.openai_provider.OpenAI')
    def test_generate_completion_success(self, mock_openai_class):
        """Test successful completion generation."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated content"
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create provider and request
        provider = OpenAIProvider(api_key="test-key")
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7
        )
        
        # Generate completion
        response = provider.generate_completion(request)
        
        # Verify response
        assert isinstance(response, CompletionResponse)
        assert response.content == "Generated content"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30
        
        # Verify API call
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=4096
        )
    
    @patch('doc_generator.providers.openai_provider.OpenAI')
    def test_generate_completion_with_max_tokens(self, mock_openai_class):
        """Test completion generation with custom max_tokens."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated content"
        mock_response.model = "gpt-4"
        mock_response.usage = None  # Test without usage data
        
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(api_key="test-key")
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )
        
        response = provider.generate_completion(request)
        
        assert response.usage is None
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=2000
        )
    
    @patch.dict('os.environ', {}, clear=True)  # Clear environment variables
    def test_generate_completion_no_client(self):
        """Test completion generation without API key."""
        provider = OpenAIProvider()  # No API key
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7
        )
        
        with pytest.raises(ValueError, match="OpenAI API key not configured"):
            provider.generate_completion(request)
    
    @patch('doc_generator.providers.openai_provider.OpenAI')
    def test_generate_completion_api_error(self, mock_openai_class):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        provider = OpenAIProvider(api_key="test-key")
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7
        )
        
        with pytest.raises(Exception, match="API Error"):
            provider.generate_completion(request)
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        provider = OpenAIProvider(api_key="test-key")
        
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        response = CompletionResponse(
            content="Generated content",
            model="gpt-4o-mini",
            provider="openai",
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 2000,
                "total_tokens": 3000
            }
        )
        
        cost = provider.estimate_cost(request, response)
        
        # gpt-4o-mini: input $0.00015, output $0.0006 per 1K tokens
        # 1000 input tokens = $0.00015, 2000 output tokens = $0.0012
        expected_cost = (1000 / 1000) * 0.00015 + (2000 / 1000) * 0.0006
        assert cost == pytest.approx(expected_cost, rel=1e-6)
    
    def test_estimate_cost_no_usage(self):
        """Test cost estimation without usage data."""
        provider = OpenAIProvider(api_key="test-key")
        
        request = CompletionRequest([], "gpt-4", 0.7)
        response = CompletionResponse("content", "gpt-4", "openai", usage=None)
        
        assert provider.estimate_cost(request, response) is None
    
    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        provider = OpenAIProvider(api_key="test-key")
        
        request = CompletionRequest([], "unknown-model", 0.7)
        response = CompletionResponse(
            "content", "unknown-model", "openai",
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
        )
        
        assert provider.estimate_cost(request, response) is None