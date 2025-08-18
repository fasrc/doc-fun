"""
Tests for Claude provider implementation.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from doc_generator.providers.claude_provider import ClaudeProvider
from doc_generator.providers.base import CompletionRequest, CompletionResponse
from doc_generator.exceptions import ProviderError


class TestClaudeProvider:
    """Test Claude provider implementation."""
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', True)
    def test_initialization_with_api_key(self):
        """Test provider initialization with explicit API key."""
        with patch('doc_generator.providers.claude_provider.anthropic'):
            provider = ClaudeProvider(api_key="test-key")
            
            assert provider.api_key == "test-key"
            assert provider.client is not None
            assert provider.get_provider_name() == "claude"
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', True)
    def test_initialization_from_environment(self):
        """Test provider initialization from environment variable."""
        with patch('doc_generator.providers.claude_provider.anthropic'):
            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'env-key'}):
                provider = ClaudeProvider()
                
                assert provider.api_key == "env-key"
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', False)
    def test_initialization_no_anthropic_package(self):
        """Test provider initialization without anthropic package."""
        provider = ClaudeProvider(api_key="test-key")
        
        assert provider.client is None
        assert provider.is_available() == False
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', True)
    def test_initialization_no_api_key(self):
        """Test provider initialization without API key."""
        with patch('doc_generator.providers.claude_provider.anthropic'):
            with patch.dict(os.environ, {}, clear=True):
                provider = ClaudeProvider()
                
                assert provider.api_key is None
                assert provider.client is None
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', True)
    def test_is_available(self):
        """Test is_available method."""
        with patch('doc_generator.providers.claude_provider.anthropic'):
            # With API key and package
            provider = ClaudeProvider(api_key="test-key")
            assert provider.is_available() == True
            
            # Without API key
            with patch.dict(os.environ, {}, clear=True):
                provider = ClaudeProvider()
                assert provider.is_available() == False
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', False)
    def test_is_available_no_package(self):
        """Test is_available without anthropic package."""
        provider = ClaudeProvider(api_key="test-key")
        assert provider.is_available() == False
    
    def test_get_available_models(self):
        """Test get_available_models method."""
        provider = ClaudeProvider(api_key="test-key")
        models = provider.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'claude-opus-4-1-20250805' in models  # Claude 4.1
        assert 'claude-3-5-sonnet-20241022' in models
        assert 'claude-3-5-sonnet-20240620' in models
        assert 'claude-3-5-haiku-20241022' in models
        assert 'claude-3-haiku-20240307' in models
        assert 'claude-3-sonnet-20240229' in models
    
    def test_validate_model(self):
        """Test model validation."""
        provider = ClaudeProvider(api_key="test-key")
        
        assert provider.validate_model('claude-3-5-sonnet-20240620') == True
        assert provider.validate_model('claude-3-haiku-20240307') == True
        assert provider.validate_model('invalid-model') == False
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', False)
    def test_generate_completion_no_package(self):
        """Test completion generation without anthropic package."""
        provider = ClaudeProvider(api_key="test-key")
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-haiku-20240307",
            temperature=0.7
        )
        
        with pytest.raises(ImportError, match="anthropic package not installed"):
            provider.generate_completion(request)
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', True)
    def test_generate_completion_no_client(self):
        """Test completion generation without API key."""
        with patch('doc_generator.providers.claude_provider.anthropic'):
            with patch.dict(os.environ, {}, clear=True):  # Clear environment
                provider = ClaudeProvider()  # No API key
                request = CompletionRequest(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="claude-3-haiku-20240307",
                    temperature=0.7
                )
                
                with pytest.raises(ProviderError, match="Anthropic API key not configured"):
                    provider.generate_completion(request)
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', True)
    @patch('doc_generator.providers.claude_provider.anthropic')
    def test_generate_completion_success(self, mock_anthropic):
        """Test successful completion generation."""
        # Mock Anthropic client and response
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Generated content"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        
        mock_client.messages.create.return_value = mock_response
        
        # Create provider and request
        provider = ClaudeProvider(api_key="test-key")
        request = CompletionRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            model="claude-3-haiku-20240307",
            temperature=0.7
        )
        
        # Generate completion
        response = provider.generate_completion(request)
        
        # Verify response
        assert isinstance(response, CompletionResponse)
        assert response.content == "Generated content"
        assert response.model == "claude-3-haiku-20240307"
        assert response.provider == "claude"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30
        
        # Verify API call (system message separated from user messages)
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-haiku-20240307",
            system="You are helpful",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=4096
        )
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', True)
    @patch('doc_generator.providers.claude_provider.anthropic')
    def test_generate_completion_no_system_message(self, mock_anthropic):
        """Test completion generation without system message."""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Generated content"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        
        mock_client.messages.create.return_value = mock_response
        
        provider = ClaudeProvider(api_key="test-key")
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=2000
        )
        
        response = provider.generate_completion(request)
        
        # Verify API call (no system parameter when no system message)
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-haiku-20240307",
            system=None,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=2000
        )
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', True)
    @patch('doc_generator.providers.claude_provider.anthropic')
    def test_generate_completion_no_usage_data(self, mock_anthropic):
        """Test completion generation without usage data."""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Generated content"
        mock_response.usage = None
        
        mock_client.messages.create.return_value = mock_response
        
        provider = ClaudeProvider(api_key="test-key")
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-haiku-20240307",
            temperature=0.7
        )
        
        response = provider.generate_completion(request)
        assert response.usage is None
    
    @patch('doc_generator.providers.claude_provider.HAS_ANTHROPIC', True)
    @patch('doc_generator.providers.claude_provider.anthropic')
    def test_generate_completion_api_error(self, mock_anthropic):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")
        
        provider = ClaudeProvider(api_key="test-key")
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-haiku-20240307",
            temperature=0.7
        )
        
        with pytest.raises(Exception, match="API Error"):
            provider.generate_completion(request)
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        provider = ClaudeProvider(api_key="test-key")
        
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-haiku-20240307",
            temperature=0.7
        )
        
        response = CompletionResponse(
            content="Generated content",
            model="claude-3-haiku-20240307",
            provider="claude",
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 2000,
                "total_tokens": 3000
            }
        )
        
        cost = provider.estimate_cost(request, response)
        
        # claude-3-haiku: input $0.00025, output $0.00125 per 1K tokens
        # 1000 input tokens = $0.00025, 2000 output tokens = $0.0025
        expected_cost = (1000 / 1000) * 0.00025 + (2000 / 1000) * 0.00125
        assert cost == pytest.approx(expected_cost, rel=1e-6)
    
    def test_estimate_cost_no_usage(self):
        """Test cost estimation without usage data."""
        provider = ClaudeProvider(api_key="test-key")
        
        request = CompletionRequest([], "claude-3-haiku-20240307", 0.7)
        response = CompletionResponse("content", "claude-3-haiku-20240307", "claude", usage=None)
        
        assert provider.estimate_cost(request, response) is None
    
    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        provider = ClaudeProvider(api_key="test-key")
        
        request = CompletionRequest([], "unknown-model", 0.7)
        response = CompletionResponse(
            "content", "unknown-model", "claude",
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
        )
        
        assert provider.estimate_cost(request, response) is None