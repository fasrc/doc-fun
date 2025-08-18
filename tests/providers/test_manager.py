"""
Tests for ProviderManager class.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from doc_generator.providers.manager import ProviderManager
from doc_generator.providers.base import LLMProvider


class MockProvider(LLMProvider):
    """Mock provider for testing."""
    
    def __init__(self, name, models=None, available=True, **kwargs):
        super().__init__(**kwargs)
        self._name = name
        self._models = models or [f"{name}-model-1", f"{name}-model-2"]
        self._available = available
    
    def generate_completion(self, request):
        pass
    
    def get_available_models(self):
        return self._models
    
    def is_available(self):
        return self._available
    
    def get_provider_name(self):
        return self._name


class TestProviderManager:
    """Test ProviderManager functionality."""
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_initialization(self, mock_claude_provider, mock_openai_provider):
        """Test ProviderManager initialization."""
        # Mock providers
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_openai.get_available_models.return_value = ['gpt-4', 'gpt-3.5-turbo']
        mock_openai_provider.return_value = mock_openai
        
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        manager = ProviderManager()
        
        assert isinstance(manager.providers, dict)
        assert isinstance(manager.model_mapping, dict)
        mock_openai_provider.assert_called_once()
        mock_claude_provider.assert_called_once()
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_load_providers_success(self, mock_claude_provider, mock_openai_provider):
        """Test successful loading of available providers."""
        # Mock OpenAI provider (available)
        mock_openai = MockProvider('openai', ['gpt-4', 'gpt-3.5-turbo'], available=True)
        mock_openai_provider.return_value = mock_openai
        
        # Mock Claude provider (available)
        mock_claude = MockProvider('claude', ['claude-3-haiku'], available=True)
        mock_claude_provider.return_value = mock_claude
        
        manager = ProviderManager()
        
        assert 'openai' in manager.providers
        assert 'claude' in manager.providers
        assert 'gpt-4' in manager.model_mapping
        assert 'gpt-3.5-turbo' in manager.model_mapping
        assert 'claude-3-haiku' in manager.model_mapping
        assert manager.model_mapping['gpt-4'] == 'openai'
        assert manager.model_mapping['claude-3-haiku'] == 'claude'
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_load_providers_unavailable(self, mock_claude_provider, mock_openai_provider):
        """Test loading when providers are unavailable."""
        # Mock providers as unavailable
        mock_openai = MockProvider('openai', available=False)
        mock_openai_provider.return_value = mock_openai
        
        mock_claude = MockProvider('claude', available=False)
        mock_claude_provider.return_value = mock_claude
        
        manager = ProviderManager()
        
        # OpenAI provider is always registered for model listing
        assert 'openai' in manager.providers
        assert not manager.providers['openai'].is_available()
        # Claude is not registered when unavailable
        assert 'claude' not in manager.providers
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_load_providers_with_exceptions(self, mock_claude_provider, mock_openai_provider):
        """Test loading providers when exceptions occur."""
        # Mock OpenAI provider to raise exception
        mock_openai_provider.side_effect = Exception("OpenAI error")
        
        # Mock Claude provider as available
        mock_claude = MockProvider('claude', ['claude-3-haiku'], available=True)
        mock_claude_provider.return_value = mock_claude
        
        # Should not raise exception
        manager = ProviderManager()
        
        # Only Claude should be loaded
        assert 'openai' not in manager.providers
        assert 'claude' in manager.providers
        assert 'claude-3-haiku' in manager.model_mapping
    
    def test_get_provider(self):
        """Test getting provider by name."""
        manager = ProviderManager()
        
        # Mock a provider
        mock_provider = MockProvider('test')
        manager.providers['test'] = mock_provider
        
        assert manager.get_provider('test') == mock_provider
        assert manager.get_provider('nonexistent') is None
    
    def test_get_provider_for_model(self):
        """Test getting provider for specific model."""
        manager = ProviderManager()
        
        # Setup test data
        mock_provider = MockProvider('test')
        manager.providers['test'] = mock_provider
        manager.model_mapping['test-model'] = 'test'
        
        assert manager.get_provider_for_model('test-model') == mock_provider
        assert manager.get_provider_for_model('nonexistent-model') is None
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_get_available_providers(self, mock_claude_provider, mock_openai_provider):
        """Test getting list of available providers."""
        # Mock providers as unavailable to avoid loading real ones
        mock_openai = Mock()
        mock_openai.is_available.return_value = False
        mock_openai_provider.return_value = mock_openai
        
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        manager = ProviderManager()
        
        # Add test providers manually
        manager.providers['provider1'] = MockProvider('provider1')
        manager.providers['provider2'] = MockProvider('provider2')
        
        providers = manager.get_available_providers()
        # OpenAI is always registered, plus our test providers
        assert set(providers) == {'openai', 'provider1', 'provider2'}
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_get_available_models(self, mock_claude_provider, mock_openai_provider):
        """Test getting available models grouped by provider."""
        # Mock providers as unavailable to avoid loading real ones
        mock_openai = Mock()
        mock_openai.is_available.return_value = False
        mock_openai.get_available_models.return_value = []  # Return empty list for models
        mock_openai_provider.return_value = mock_openai
        
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        manager = ProviderManager()
        
        # Add test providers manually
        provider1 = MockProvider('provider1', ['model1', 'model2'])
        provider2 = MockProvider('provider2', ['model3', 'model4'])
        manager.providers['provider1'] = provider1
        manager.providers['provider2'] = provider2
        
        models = manager.get_available_models()
        
        assert models == {
            'openai': [],  # OpenAI is always present but has no models when unavailable
            'provider1': ['model1', 'model2'],
            'provider2': ['model3', 'model4']
        }
    
    def test_get_all_models(self):
        """Test getting flat list of all models."""
        manager = ProviderManager()
        
        # Setup test data
        manager.model_mapping = {
            'model1': 'provider1',
            'model2': 'provider1', 
            'model3': 'provider2'
        }
        
        models = manager.get_all_models()
        assert set(models) == {'model1', 'model2', 'model3'}
    
    def test_get_default_provider_prefers_openai(self):
        """Test that default provider prefers OpenAI when available."""
        manager = ProviderManager()
        
        # Add providers (OpenAI should be preferred)
        manager.providers['claude'] = MockProvider('claude')
        manager.providers['openai'] = MockProvider('openai')
        
        assert manager.get_default_provider() == 'openai'
    
    @patch.dict('os.environ', {}, clear=True)  # Clear environment variables
    def test_get_default_provider_fallback(self):
        """Test default provider fallback behavior."""
        manager = ProviderManager()
        
        # Add additional providers
        manager.providers['claude'] = MockProvider('claude')
        manager.providers['other'] = MockProvider('other')
        
        default = manager.get_default_provider()
        # OpenAI is always present and preferred when available
        assert default == 'openai'
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_get_default_provider_none_available(self, mock_claude_provider, mock_openai_provider):
        """Test default provider when none available."""
        # Mock providers as unavailable
        mock_openai = Mock()
        mock_openai.is_available.return_value = False
        mock_openai_provider.return_value = mock_openai
        
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        manager = ProviderManager()
        
        assert manager.get_default_provider() is None
    
    def test_get_default_model_openai(self):
        """Test getting default model for OpenAI provider."""
        manager = ProviderManager()
        
        # Mock OpenAI provider with preferred models
        openai_provider = MockProvider('openai', ['gpt-4', 'gpt-4o-mini', 'gpt-3.5-turbo'])
        manager.providers['openai'] = openai_provider
        
        with patch.object(manager, 'get_default_provider', return_value='openai'):
            model = manager.get_default_model()
            assert model == 'gpt-4o-mini'  # Preferred model
    
    def test_get_default_model_claude(self):
        """Test getting default model for Claude provider."""
        manager = ProviderManager()
        
        # Mock Claude provider
        claude_provider = MockProvider('claude', ['claude-3-opus-20240229', 'claude-3-5-sonnet-20240620'])
        manager.providers['claude'] = claude_provider
        
        with patch.object(manager, 'get_default_provider', return_value='claude'):
            model = manager.get_default_model()
            assert model == 'claude-3-5-sonnet-20240620'  # Preferred model
    
    def test_get_default_model_fallback(self):
        """Test getting default model fallback to first available."""
        manager = ProviderManager()
        
        # Mock provider with non-preferred models
        test_provider = MockProvider('test', ['custom-model-1', 'custom-model-2'])
        manager.providers['test'] = test_provider
        
        with patch.object(manager, 'get_default_provider', return_value='test'):
            model = manager.get_default_model()
            assert model == 'custom-model-1'  # First available
    
    def test_get_default_model_no_provider(self):
        """Test getting default model when no provider available."""
        manager = ProviderManager()
        
        with patch.object(manager, 'get_default_provider', return_value=None):
            assert manager.get_default_model() is None
    
    def test_validate_model_provider_combination_valid(self):
        """Test validation of valid model/provider combination."""
        manager = ProviderManager()
        
        # Setup test data
        provider = MockProvider('test', ['test-model'])
        manager.providers['test'] = provider
        
        is_valid, error = manager.validate_model_provider_combination('test-model', 'test')
        assert is_valid == True
        assert error is None
    
    def test_validate_model_provider_combination_invalid_provider(self):
        """Test validation with invalid provider."""
        manager = ProviderManager()
        
        is_valid, error = manager.validate_model_provider_combination('model', 'nonexistent')
        assert is_valid == False
        assert "Provider 'nonexistent' is not available" in error
    
    def test_validate_model_provider_combination_invalid_model(self):
        """Test validation with invalid model for provider."""
        manager = ProviderManager()
        
        # Setup provider without the requested model
        provider = MockProvider('test', ['other-model'])
        manager.providers['test'] = provider
        
        is_valid, error = manager.validate_model_provider_combination('test-model', 'test')
        assert is_valid == False
        assert "Model 'test-model' is not available from provider 'test'" in error
    
    def test_validate_model_provider_combination_auto_detect(self):
        """Test validation with auto-detection of provider."""
        manager = ProviderManager()
        
        # Setup test data
        manager.model_mapping['test-model'] = 'test'
        
        is_valid, error = manager.validate_model_provider_combination('test-model', None)
        assert is_valid == True
        assert error is None
        
        # Test with unavailable model
        is_valid, error = manager.validate_model_provider_combination('nonexistent-model', None)
        assert is_valid == False
        assert "Model 'nonexistent-model' is not available from any provider" in error
    
    def test_list_providers_info(self):
        """Test getting detailed provider information."""
        manager = ProviderManager()
        
        # Add test providers
        provider1 = MockProvider('provider1', ['model1'], available=True)
        provider2 = MockProvider('provider2', ['model2'], available=False)
        manager.providers['provider1'] = provider1
        manager.providers['provider2'] = provider2
        
        info = manager.list_providers_info()
        
        assert 'provider1' in info
        assert 'provider2' in info
        
        assert info['provider1']['name'] == 'provider1'
        assert info['provider1']['available'] == True
        assert info['provider1']['models'] == ['model1']
        assert info['provider1']['class'] == 'MockProvider'
        
        assert info['provider2']['available'] == False