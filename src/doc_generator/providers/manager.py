"""
Provider manager for handling multiple LLM providers.
"""

import logging
from typing import Dict, List, Optional, Tuple

from .base import LLMProvider
from .openai_provider import OpenAIProvider

# Optional Claude provider import
try:
    from .claude_provider import ClaudeProvider
    HAS_CLAUDE_PROVIDER = True
except ImportError:
    ClaudeProvider = None
    HAS_CLAUDE_PROVIDER = False


class ProviderManager:
    """Manages multiple LLM providers and model routing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize provider manager."""
        self.logger = logger or logging.getLogger(__name__)
        self.providers: Dict[str, LLMProvider] = {}
        self.model_mapping: Dict[str, str] = {}
        self._load_providers()
    
    def _load_providers(self):
        """Load and register available providers."""
        # Register OpenAI provider
        try:
            openai_provider = OpenAIProvider()
            # Always register the provider for model listing, but mark availability
            self.providers['openai'] = openai_provider
            for model in openai_provider.get_available_models():
                self.model_mapping[model] = 'openai'
            
            if openai_provider.is_available():
                self.logger.info("OpenAI provider loaded successfully")
            else:
                self.logger.info("OpenAI provider loaded (API key not configured - models listed but not usable)")
        except Exception as e:
            self.logger.warning(f"Failed to load OpenAI provider: {e}")
        
        # Register Claude provider (if available)
        if HAS_CLAUDE_PROVIDER:
            try:
                claude_provider = ClaudeProvider()
                if claude_provider.is_available():
                    self.providers['claude'] = claude_provider
                    for model in claude_provider.get_available_models():
                        self.model_mapping[model] = 'claude'
                    self.logger.info("Claude provider loaded successfully")
                else:
                    self.logger.info("Claude provider not available (API key not configured or anthropic package missing)")
            except Exception as e:
                self.logger.warning(f"Failed to load Claude provider: {e}")
        else:
            self.logger.info("Claude provider not available (claude_provider module not found)")
    
    def get_provider(self, provider_name: str) -> Optional[LLMProvider]:
        """Get provider by name."""
        return self.providers.get(provider_name)
    
    def get_provider_for_model(self, model: str) -> Optional[LLMProvider]:
        """Get provider that supports the specified model."""
        provider_name = self.model_mapping.get(model)
        return self.providers.get(provider_name) if provider_name else None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models grouped by provider."""
        return {
            name: provider.get_available_models() 
            for name, provider in self.providers.items()
        }
    
    def get_all_models(self) -> List[str]:
        """Get flat list of all available models across providers."""
        return list(self.model_mapping.keys())
    
    def get_default_provider(self) -> Optional[str]:
        """Get the default provider (first usable with API key configured)."""
        # Check providers that are actually usable (have API keys configured)
        usable_providers = [
            name for name, provider in self.providers.items() 
            if provider.is_available()
        ]
        
        if 'openai' in usable_providers:
            return 'openai'  # Prefer OpenAI for backward compatibility
        elif usable_providers:
            return usable_providers[0]
        return None
    
    def get_default_model(self) -> Optional[str]:
        """Get a default model from the default provider."""
        default_provider = self.get_default_provider()
        if not default_provider:
            return None
        
        provider = self.providers[default_provider]
        models = provider.get_available_models()
        
        # Return preferred models by provider
        if default_provider == 'openai':
            for preferred in ['gpt-5-mini', 'gpt-5', 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo']:
                if preferred in models:
                    return preferred
        elif default_provider == 'claude':
            for preferred in ['claude-3-5-sonnet-20240620', 'claude-3-haiku-20240307']:
                if preferred in models:
                    return preferred
        
        # Fallback to first available model
        return models[0] if models else None
    
    def validate_model_provider_combination(self, model: str, provider: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate if a model/provider combination is valid.
        
        Args:
            model: Model identifier
            provider: Provider name (optional, will auto-detect if None)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if provider:
            # Specific provider requested
            if provider not in self.providers:
                return False, f"Provider '{provider}' is not available"
            
            if not self.providers[provider].validate_model(model):
                return False, f"Model '{model}' is not available from provider '{provider}'"
                
            return True, None
        else:
            # Auto-detect provider
            if model not in self.model_mapping:
                return False, f"Model '{model}' is not available from any provider"
            
            return True, None
    
    def list_providers_info(self) -> Dict[str, Dict]:
        """Get detailed information about all providers."""
        info = {}
        
        for name, provider in self.providers.items():
            info[name] = {
                'name': name,
                'available': provider.is_available(),
                'models': provider.get_available_models(),
                'class': provider.__class__.__name__
            }
        
        return info