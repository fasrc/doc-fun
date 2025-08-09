"""
Abstract base classes for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CompletionRequest:
    """Request object for LLM completion."""
    messages: List[Dict[str, str]]
    model: str
    temperature: float
    max_tokens: Optional[int] = None


@dataclass 
class CompletionResponse:
    """Response object from LLM completion."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize provider with API key and additional configuration."""
        self.api_key = api_key
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using the provider's API.
        
        Args:
            request: CompletionRequest with messages, model, and parameters
            
        Returns:
            CompletionResponse with generated content and metadata
            
        Raises:
            ValueError: If provider is not properly configured
            Exception: For API-specific errors
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Return list of available models for this provider.
        
        Returns:
            List of model names/identifiers
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is configured and available.
        
        Returns:
            True if provider can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name/identifier of this provider.
        
        Returns:
            Provider name (e.g., 'openai', 'claude')
        """
        pass
    
    def validate_model(self, model: str) -> bool:
        """
        Validate if a model is supported by this provider.
        
        Args:
            model: Model identifier to validate
            
        Returns:
            True if model is supported, False otherwise
        """
        return model in self.get_available_models()
    
    def estimate_cost(self, request: CompletionRequest, response: CompletionResponse) -> Optional[float]:
        """
        Estimate the cost of a completion (if cost data is available).
        
        Args:
            request: The original request
            response: The response with usage data
            
        Returns:
            Estimated cost in USD, or None if cost data unavailable
        """
        # Default implementation returns None
        # Subclasses can override with provider-specific cost calculation
        return None