"""
OpenAI provider implementation.
"""

import os
import logging
from typing import List, Optional
from openai import OpenAI

from .base import LLMProvider, CompletionRequest, CompletionResponse


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI provider."""
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.logger = logging.getLogger(__name__)
    
    def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using OpenAI API."""
        if not self.client:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
            
        try:
            response = self.client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 4096
            )
            
            return CompletionResponse(
                content=response.choices[0].message.content.strip(),
                model=response.model,
                provider=self.get_provider_name(),
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                } if response.usage else None
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        return [
            'gpt-4',
            'gpt-4o', 
            'gpt-4o-mini',
            'gpt-4-turbo',
            'gpt-3.5-turbo'
        ]
    
    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return self.api_key is not None
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return 'openai'
    
    def estimate_cost(self, request: CompletionRequest, response: CompletionResponse) -> Optional[float]:
        """Estimate cost based on OpenAI pricing."""
        if not response.usage:
            return None
        
        # OpenAI pricing (as of 2025, subject to change)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015}
        }
        
        model_pricing = pricing.get(response.model)
        if not model_pricing:
            return None
            
        input_cost = (response.usage['prompt_tokens'] / 1000) * model_pricing['input']
        output_cost = (response.usage['completion_tokens'] / 1000) * model_pricing['output']
        
        return input_cost + output_cost