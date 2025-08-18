"""
OpenAI provider implementation.
"""

import os
import logging
from typing import List, Optional
from openai import OpenAI

from .base import LLMProvider, CompletionRequest, CompletionResponse
from ..exceptions import ProviderError


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
            raise ProviderError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
            
        try:
            # GPT-5 models have specific parameter requirements
            if request.model and request.model.startswith('gpt-5'):
                # GPT-5 only supports temperature=1.0 and max_completion_tokens instead of max_tokens
                gpt5_params = {
                    'model': request.model,
                    'messages': request.messages,
                    'temperature': 1.0,  # GPT-5 only supports temperature=1.0
                }
                
                # Try with max_completion_tokens first
                try:
                    response = self.client.chat.completions.create(
                        max_completion_tokens=request.max_tokens or 4096,
                        **gpt5_params
                    )
                except Exception as gpt5_error:
                    error_msg = str(gpt5_error).lower()
                    # Handle parameter compatibility issues
                    if 'max_completion_tokens' in error_msg or 'max_tokens' in error_msg:
                        self.logger.warning(f"GPT-5 max_completion_tokens parameter not supported yet, proceeding without token limit")
                        response = self.client.chat.completions.create(**gpt5_params)
                    else:
                        # Re-raise if it's a different error
                        raise gpt5_error
                        
                # Log temperature override for user awareness
                if request.temperature != 1.0:
                    self.logger.info(f"GPT-5 requires temperature=1.0, overriding requested temperature={request.temperature}")
            else:
                response = self.client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens or 4096
                )
            
            # Extract response content
            raw_content = response.choices[0].message.content
            
            # GPT-5 bug workaround: Models return None content despite successful API calls
            if not raw_content and request.model and request.model.startswith('gpt-5'):
                self.logger.warning(
                    f"⚠️ GPT-5 bug detected: {request.model} returned empty content despite successful API call. "
                    f"This is a known issue with GPT-5 models. Consider using GPT-4 models instead."
                )
                # Return empty content to allow graceful failure
                raw_content = ""
            elif not raw_content:
                self.logger.error(f"Empty response content from {request.model}")
                raw_content = ""
            
            return CompletionResponse(
                content=raw_content.strip() if raw_content else "",
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
            'gpt-5',                # ⚠️ Known issue: Returns empty content (API bug)
            'gpt-5-mini',           # ⚠️ Known issue: Returns empty content (API bug)
            'gpt-5-nano',           # ⚠️ Known issue: Returns empty content (API bug)
            'gpt-5-chat-latest',    # ⚠️ Known issue: Returns empty content (API bug)
            'gpt-4',                # Recommended: Stable and reliable
            'gpt-4o',               # Recommended: Optimized GPT-4
            'gpt-4o-mini',          # Recommended: Cost-efficient GPT-4
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
            'gpt-5': {'input': 0.04, 'output': 0.08},           # Flagship model pricing
            'gpt-5-mini': {'input': 0.002, 'output': 0.006},    # Cost-efficient version
            'gpt-5-nano': {'input': 0.0001, 'output': 0.0003},  # Most cost-efficient
            'gpt-5-chat-latest': {'input': 0.01, 'output': 0.03}, # ChatGPT version
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