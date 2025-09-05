"""
Claude (Anthropic) provider implementation.
"""

import os
import logging
from typing import List, Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .base import LLMProvider, CompletionRequest, CompletionResponse
from ..exceptions import ProviderError


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Claude provider."""
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.logger = logging.getLogger(__name__)
        
        if not HAS_ANTHROPIC:
            self.client = None
            self.logger.warning("anthropic package not installed. Install with: pip install anthropic")
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
    
    def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Claude API."""
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
            
        if not self.client:
            raise ProviderError("Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            # Convert OpenAI message format to Claude format
            system_message = ""
            user_messages = []
            
            for msg in request.messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    user_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
            
            # Make API call to Claude
            response = self.client.messages.create(
                model=request.model,
                system=system_message if system_message else None,
                messages=user_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 4096
            )
            
            return CompletionResponse(
                content=response.content[0].text,
                model=request.model,
                provider=self.get_provider_name(),
                usage={
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                } if response.usage else None
            )
            
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Claude API error: {e}")
            
            # Check for specific error types and provide better messages
            if 'credit balance is too low' in error_str.lower():
                raise ProviderError(
                    f"Anthropic API credit balance insufficient. "
                    f"Please add credits at https://console.anthropic.com/settings/billing "
                    f"or switch to OpenAI provider with --provider openai"
                )
            elif 'invalid_request_error' in error_str.lower():
                # Re-raise with the original message for other invalid request errors
                raise ProviderError(f"Claude API request error: {e}")
            else:
                # Re-raise the original exception for other errors
                raise
    
    def get_available_models(self) -> List[str]:
        """Get available Claude models."""
        return [
            # Claude 4 Family (Latest - January 2025)
            'claude-opus-4-20250514',      # Claude Opus 4 - Most advanced
            'claude-sonnet-4-20250514',    # Claude Sonnet 4 - Balanced performance
            'claude-4-0-sonnet-20250219',  # Alternative Claude 4 Sonnet identifier
            
            # Claude 3.7 Family (Latest reasoning models)
            'claude-3-7-sonnet-20250219',  # Claude 3.7 Sonnet with hybrid reasoning
            
            # Claude 3.5 Family (Current stable models)
            'claude-3-5-sonnet-20241022',  # Latest Sonnet 3.5
            'claude-3-5-sonnet-20240620',  # Previous Sonnet 3.5
            'claude-3-5-haiku-20241022',   # Latest Haiku 3.5
            
            # Claude 3 Family (Legacy support)
            'claude-3-haiku-20240307',     # Haiku 3.0
            'claude-3-sonnet-20240229',    # Sonnet 3.0
            'claude-3-opus-20240229',      # Opus 3.0 (legacy)
        ]
    
    def is_available(self) -> bool:
        """Check if Claude provider is available."""
        return HAS_ANTHROPIC and self.api_key is not None
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return 'claude'
    
    def estimate_cost(self, request: CompletionRequest, response: CompletionResponse) -> Optional[float]:
        """Estimate cost based on Claude pricing."""
        if not response.usage:
            return None
        
        # Claude pricing (per 1M tokens, as of January 2025)
        pricing = {
            # Claude 4 Family
            'claude-opus-4-20250514': {'input': 15.0, 'output': 75.0},      # $15/$75 per MTok
            'claude-sonnet-4-20250514': {'input': 3.0, 'output': 15.0},     # $3/$15 per MTok
            'claude-4-0-sonnet-20250219': {'input': 3.0, 'output': 15.0},   # Alternative identifier
            
            # Claude 3.7 Family
            'claude-3-7-sonnet-20250219': {'input': 3.0, 'output': 15.0},   # Hybrid reasoning model
            
            # Claude 3.5 Family
            'claude-3-5-sonnet-20241022': {'input': 3.0, 'output': 15.0},
            'claude-3-5-sonnet-20240620': {'input': 3.0, 'output': 15.0},
            'claude-3-5-haiku-20241022': {'input': 0.8, 'output': 4.0},     # $0.80/$4 per MTok
            
            # Claude 3 Family (Legacy)
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
            'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
            'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},
        }
        
        model_pricing = pricing.get(response.model)
        if not model_pricing:
            return None
            
        input_cost = (response.usage['prompt_tokens'] / 1_000_000) * model_pricing['input']
        output_cost = (response.usage['completion_tokens'] / 1_000_000) * model_pricing['output']
        
        return input_cost + output_cost