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
            self.logger.error(f"Claude API error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available Claude models."""
        return [
            # Claude 4 Family (Latest - January 2025)
            'claude-opus-4-20250514',      # Claude Opus 4 - Most advanced model
            'claude-sonnet-4-20250514',    # Claude Sonnet 4 - Balanced intelligence and speed
            'claude-4-0-sonnet-20250219',  # Alternative Sonnet 4 release
            
            # Claude 3.7 Family
            'claude-3-7-sonnet-20250109',  # Latest Claude 3.7 Sonnet
            
            # Claude 3.5 Family
            'claude-3-5-sonnet-20241022',  # Latest Sonnet 3.5
            'claude-3-5-sonnet-20240620',  # Previous Sonnet 3.5
            'claude-3-5-haiku-20241022',   # Latest Haiku 3.5
            
            # Claude 3.0 Family (Legacy)
            'claude-3-haiku-20240307',     # Haiku 3.0
            'claude-3-sonnet-20240229',    # Sonnet 3.0
            # 'claude-3-opus-20240229',   # Opus 3.0 deprecated in favor of Opus 4
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
        
        # Claude pricing per 1M tokens (as of January 2025, subject to change)
        pricing = {
            # Claude 4 Family
            'claude-opus-4-20250514': {'input': 15.0, 'output': 75.0},    # Most advanced model
            'claude-sonnet-4-20250514': {'input': 3.0, 'output': 15.0},   # Balanced model
            'claude-4-0-sonnet-20250219': {'input': 3.0, 'output': 15.0}, # Alternative Sonnet 4
            
            # Claude 3.7 Family
            'claude-3-7-sonnet-20250109': {'input': 3.0, 'output': 15.0},
            
            # Claude 3.5 Family
            'claude-3-5-sonnet-20241022': {'input': 3.0, 'output': 15.0},
            'claude-3-5-sonnet-20240620': {'input': 3.0, 'output': 15.0},
            'claude-3-5-haiku-20241022': {'input': 0.8, 'output': 4.0},
            
            # Claude 3.0 Family (Legacy)
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
            'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
            'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},     # Deprecated
        }
        
        model_pricing = pricing.get(response.model)
        if not model_pricing:
            return None
            
        # Calculate cost based on per-million-token pricing
        input_cost = (response.usage['prompt_tokens'] / 1_000_000) * model_pricing['input']
        output_cost = (response.usage['completion_tokens'] / 1_000_000) * model_pricing['output']
        
        return input_cost + output_cost