# Claude API Integration Design

**Status**: Draft  
**Author**: Development Team  
**Date**: 2025-07-31  
**Version**: 1.0  

---

## Overview

This document outlines the design for integrating Anthropic's Claude API alongside the existing OpenAI integration in doc-generator, providing users with choice between multiple LLM providers while maintaining backward compatibility.

## Goals

### **Primary Goals**
- Add Claude API support alongside existing OpenAI integration
- Maintain 100% backward compatibility with existing functionality
- Provide seamless provider switching and auto-detection
- Enable provider-specific optimizations and features

### **Secondary Goals**
- Support future LLM providers with minimal code changes
- Provide fallback mechanisms between providers
- Enable cost optimization through provider selection
- Support provider-specific model capabilities

## Current State Analysis

### **Existing Architecture**
```python
# Current implementation (simplified)
class DocumentationGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Hard-coded
    
    def generate_documentation(self, model='gpt-4'):
        response = self.client.chat.completions.create(...)  # Direct OpenAI call
```

### **Limitations**
- Single provider (OpenAI) hard-coded
- No abstraction for different API formats
- Model selection limited to OpenAI models
- No fallback mechanism if OpenAI is unavailable

## Proposed Architecture

### **1. Provider Abstraction Layer**

```python
# src/doc_generator/providers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CompletionRequest:
    messages: List[Dict[str, str]]
    model: str
    temperature: float
    max_tokens: Optional[int] = None

@dataclass 
class CompletionResponse:
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using the provider's API."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return list of available models for this provider."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        pass
```

### **2. Provider Implementations**

```python
# src/doc_generator/providers/openai_provider.py
import openai
from .base import LLMProvider, CompletionRequest, CompletionResponse

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
    
    def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        if not self.client:
            raise ValueError("OpenAI API key not configured")
            
        response = self.client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 4096
        )
        
        return CompletionResponse(
            content=response.choices[0].message.content.strip(),
            model=response.model,
            provider='openai',
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        )
    
    def get_available_models(self) -> List[str]:
        return ['gpt-4', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']
    
    def is_available(self) -> bool:
        return self.api_key is not None

# src/doc_generator/providers/claude_provider.py
import anthropic
from .base import LLMProvider, CompletionRequest, CompletionResponse

class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
    
    def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        if not self.client:
            raise ValueError("Anthropic API key not configured")
        
        # Convert OpenAI message format to Claude format
        system_message = ""
        user_messages = []
        
        for msg in request.messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                user_messages.append(msg)
        
        response = self.client.messages.create(
            model=request.model,
            system=system_message,
            messages=user_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 4096
        )
        
        return CompletionResponse(
            content=response.content[0].text,
            model=request.model,
            provider='claude',
            usage={
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
        )
    
    def get_available_models(self) -> List[str]:
        return [
            'claude-3-5-sonnet-20240620',
            'claude-3-opus-20240229', 
            'claude-3-haiku-20240307'
        ]
    
    def is_available(self) -> bool:
        return self.api_key is not None
```

### **3. Provider Manager**

```python
# src/doc_generator/providers/manager.py
from typing import Dict, List, Optional
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .claude_provider import ClaudeProvider

class ProviderManager:
    """Manages multiple LLM providers."""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.model_mapping: Dict[str, str] = {}
        self._load_providers()
    
    def _load_providers(self):
        """Load and register available providers."""
        # Register OpenAI provider
        openai_provider = OpenAIProvider()
        if openai_provider.is_available():
            self.providers['openai'] = openai_provider
            for model in openai_provider.get_available_models():
                self.model_mapping[model] = 'openai'
        
        # Register Claude provider
        claude_provider = ClaudeProvider()
        if claude_provider.is_available():
            self.providers['claude'] = claude_provider
            for model in claude_provider.get_available_models():
                self.model_mapping[model] = 'claude'
    
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
```

### **4. Updated DocumentationGenerator**

```python
# Updated src/doc_generator/core.py
from .providers.manager import ProviderManager
from .providers.base import CompletionRequest

class DocumentationGenerator:
    """Main class for generating documentation using multiple LLM providers."""
    
    def __init__(self, prompt_yaml_path: str = './prompts/generator/default.yaml', 
                 examples_dir: str = 'examples/',
                 terminology_path: str = 'terminology.yaml', 
                 provider: str = 'auto',
                 logger: Optional[logging.Logger] = None):
        """Initialize the documentation generator with configuration."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize provider manager
        self.provider_manager = ProviderManager()
        
        # Set default provider
        if provider == 'auto':
            available = self.provider_manager.get_available_providers()
            self.default_provider = available[0] if available else None
            if not self.default_provider:
                raise ValueError("No LLM providers are available. Check API keys.")
        else:
            if provider not in self.provider_manager.get_available_providers():
                raise ValueError(f"Provider '{provider}' is not available")
            self.default_provider = provider
        
        # Rest of initialization...
        self.examples_dir = Path(examples_dir)
        self.prompt_config = self._load_prompt_config(prompt_yaml_path)
        self.terminology = self._load_terminology(terminology_path)
        self.examples = self._load_examples()
        
        # Initialize plugin manager
        self.plugin_manager = PluginManager(
            terminology=self.terminology,
            logger=self.logger
        )
        self.plugin_manager.load_plugins()
    
    def generate_documentation(self, query: str, runs: int = 5, 
                             model: str = 'gpt-4o-mini', 
                             temperature: float = 0.7,
                             topic_filename: str = None,
                             output_dir: str = 'output',
                             provider: Optional[str] = None) -> List[str]:
        """Generate multiple documentation pages based on the query."""
        
        # Determine which provider to use
        if provider:
            llm_provider = self.provider_manager.get_provider(provider)
            if not llm_provider:
                raise ValueError(f"Provider '{provider}' not available")
        else:
            llm_provider = self.provider_manager.get_provider_for_model(model)
            if not llm_provider:
                # Fallback to default provider
                llm_provider = self.provider_manager.get_provider(self.default_provider)
                self.logger.warning(f"Model '{model}' not found, using default provider")
        
        if topic_filename is None:
            topic_filename = self._extract_topic_from_query(query)
        
        # Extract topic for terminology context
        topic = self._extract_topic_from_query(query).replace('_', ' ')
        
        generated_files = []
        
        # Build messages with topic-aware system prompt
        system_prompt = self._build_system_prompt(topic)
        
        for i in range(runs):
            try:
                messages = [
                    {'role': 'system', 'content': system_prompt}
                ]
                
                # Add few-shot examples
                for example in self.examples:
                    if 'role' in example and 'content' in example:
                        messages.append({
                            'role': example['role'],
                            'content': example['content']
                        })
                
                # Add user query
                messages.append({'role': 'user', 'content': query})
                
                # Create completion request
                request = CompletionRequest(
                    messages=messages,
                    model=model,
                    temperature=temperature
                )
                
                # Generate completion using provider
                response = llm_provider.generate_completion(request)
                content = response.content
                
                # Generate filename with provider info
                provider_name = response.provider
                model_name = model.replace('-', '').replace('.', '')
                temp_str = str(temperature).replace('.', '')

                if runs == 1:
                    filename = f'{topic_filename}_{provider_name}_{model_name}_temp{temp_str}.html'
                else:
                    filename = f'{topic_filename}_{provider_name}_{model_name}_temp{temp_str}_v{i+1}.html'
                
                # Save the response
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                
                file_path = output_path / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files.append(str(file_path))
                self.logger.info(f"Generated: {filename} using {provider_name}")
                
            except Exception as e:
                self.logger.error(f"Error generating documentation (run {i+1}): {e}")
                continue
        
        return generated_files
```

### **5. CLI Updates**

```python
# Updated src/doc_generator/cli.py
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(...)
    
    # Provider selection
    parser.add_argument('--provider', 
                       choices=['openai', 'claude', 'auto'], 
                       default='auto',
                       help='LLM provider to use (default: auto-detect)')
    
    # Model selection (now supports both OpenAI and Claude models)
    parser.add_argument('--model', default='gpt-4o-mini',
                       help='Model to use. Examples: gpt-4o-mini, claude-3-5-sonnet-20240620')
    
    # List available providers and models
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and providers')
    
    # ... rest of arguments

def main():
    """Main CLI entry point."""
    load_dotenv()
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Handle list-models command
    if args.list_models:
        from .providers.manager import ProviderManager
        manager = ProviderManager()
        
        print("Available LLM Providers and Models:")
        print("=" * 50)
        
        available_models = manager.get_available_models()
        if not available_models:
            print("No providers available. Check your API keys:")
            print("  - OPENAI_API_KEY for OpenAI models")
            print("  - ANTHROPIC_API_KEY for Claude models")
            return
        
        for provider, models in available_models.items():
            print(f"\n{provider.upper()}:")
            for model in models:
                print(f"  - {model}")
        
        return
    
    # Rest of main function...
```

### **6. Configuration Updates**

```python
# src/doc_generator/config/models.py
MODEL_CONFIGS = {
    # OpenAI Models
    'gpt-4': {
        'provider': 'openai',
        'max_tokens': 4096,
        'supports_system_message': True,
        'cost_per_1k_tokens': {'input': 0.03, 'output': 0.06}
    },
    'gpt-4o-mini': {
        'provider': 'openai', 
        'max_tokens': 4096,
        'supports_system_message': True,
        'cost_per_1k_tokens': {'input': 0.00015, 'output': 0.0006}
    },
    
    # Claude Models
    'claude-3-5-sonnet-20240620': {
        'provider': 'claude',
        'max_tokens': 4096,
        'supports_system_message': True,
        'cost_per_1k_tokens': {'input': 0.003, 'output': 0.015}
    },
    'claude-3-opus-20240229': {
        'provider': 'claude',
        'max_tokens': 4096, 
        'supports_system_message': True,
        'cost_per_1k_tokens': {'input': 0.015, 'output': 0.075}
    },
    'claude-3-haiku-20240307': {
        'provider': 'claude',
        'max_tokens': 4096,
        'supports_system_message': True,
        'cost_per_1k_tokens': {'input': 0.00025, 'output': 0.00125}
    }
}
```

## Implementation Plan

### **Phase 1: Foundation** (Week 1)
- [ ] Create provider abstraction layer (`base.py`)
- [ ] Implement OpenAI provider (refactor existing code)
- [ ] Create provider manager
- [ ] Update tests for provider abstraction

### **Phase 2: Claude Integration** (Week 2)  
- [ ] Add anthropic dependency to requirements
- [ ] Implement Claude provider
- [ ] Add message format conversion logic
- [ ] Test Claude provider with various models

### **Phase 3: CLI & UX** (Week 3)
- [ ] Update CLI with provider options
- [ ] Add `--list-models` command
- [ ] Update documentation with new usage examples
- [ ] Add configuration validation

### **Phase 4: Advanced Features** (Week 4)
- [ ] Add provider fallback mechanisms
- [ ] Implement cost estimation
- [ ] Add provider-specific optimizations
- [ ] Performance testing and benchmarking

## Usage Examples

### **Basic Usage (Backward Compatible)**
```bash
# Existing usage continues to work
doc-gen --topic "Python Programming" --model gpt-4o-mini
```

### **Explicit Provider Selection**
```bash
# Use Claude explicitly
doc-gen --topic "Python Programming" --provider claude --model claude-3-5-sonnet-20240620

# Use OpenAI explicitly  
doc-gen --topic "Python Programming" --provider openai --model gpt-4
```

### **Model Auto-Detection**
```bash
# Provider auto-detected from model name
doc-gen --topic "Python Programming" --model claude-3-haiku-20240307
```

### **List Available Options**
```bash
# See what's available
doc-gen --list-models
```

## Testing Strategy

### **Unit Tests**
- Provider abstraction layer
- Individual provider implementations
- Provider manager functionality
- Message format conversions

### **Integration Tests**
- End-to-end documentation generation with each provider
- Provider fallback mechanisms
- CLI integration with multiple providers

### **Performance Tests**
- Response time comparison between providers
- Token usage and cost analysis
- Concurrent request handling

## Success Metrics

### **Functional Metrics**
- [ ] 100% backward compatibility maintained
- [ ] Both OpenAI and Claude providers working
- [ ] Auto-detection and fallback working
- [ ] All existing tests passing

### **Quality Metrics**
- [ ] Documentation quality equivalent across providers
- [ ] Response time within acceptable ranges (<30s)
- [ ] Error handling robust and informative

### **User Experience Metrics**
- [ ] CLI remains intuitive and easy to use
- [ ] Clear error messages for configuration issues
- [ ] Comprehensive help and documentation

## 🔮 Future Considerations

### **Additional Providers**
- Google PaLM/Gemini
- Cohere
- Local models (Ollama, etc.)

### **Advanced Features**
- Provider load balancing
- Cost optimization algorithms
- Quality comparison across providers
- Custom provider plugins

### **Enterprise Features**
- Rate limiting and quotas
- Usage analytics and reporting
- Provider-specific security controls

---

## References

- [Anthropic Claude API Documentation](https://docs.anthropic.com/claude/reference/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Provider Pattern Implementation](https://en.wikipedia.org/wiki/Provider_model)
- [Abstract Factory Pattern](https://refactoring.guru/design-patterns/abstract-factory)