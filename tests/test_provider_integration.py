"""
Integration tests for provider system with DocumentationGenerator.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from doc_generator.core import DocumentationGenerator
from doc_generator.providers.base import CompletionResponse


class TestProviderIntegration:
    """Test integration of provider system with DocumentationGenerator."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create minimal config files
        self.prompt_file = self.temp_path / "prompt.yaml"
        self.terminology_file = self.temp_path / "terminology.yaml"
        self.shots_dir = self.temp_path / "examples"
        self.output_dir = self.temp_path / "output"
        
        self.shots_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create basic config files
        self.prompt_file.write_text("""
system_prompt: "You are a documentation generator."
terms:
  FASRC: "Faculty Arts and Sciences Research Computing"
""")
        
        self.terminology_file.write_text("""
hpc_modules:
  - name: "python/3.12.8-fasrc01"
    category: "programming"
    description: "Python 3.12"
""")
    
    def teardown_method(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_generator_initialization_auto_provider(self, mock_claude_provider, mock_openai_provider):
        """Test DocumentationGenerator initialization with auto provider detection."""
        # Mock OpenAI provider as available
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_openai.get_available_models.return_value = ['gpt-4', 'gpt-4o-mini']
        mock_openai.get_provider_name.return_value = 'openai'
        mock_openai_provider.return_value = mock_openai
        
        # Mock Claude provider as unavailable
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        generator = DocumentationGenerator(
            prompt_yaml_path=str(self.prompt_file),
            shots_dir=str(self.shots_dir),
            terminology_path=str(self.terminology_file),
            provider='auto'
        )
        
        assert generator.default_provider == 'openai'
        assert 'openai' in generator.provider_manager.get_available_providers()
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_generator_initialization_specific_provider(self, mock_claude_provider, mock_openai_provider):
        """Test DocumentationGenerator initialization with specific provider."""
        # Mock Claude provider as available
        mock_claude = Mock()
        mock_claude.is_available.return_value = True
        mock_claude.get_available_models.return_value = ['claude-3-haiku-20240307']
        mock_claude.get_provider_name.return_value = 'claude'
        mock_claude_provider.return_value = mock_claude
        
        # Mock OpenAI provider as available
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_openai.get_available_models.return_value = ['gpt-4']
        mock_openai.get_provider_name.return_value = 'openai'
        mock_openai_provider.return_value = mock_openai
        
        generator = DocumentationGenerator(
            prompt_yaml_path=str(self.prompt_file),
            shots_dir=str(self.shots_dir),
            terminology_path=str(self.terminology_file),
            provider='claude'
        )
        
        assert generator.default_provider == 'claude'
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_generator_initialization_no_providers(self, mock_claude_provider, mock_openai_provider):
        """Test DocumentationGenerator initialization when no providers available."""
        # Mock both providers as unavailable
        mock_openai = Mock()
        mock_openai.is_available.return_value = False
        mock_openai_provider.return_value = mock_openai
        
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        with pytest.raises(ValueError, match="No LLM providers are available"):
            DocumentationGenerator(
                prompt_yaml_path=str(self.prompt_file),
                shots_dir=str(self.shots_dir),
                terminology_path=str(self.terminology_file),
                provider='auto'
            )
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')  
    def test_generator_initialization_invalid_provider(self, mock_claude_provider, mock_openai_provider):
        """Test DocumentationGenerator initialization with invalid provider."""
        # Mock providers
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_openai_provider.return_value = mock_openai
        
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        with pytest.raises(ValueError, match="Provider 'invalid' is not available"):
            DocumentationGenerator(
                prompt_yaml_path=str(self.prompt_file),
                shots_dir=str(self.shots_dir),
                terminology_path=str(self.terminology_file),
                provider='invalid'
            )
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_generate_documentation_with_openai(self, mock_claude_provider, mock_openai_provider):
        """Test documentation generation using OpenAI provider."""
        # Mock OpenAI provider
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_openai.get_available_models.return_value = ['gpt-4o-mini']
        mock_openai.get_provider_name.return_value = 'openai'
        mock_openai.validate_model.return_value = True
        
        # Mock completion response
        mock_response = CompletionResponse(
            content="<html><body><h1>Test Documentation</h1></body></html>",
            model="gpt-4o-mini",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        mock_openai.generate_completion.return_value = mock_response
        mock_openai.estimate_cost.return_value = 0.001
        
        mock_openai_provider.return_value = mock_openai
        
        # Mock Claude provider as unavailable
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        generator = DocumentationGenerator(
            prompt_yaml_path=str(self.prompt_file),
            shots_dir=str(self.shots_dir),
            terminology_path=str(self.terminology_file)
        )
        
        # Generate documentation
        results = generator.generate_documentation(
            query="Create documentation for Python programming",
            model="gpt-4o-mini",
            runs=1,
            output_dir=str(self.output_dir)
        )
        
        assert len(results) == 1
        assert self.output_dir / results[0].split('/')[-1] in list(self.output_dir.iterdir())
        
        # Verify the provider was called correctly
        mock_openai.generate_completion.assert_called_once()
        call_args = mock_openai.generate_completion.call_args[0][0]
        assert call_args.model == "gpt-4o-mini"
        assert call_args.temperature == 0.7  # Default temperature
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_generate_documentation_with_claude(self, mock_claude_provider, mock_openai_provider):
        """Test documentation generation using Claude provider."""
        # Mock Claude provider
        mock_claude = Mock()
        mock_claude.is_available.return_value = True
        mock_claude.get_available_models.return_value = ['claude-3-haiku-20240307']
        mock_claude.get_provider_name.return_value = 'claude'
        mock_claude.validate_model.return_value = True
        
        # Mock completion response
        mock_response = CompletionResponse(
            content="<html><body><h1>Claude Generated Documentation</h1></body></html>",
            model="claude-3-haiku-20240307",
            provider="claude",
            usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
        )
        mock_claude.generate_completion.return_value = mock_response
        mock_claude.estimate_cost.return_value = 0.0005
        
        mock_claude_provider.return_value = mock_claude
        
        # Mock OpenAI provider as available but not used
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_openai.get_available_models.return_value = ['gpt-4']
        mock_openai_provider.return_value = mock_openai
        
        generator = DocumentationGenerator(
            prompt_yaml_path=str(self.prompt_file),
            shots_dir=str(self.shots_dir),
            terminology_path=str(self.terminology_file)
        )
        
        # Generate documentation explicitly using Claude
        results = generator.generate_documentation(
            query="Create documentation for machine learning",
            model="claude-3-haiku-20240307",
            runs=1,
            output_dir=str(self.output_dir),
            provider="claude"
        )
        
        assert len(results) == 1
        
        # Verify Claude provider was used
        mock_claude.generate_completion.assert_called_once()
        
        # Verify filename includes provider name
        filename = results[0].split('/')[-1]
        assert 'claude' in filename
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_generate_documentation_model_auto_detection(self, mock_claude_provider, mock_openai_provider):
        """Test documentation generation with model auto-detection."""
        # Mock OpenAI provider
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_openai.get_available_models.return_value = ['gpt-4o-mini']
        mock_openai.get_provider_name.return_value = 'openai'
        mock_openai_provider.return_value = mock_openai
        
        # Mock Claude provider
        mock_claude = Mock()
        mock_claude.is_available.return_value = True
        mock_claude.get_available_models.return_value = ['claude-3-haiku-20240307']
        mock_claude.get_provider_name.return_value = 'claude'
        mock_claude.validate_model.return_value = True
        
        # Mock completion response
        mock_response = CompletionResponse(
            content="<html><body><h1>Auto-detected Documentation</h1></body></html>",
            model="claude-3-haiku-20240307",
            provider="claude"
        )
        mock_claude.generate_completion.return_value = mock_response
        
        mock_claude_provider.return_value = mock_claude
        
        generator = DocumentationGenerator(
            prompt_yaml_path=str(self.prompt_file),
            shots_dir=str(self.shots_dir),
            terminology_path=str(self.terminology_file)
        )
        
        # Set up model mapping
        generator.provider_manager.model_mapping['claude-3-haiku-20240307'] = 'claude'
        
        # Generate documentation - provider should be auto-detected from model
        results = generator.generate_documentation(
            query="Create documentation",
            model="claude-3-haiku-20240307",  # This should auto-detect Claude provider
            runs=1,
            output_dir=str(self.output_dir)
        )
        
        assert len(results) == 1
        mock_claude.generate_completion.assert_called_once()
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_generate_documentation_invalid_model(self, mock_claude_provider, mock_openai_provider):
        """Test documentation generation with invalid model."""
        # Mock providers
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_openai.get_available_models.return_value = ['gpt-4']
        mock_openai.get_provider_name.return_value = 'openai'
        mock_openai.validate_model.return_value = False
        mock_openai_provider.return_value = mock_openai
        
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        generator = DocumentationGenerator(
            prompt_yaml_path=str(self.prompt_file),
            shots_dir=str(self.shots_dir),
            terminology_path=str(self.terminology_file)
        )
        
        # Try to generate with invalid model
        with pytest.raises(ValueError, match="Model.*is not available"):
            generator.generate_documentation(
                query="Create documentation",
                model="invalid-model",
                runs=1,
                output_dir=str(self.output_dir),
                provider="openai"
            )
    
    @patch('doc_generator.providers.manager.OpenAIProvider')
    @patch('doc_generator.providers.manager.ClaudeProvider')
    def test_generate_documentation_provider_fallback(self, mock_claude_provider, mock_openai_provider):
        """Test provider fallback when model not found."""
        # Mock OpenAI provider as default
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_openai.get_available_models.return_value = ['gpt-4o-mini']
        mock_openai.get_provider_name.return_value = 'openai'
        mock_openai.validate_model.return_value = True
        
        # Mock response
        mock_response = CompletionResponse(
            content="<html><body><h1>Fallback Documentation</h1></body></html>",
            model="gpt-4o-mini",
            provider="openai"
        )
        mock_openai.generate_completion.return_value = mock_response
        
        mock_openai_provider.return_value = mock_openai
        
        # Mock Claude provider as unavailable
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        mock_claude_provider.return_value = mock_claude
        
        generator = DocumentationGenerator(
            prompt_yaml_path=str(self.prompt_file),
            shots_dir=str(self.shots_dir),
            terminology_path=str(self.terminology_file)
        )
        
        # Generate with unknown model - should fallback to default provider and model
        results = generator.generate_documentation(
            query="Create documentation",
            model="unknown-model",  # This model doesn't exist
            runs=1,
            output_dir=str(self.output_dir)
        )
        
        assert len(results) == 1
        mock_openai.generate_completion.assert_called_once()
        
        # Should have used default model from default provider
        call_args = mock_openai.generate_completion.call_args[0][0]
        assert call_args.model == "gpt-4o-mini"  # Fallback model