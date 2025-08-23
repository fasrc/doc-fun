"""Tests for DocumentStandardizer."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.doc_generator.standardizers import DocumentStandardizer
from src.doc_generator.extractors import ExtractedContent
from src.doc_generator.providers.base import CompletionResponse
from src.doc_generator.exceptions import DocumentStandardizerError


class TestDocumentStandardizer:
    """Test DocumentStandardizer functionality."""
    
    def test_initialization(self):
        """Test DocumentStandardizer initialization."""
        with patch('src.doc_generator.standardizers.document_standardizer.ProviderManager'):
            standardizer = DocumentStandardizer()
            assert standardizer is not None
            assert standardizer.temperature == 0.3
            assert standardizer.output_format == 'html'
    
    @patch('src.doc_generator.standardizers.document_standardizer.ProviderManager')
    def test_initialization_with_params(self, mock_provider_manager):
        """Test DocumentStandardizer initialization with parameters."""
        mock_provider = Mock()
        mock_provider.get_name.return_value = "test_provider"
        mock_provider.get_model.return_value = "test_model"
        
        mock_manager = Mock()
        mock_manager.get_provider.return_value = mock_provider
        mock_provider_manager.return_value = mock_manager
        
        standardizer = DocumentStandardizer(
            provider="test_provider",
            model="test_model",
            temperature=0.7,
            output_format="markdown"
        )
        
        assert standardizer.provider_name == "test_provider"
        assert standardizer.model == "test_model"
        assert standardizer.temperature == 0.7
        assert standardizer.output_format == "markdown"
    
    def test_get_default_standardization_prompts(self):
        """Test default prompts generation."""
        with patch('src.doc_generator.standardizers.document_standardizer.ProviderManager'):
            standardizer = DocumentStandardizer()
            prompts = standardizer._get_default_standardization_prompts()
            
            assert 'system_prompt' in prompts
            assert 'user_prompt_template' in prompts
            assert len(prompts['system_prompt']) > 0
            assert '{title}' in prompts['user_prompt_template']
            assert '{content}' in prompts['user_prompt_template']
    
    @patch('src.doc_generator.standardizers.document_standardizer.ProviderManager')
    def test_find_suitable_extractor(self, mock_provider_manager):
        """Test finding suitable extractor."""
        standardizer = DocumentStandardizer()
        
        # HTML content should find HTML extractor
        html_content = "<html><body><h1>Test</h1></body></html>"
        extractor = standardizer._find_suitable_extractor(html_content)
        assert extractor is not None
        assert extractor.get_format_type() == "html"
        
        # Plain text should not find extractor
        plain_content = "Just plain text"
        extractor = standardizer._find_suitable_extractor(plain_content)
        assert extractor is None
    
    @patch('src.doc_generator.standardizers.document_standardizer.ProviderManager')
    def test_build_terminology_context(self, mock_provider_manager):
        """Test terminology context building."""
        standardizer = DocumentStandardizer()
        
        # Test with empty terminology
        standardizer.terminology = {}
        context = standardizer._build_terminology_context()
        assert "No specific terminology provided" in context
        
        # Test with terminology
        standardizer.terminology = {
            'definitions': {
                'API': 'Application Programming Interface',
                'SDK': 'Software Development Kit'
            },
            'modules': {
                'authentication': ['oauth', 'jwt'],
                'database': 'postgresql'
            }
        }
        
        context = standardizer._build_terminology_context()
        assert "Key Terms:" in context
        assert "API: Application Programming Interface" in context
        assert "SDK: Software Development Kit" in context
        assert "Relevant Modules/Categories:" in context
        assert "authentication: oauth, jwt" in context
        assert "database: postgresql" in context
    
    @patch('src.doc_generator.standardizers.document_standardizer.ProviderManager')
    def test_build_standardization_prompt(self, mock_provider_manager):
        """Test standardization prompt building."""
        standardizer = DocumentStandardizer()
        standardizer.prompt_config = {
            'system_prompt': 'Test system prompt',
            'user_prompt_template': 'Title: {title}\nContent: {content}\nFormat: {output_format}'
        }
        
        extracted = ExtractedContent(
            title="Test Document",
            sections={"intro": "Introduction content", "usage": "Usage content"},
            format_type="html"
        )
        
        prompt_data = standardizer._build_standardization_prompt(extracted)
        
        assert 'system_prompt' in prompt_data
        assert 'user_prompt' in prompt_data
        assert prompt_data['system_prompt'] == 'Test system prompt'
        assert "Test Document" in prompt_data['user_prompt']
        assert "Introduction content" in prompt_data['user_prompt']
        assert "HTML" in prompt_data['user_prompt']
    
    @patch('src.doc_generator.standardizers.document_standardizer.ProviderManager')
    def test_process_standardization_response(self, mock_provider_manager):
        """Test standardization response processing."""
        mock_provider = Mock()
        mock_provider.get_model.return_value = "test_model"
        
        standardizer = DocumentStandardizer()
        standardizer.client = mock_provider
        standardizer.provider_name = "test_provider"
        
        # Mock response
        response = CompletionResponse(
            content="Standardized content here",
            model="test_model",
            tokens_used=150
        )
        
        # Mock original content
        original = ExtractedContent(
            title="Original Title",
            sections={"intro": "content"},
            format_type="html",
            metadata={"author": "test"}
        )
        
        result = standardizer._process_standardization_response(response, original)
        
        assert result['standardized_content'] == "Standardized content here"
        assert result['original_title'] == "Original Title"
        assert result['original_format'] == "html"
        assert result['target_format'] == "html"
        assert result['sections_processed'] == ["intro"]
        assert result['metadata']['provider'] == "test_provider"
        assert result['metadata']['model'] == "test_model"
        assert result['metadata']['tokens_used'] == 150
    
    @patch('src.doc_generator.standardizers.document_standardizer.ProviderManager')
    def test_standardize_document_no_extractor(self, mock_provider_manager):
        """Test standardization when no suitable extractor is found."""
        standardizer = DocumentStandardizer()
        
        # Plain text that won't match any extractor
        plain_content = "Just plain text with no markup"
        
        with pytest.raises(DocumentStandardizerError) as exc_info:
            standardizer.standardize_document(plain_content)
        
        assert "No suitable extractor found" in str(exc_info.value)
    
    def test_standardize_file_not_found(self):
        """Test standardization of non-existent file."""
        with patch('src.doc_generator.standardizers.document_standardizer.ProviderManager'):
            standardizer = DocumentStandardizer()
            
            with pytest.raises(DocumentStandardizerError) as exc_info:
                standardizer.standardize_file("non_existent_file.html")
            
            assert "File not found" in str(exc_info.value)
    
    @patch('src.doc_generator.standardizers.document_standardizer.ProviderManager')
    def test_standardize_file_success(self, mock_provider_manager):
        """Test successful file standardization."""
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
            tmp.write("""
            <html>
            <head><title>Test Doc</title></head>
            <body>
                <h1>Introduction</h1>
                <p>This is test content.</p>
            </body>
            </html>
            """)
            tmp_path = tmp.name
        
        try:
            # Mock provider and client
            mock_provider = Mock()
            mock_provider.get_model.return_value = "test_model"
            mock_provider.get_name.return_value = "test_provider"
            mock_provider.complete.return_value = CompletionResponse(
                content="# Test Doc\n\n## Introduction\n\nStandardized content.",
                model="test_model",
                tokens_used=100
            )
            
            mock_manager = Mock()
            mock_manager.get_first_available_provider.return_value = mock_provider
            mock_provider_manager.return_value = mock_manager
            
            standardizer = DocumentStandardizer()
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as out_tmp:
                out_path = out_tmp.name
            
            try:
                result = standardizer.standardize_file(tmp_path, out_path)
                
                assert 'standardized_content' in result
                assert 'output_path' in result
                assert result['original_format'] == 'html'
                assert result['target_format'] == 'html'  # Default
                
                # Check output file was created
                assert os.path.exists(out_path)
                
            finally:
                if os.path.exists(out_path):
                    os.unlink(out_path)
                    
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch('src.doc_generator.standardizers.document_standardizer.ProviderManager')
    def test_load_terminology(self, mock_provider_manager):
        """Test terminology loading."""
        # Create temporary terminology file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write("""
definitions:
  API: Application Programming Interface
  REST: Representational State Transfer

modules:
  web:
    - flask
    - django
  testing:
    - pytest
    - unittest
            """)
            tmp_path = tmp.name
        
        try:
            standardizer = DocumentStandardizer()
            standardizer.terminology_path = Path(tmp_path)
            
            terminology = standardizer._load_terminology()
            
            assert 'definitions' in terminology
            assert 'modules' in terminology
            assert terminology['definitions']['API'] == 'Application Programming Interface'
            assert 'flask' in terminology['modules']['web']
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)