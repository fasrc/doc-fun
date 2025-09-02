"""
Tests for the DocumentCompiler analysis plugin.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

from doc_generator.analysis.compiler import DocumentCompiler
from doc_generator.core import DocumentAnalyzer


class TestDocumentCompiler:
    """Test the DocumentCompiler analysis plugin."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for DocumentCompiler."""
        return {
            'weights': {'algorithmic': 0.8, 'gpt_quality': 0.2},
            'use_gpt': False,
            'section_headers': ['Description', 'Installation', 'Usage'],
            'min_runs': 2,
            'report_format': 'markdown'
        }
    
    @pytest.fixture
    def compiler(self, mock_logger, sample_config):
        """Create DocumentCompiler instance."""
        return DocumentCompiler(logger=mock_logger, config=sample_config)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                'path': '/test/doc1.html',
                'content': '''
                <h2>Description</h2>
                <p>This is a good description with detailed information about the topic.</p>
                <h2>Installation</h2>
                <p>Simple installation instructions.</p>
                <h2>Usage</h2>
                <p>Basic usage examples.</p>
                '''
            },
            {
                'path': '/test/doc2.html', 
                'content': '''
                <h2>Description</h2>
                <p>Short description.</p>
                <h2>Installation</h2>
                <p>Comprehensive installation guide with multiple steps and troubleshooting.</p>
                <h2>Usage</h2>
                <p>Advanced usage with examples.</p>
                '''
            },
            {
                'path': '/test/doc3.html',
                'content': '''
                <h2>Description</h2>
                <p>Very detailed and comprehensive description with examples and background.</p>
                <h2>Installation</h2>
                <p>Installation via pip.</p>
                <h2>Usage</h2>
                <p>Detailed usage guide with multiple examples and edge cases.</p>
                '''
            }
        ]

    def test_init_default_config(self, mock_logger):
        """Test DocumentCompiler initialization with default config."""
        compiler = DocumentCompiler(logger=mock_logger)
        
        assert compiler.logger == mock_logger
        assert compiler.weights == {'algorithmic': 0.7, 'gpt_quality': 0.3}
        assert compiler.use_gpt == False
        assert compiler.section_headers == ['Description', 'Installation', 'Usage', 'Examples', 'References']
        assert compiler.min_runs == 2
        assert compiler.report_format == 'markdown'
        assert isinstance(compiler.analyzer, DocumentAnalyzer)
        assert compiler.gpt_evaluator is None

    def test_init_custom_config(self, compiler, sample_config):
        """Test DocumentCompiler initialization with custom config."""
        assert compiler.weights == sample_config['weights']
        assert compiler.use_gpt == sample_config['use_gpt']
        assert compiler.section_headers == sample_config['section_headers']
        assert compiler.min_runs == sample_config['min_runs']
        assert compiler.report_format == sample_config['report_format']

    def test_get_name(self, compiler):
        """Test get_name method."""
        assert compiler.get_name() == 'compiler'

    def test_get_priority(self, compiler):
        """Test get_priority method."""
        assert compiler.get_priority() == 100

    def test_get_supported_formats(self, compiler):
        """Test get_supported_formats method."""
        assert compiler.get_supported_formats() == ['html', 'markdown']

    @patch('doc_generator.analysis.compiler.DocumentAnalyzer')
    def test_analyze_insufficient_documents(self, mock_analyzer_class, compiler):
        """Test analyze method with insufficient documents."""
        # Setup
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        compiler.analyzer = mock_analyzer
        
        documents = [{'path': '/test/doc1.html', 'content': 'test content'}]
        topic = "Test Topic"
        
        # Execute
        result = compiler.analyze(documents, topic)
        
        # Verify
        assert 'message' in result
        assert 'Insufficient documents' in result['message']
        assert result['best_sections'] == {}
        assert result['section_scores'] == {}
        assert result['compilation_html'] is None

    @patch('doc_generator.analysis.compiler.DocumentAnalyzer')
    def test_analyze_with_documents(self, mock_analyzer_class, compiler, sample_documents):
        """Test analyze method with sufficient documents."""
        # Setup mock analyzer
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        compiler.analyzer = mock_analyzer
        
        # Mock section extraction
        def mock_extract_sections(content):
            if 'doc1' in content:
                return {
                    'Description': '<p>This is a good description with detailed information about the topic.</p>',
                    'Installation': '<p>Simple installation instructions.</p>',
                    'Usage': '<p>Basic usage examples.</p>'
                }
            elif 'doc2' in content:
                return {
                    'Description': '<p>Short description.</p>',
                    'Installation': '<p>Comprehensive installation guide with multiple steps and troubleshooting.</p>',
                    'Usage': '<p>Advanced usage with examples.</p>'
                }
            else:
                return {
                    'Description': '<p>Very detailed and comprehensive description with examples and background.</p>',
                    'Installation': '<p>Installation via pip.</p>',
                    'Usage': '<p>Detailed usage guide with multiple examples and edge cases.</p>'
                }
        
        mock_analyzer.extract_sections.side_effect = mock_extract_sections
        
        # Mock section scoring - return different scores for different sections
        def mock_score_section(content, section_name):
            content_length = len(content)
            if section_name == 'Description':
                return content_length * 0.1
            elif section_name == 'Installation':
                return content_length * 0.08
            else:
                return content_length * 0.09
        
        mock_analyzer.calculate_section_score.side_effect = mock_score_section
        
        topic = "Test Topic"
        
        # Execute
        result = compiler.analyze(sample_documents, topic)
        
        # Verify structure
        assert 'best_sections' in result
        assert 'section_scores' in result
        assert 'compilation_html' in result
        
        # Verify best sections were selected
        best_sections = result['best_sections']
        assert len(best_sections) == 3  # Description, Installation, Usage
        
        # Verify each section has the expected structure
        for section_name in ['Description', 'Installation', 'Usage']:
            assert section_name in best_sections
            section = best_sections[section_name]
            assert 'content' in section
            assert 'score' in section
            assert 'algo_score' in section
            assert 'gpt_score' in section
            assert 'doc_index' in section
            assert 'doc_path' in section
        
        # Verify section scores structure
        section_scores = result['section_scores']
        for section_name in ['Description', 'Installation', 'Usage']:
            assert section_name in section_scores
            assert len(section_scores[section_name]) == 3  # Three documents

    def test_analyze_empty_content(self, compiler, mock_logger):
        """Test analyze method with documents containing empty content."""
        documents = [
            {'path': '/test/doc1.html', 'content': ''},
            {'path': '/test/doc2.html', 'content': '   '},
            {'path': '/test/doc3.html', 'content': None}
        ]
        topic = "Test Topic"
        
        with patch.object(compiler.analyzer, 'extract_sections') as mock_extract:
            mock_extract.return_value = {}
            result = compiler.analyze(documents, topic)
        
        # Should still return structure but with empty results
        assert 'best_sections' in result
        assert 'section_scores' in result
        assert 'compilation_html' in result
        assert len(result['best_sections']) == 0

    def test_create_compilation_html(self, compiler):
        """Test _create_compilation_html method."""
        best_sections = {
            'Description': {
                'content': '<p>Best description content</p>',
                'score': 0.95,
                'doc_path': '/test/doc1.html'
            },
            'Installation': {
                'content': '<p>Best installation content</p>',
                'score': 0.88,
                'doc_path': '/test/doc2.html'
            }
        }
        topic = "Test Topic"
        
        html = compiler._create_compilation_html(best_sections, topic)
        
        # Verify HTML structure
        assert '<!DOCTYPE html>' in html
        assert '<title>Test Topic - Best Compilation</title>' in html
        assert '<h1>Test Topic</h1>' in html
        assert 'This is a compilation of the best sections' in html
        assert '<h2>Description</h2>' in html
        assert '<p>Best description content</p>' in html
        assert '<h2>Installation</h2>' in html
        assert '<p>Best installation content</p>' in html
        assert '<!-- Source: /test/doc1.html (score: 0.95) -->' in html
        assert '<!-- Source: /test/doc2.html (score: 0.88) -->' in html

    def test_generate_report_insufficient_docs(self, compiler):
        """Test generate_report with insufficient documents."""
        analysis_results = {
            'message': 'Insufficient documents for compilation (need at least 2)'
        }
        topic = "Test Topic"
        
        report = compiler.generate_report(analysis_results, topic)
        
        assert '# Document Compilation Report: Test Topic' in report
        assert '**Note:** Insufficient documents for compilation' in report

    def test_generate_report_with_results(self, compiler):
        """Test generate_report with analysis results."""
        analysis_results = {
            'best_sections': {
                'Description': {
                    'content': '<p>Best description</p>',
                    'score': 0.95,
                    'algo_score': 0.90,
                    'gpt_score': 0.0,
                    'doc_index': 0,
                    'doc_path': '/test/doc1.html'
                },
                'Installation': {
                    'content': '<p>Best installation</p>',
                    'score': 0.88,
                    'algo_score': 0.85,
                    'gpt_score': 0.0,
                    'doc_index': 1,
                    'doc_path': '/test/doc2.html'
                }
            },
            'section_scores': {
                'Description': [
                    {'doc_index': 0, 'doc_path': '/test/doc1.html', 'score': 0.95, 'algo_score': 0.90, 'gpt_score': 0.0},
                    {'doc_index': 1, 'doc_path': '/test/doc2.html', 'score': 0.75, 'algo_score': 0.70, 'gpt_score': 0.0}
                ],
                'Installation': [
                    {'doc_index': 1, 'doc_path': '/test/doc2.html', 'score': 0.88, 'algo_score': 0.85, 'gpt_score': 0.0},
                    {'doc_index': 0, 'doc_path': '/test/doc1.html', 'score': 0.65, 'algo_score': 0.60, 'gpt_score': 0.0}
                ]
            }
        }
        topic = "Test Topic"
        
        report = compiler.generate_report(analysis_results, topic)
        
        # Verify report structure
        assert '# Document Compilation Report: Test Topic' in report
        assert '**Sections Compiled:** 2' in report
        assert '## Best Section Selection' in report
        assert '| Section | Best Source | Score | Algorithm Score | GPT Score |' in report
        assert '| Description | doc1.html | 0.95 | 0.90 | 0.00 |' in report
        assert '| Installation | doc2.html | 0.88 | 0.85 | 0.00 |' in report
        assert '## Detailed Section Scores' in report
        assert '### Description' in report
        assert '1. doc1.html: 0.95 (algo: 0.90, gpt: 0.00) **[SELECTED]**' in report
        assert '2. doc2.html: 0.75 (algo: 0.70, gpt: 0.00)' in report

    def test_count_unique_docs(self, compiler):
        """Test _count_unique_docs method."""
        section_scores = {
            'Description': [
                {'doc_index': 0}, {'doc_index': 1}, {'doc_index': 2}
            ],
            'Installation': [
                {'doc_index': 1}, {'doc_index': 2}
            ],
            'Usage': [
                {'doc_index': 0}, {'doc_index': 1}
            ]
        }
        
        count = compiler._count_unique_docs(section_scores)
        assert count == 3  # Documents 0, 1, and 2

    def test_save_artifacts(self, compiler, temp_dir):
        """Test save_artifacts method."""
        results = {
            'compilation_html': '<html><body><h1>Test</h1></body></html>',
            'best_sections': {
                'Description': {
                    'content': 'test',
                    'score': 0.95,
                    'algo_score': 0.90,
                    'gpt_score': 0.0,
                    'doc_index': 0,
                    'doc_path': '/test/doc1.html'
                }
            },
            'section_scores': {
                'Description': [
                    {'doc_index': 0, 'doc_path': '/test/doc1.html', 'score': 0.95, 'algo_score': 0.90, 'gpt_score': 0.0}
                ]
            }
        }
        topic = "Test Topic"
        
        saved_files = compiler.save_artifacts(results, temp_dir, topic)
        
        # Verify files were saved
        assert len(saved_files) == 2
        
        # Check compilation file
        compilation_file = temp_dir / 'test_topic_best_compilation.html'
        assert compilation_file in saved_files
        assert compilation_file.exists()
        content = compilation_file.read_text()
        assert '<html><body><h1>Test</h1></body></html>' in content
        
        # Check report file
        report_file = temp_dir / 'test_topic_compilation_report.md'
        assert report_file in saved_files
        assert report_file.exists()
        report_content = report_file.read_text()
        assert '# Document Compilation Report: Test Topic' in report_content

    def test_save_artifacts_no_compilation(self, compiler, temp_dir):
        """Test save_artifacts without compilation HTML."""
        results = {
            'compilation_html': None,
            'best_sections': {},
            'section_scores': {}
        }
        topic = "Test Topic"
        
        saved_files = compiler.save_artifacts(results, temp_dir, topic)
        
        # Should only save report file
        assert len(saved_files) == 1
        report_file = temp_dir / 'test_topic_compilation_report.md'
        assert report_file in saved_files
        assert report_file.exists()

    def test_sanitize_filename(self, compiler):
        """Test filename sanitization."""
        assert compiler.sanitize_filename("Test Topic") == "test_topic"
        assert compiler.sanitize_filename("Complex@Topic#Name!") == "complextopicname"
        assert compiler.sanitize_filename("Topic-With_Dashes") == "topic-with_dashes"
        assert compiler.sanitize_filename("   Spaces   ") == "___spaces___"

    @patch('doc_generator.analysis.compiler.DocumentAnalyzer')
    def test_weights_calculation(self, mock_analyzer_class, compiler, sample_documents):
        """Test that weights are applied correctly in score calculation."""
        # Setup mock analyzer
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        compiler.analyzer = mock_analyzer
        
        # Set specific weights
        compiler.weights = {'algorithmic': 0.6, 'gpt_quality': 0.4}
        
        # Mock returns
        mock_analyzer.extract_sections.return_value = {
            'Description': '<p>Test content</p>'
        }
        mock_analyzer.calculate_section_score.return_value = 0.8
        
        # Execute
        result = compiler.analyze(sample_documents[:2], "Test Topic")
        
        # Verify weight calculation (0.6 * 0.8 + 0.4 * 0 = 0.48)
        best_section = result['best_sections']['Description']
        expected_score = 0.6 * 0.8 + 0.4 * 0  # No GPT score
        assert abs(best_section['score'] - expected_score) < 0.001

    def test_min_runs_configuration(self, mock_logger):
        """Test min_runs configuration affects analysis."""
        # Test with min_runs = 3
        config = {'min_runs': 3}
        compiler = DocumentCompiler(logger=mock_logger, config=config)
        
        documents = [
            {'path': '/test/doc1.html', 'content': 'content1'},
            {'path': '/test/doc2.html', 'content': 'content2'}
        ]
        
        result = compiler.analyze(documents, "Test Topic")
        
        assert 'message' in result
        assert 'need at least 3' in result['message']


class TestDocumentCompilerIntegration:
    """Integration tests for DocumentCompiler with real dependencies."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_full_compilation_workflow(self, temp_dir):
        """Test complete compilation workflow with real DocumentAnalyzer."""
        # Create real DocumentCompiler (no mocks for core functionality)
        logger = Mock(spec=logging.Logger)
        compiler = DocumentCompiler(logger=logger)
        
        # Sample documents with real HTML structure
        documents = [
            {
                'path': str(temp_dir / 'doc1.html'),
                'content': '''
                <html><body>
                <h2>Description</h2>
                <p>This is a comprehensive description with detailed information.</p>
                <h2>Installation</h2>
                <p>pip install package</p>
                <h2>Usage</h2>
                <p>Basic usage example here.</p>
                </body></html>
                '''
            },
            {
                'path': str(temp_dir / 'doc2.html'),
                'content': '''
                <html><body>
                <h2>Description</h2>
                <p>Brief description.</p>
                <h2>Installation</h2>
                <p>Detailed installation guide with troubleshooting steps and multiple options.</p>
                <h2>Usage</h2>
                <p>Advanced usage with multiple examples and edge cases.</p>
                </body></html>
                '''
            }
        ]
        
        # Run analysis
        result = compiler.analyze(documents, "Integration Test Topic")
        
        # Verify results
        assert 'best_sections' in result
        assert 'compilation_html' in result
        assert result['compilation_html'] is not None
        
        # Should have selected best sections
        best_sections = result['best_sections']
        assert len(best_sections) > 0
        
        # Verify HTML compilation contains expected content
        html = result['compilation_html']
        assert 'Integration Test Topic' in html
        assert 'This is a compilation of the best sections' in html
        
        # Save artifacts and verify
        saved_files = compiler.save_artifacts(result, temp_dir, "Integration Test Topic")
        assert len(saved_files) == 2
        
        # Verify compilation file
        compilation_file = temp_dir / 'integration_test_topic_best_compilation.html'
        assert compilation_file.exists()
        
        # Verify report file
        report_file = temp_dir / 'integration_test_topic_compilation_report.md'
        assert report_file.exists()
        report_content = report_file.read_text()
        assert 'Document Compilation Report' in report_content