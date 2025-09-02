"""
Tests for the AnalysisReporter analysis plugin.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
from datetime import datetime

from doc_generator.analysis.reporter import AnalysisReporter
from doc_generator.core import DocumentAnalyzer


class TestAnalysisReporter:
    """Test the AnalysisReporter analysis plugin."""
    
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
        """Sample configuration for AnalysisReporter."""
        return {
            'formats': ['markdown', 'html', 'json'],
            'include_stats': True,
            'include_comparisons': True,
            'section_headers': ['Description', 'Installation', 'Usage']
        }
    
    @pytest.fixture
    def reporter(self, mock_logger, sample_config):
        """Create AnalysisReporter instance."""
        return AnalysisReporter(logger=mock_logger, config=sample_config)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                'path': '/test/doc1.html',
                'content': '''
                <html>
                <body>
                    <h2>Description</h2>
                    <p>This is a comprehensive description with detailed information about the topic.</p>
                    <p>It includes multiple paragraphs with thorough explanations.</p>
                    <h2>Installation</h2>
                    <p>Simple installation instructions.</p>
                    <pre><code>pip install package</code></pre>
                    <h2>Usage</h2>
                    <p>Basic usage examples.</p>
                    <ul><li>Feature 1</li><li>Feature 2</li></ul>
                    <a href="https://example.com">Link</a>
                    <img src="image.jpg" alt="Image">
                    <table><tr><td>Data</td></tr></table>
                </body>
                </html>
                '''
            },
            {
                'path': '/test/doc2.html', 
                'content': '''
                <html>
                <body>
                    <h2>Description</h2>
                    <p>Short description.</p>
                    <h2>Installation</h2>
                    <p>Comprehensive installation guide with multiple steps.</p>
                    <pre><code>git clone repo</code></pre>
                    <pre><code>cd repo && make install</code></pre>
                    <h2>Usage</h2>
                    <p>Advanced usage with examples.</p>
                    <code>inline code</code>
                    <a href="https://test.com">Test Link</a>
                    <a href="https://github.com">GitHub</a>
                </body>
                </html>
                '''
            },
            {
                'path': '/test/doc3.html',
                'content': '''
                <html>
                <body>
                    <h2>Description</h2>
                    <p>Very detailed description with examples and background information.</p>
                    <p>Multiple paragraphs explaining the concept thoroughly.</p>
                    <p>Additional context and use cases.</p>
                    <h2>Installation</h2>
                    <p>Installation via pip.</p>
                    <h2>Usage</h2>
                    <p>Extensive usage guide with multiple examples.</p>
                    <p>Advanced configurations and customizations.</p>
                    <ol><li>Step 1</li><li>Step 2</li><li>Step 3</li></ol>
                </body>
                </html>
                '''
            }
        ]

    def test_init_default_config(self, mock_logger):
        """Test AnalysisReporter initialization with default config."""
        reporter = AnalysisReporter(logger=mock_logger)
        
        assert reporter.logger == mock_logger
        assert reporter.formats == ['markdown']
        assert reporter.include_stats == True
        assert reporter.include_comparisons == True
        assert reporter.section_headers == ['Description', 'Installation', 'Usage', 'Examples', 'References']
        assert isinstance(reporter.analyzer, DocumentAnalyzer)

    def test_init_custom_config(self, reporter, sample_config):
        """Test AnalysisReporter initialization with custom config."""
        assert reporter.formats == sample_config['formats']
        assert reporter.include_stats == sample_config['include_stats']
        assert reporter.include_comparisons == sample_config['include_comparisons']
        assert reporter.section_headers == sample_config['section_headers']

    def test_get_name(self, reporter):
        """Test get_name method."""
        assert reporter.get_name() == 'reporter'

    def test_get_priority(self, reporter):
        """Test get_priority method."""
        assert reporter.get_priority() == 80

    def test_get_supported_formats(self, reporter):
        """Test get_supported_formats method."""
        assert reporter.get_supported_formats() == ['markdown', 'json', 'html']

    @patch('doc_generator.analysis.reporter.DocumentAnalyzer')
    def test_analyze_document(self, mock_analyzer_class, reporter):
        """Test _analyze_document method."""
        # Setup mock analyzer
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        reporter.analyzer = mock_analyzer
        
        # Mock section extraction
        mock_analyzer.extract_sections.return_value = {
            'Description': '<p>Test description</p>',
            'Installation': '<pre><code>pip install</code></pre>',
            'Usage': '<p>Usage info</p><ul><li>Item</li></ul>'
        }
        
        # Mock section scoring
        def mock_score_section(content, section_name):
            return len(content) * 0.01  # Simple scoring based on length
        mock_analyzer.calculate_section_score.side_effect = mock_score_section
        
        content = '''
        <html>
        <body>
            <h2>Description</h2>
            <p>Test description</p>
            <h2>Installation</h2>
            <pre><code>pip install</code></pre>
            <a href="https://example.com">Link</a>
            <img src="image.jpg" alt="Image">
        </body>
        </html>
        '''
        
        result = reporter._analyze_document(content, '/test/doc.html', 0)
        
        # Verify structure
        assert result['path'] == '/test/doc.html'
        assert result['index'] == 0
        assert result['total_length'] == len(content)
        assert result['total_words'] > 0
        assert 'sections' in result
        assert 'code_blocks' in result
        assert 'links' in result
        assert 'images' in result
        
        # Verify sections were analyzed
        assert 'Description' in result['sections']
        assert 'Installation' in result['sections']
        assert 'Usage' in result['sections']
        
        # Verify section structure
        desc_section = result['sections']['Description']
        assert desc_section['exists'] == True
        assert desc_section['length'] > 0
        assert desc_section['words'] > 0
        assert desc_section['score'] > 0

    def test_analyze_document_missing_sections(self, reporter):
        """Test _analyze_document with missing sections."""
        content = '''
        <html>
        <body>
            <h2>Description</h2>
            <p>Only description exists</p>
        </body>
        </html>
        '''
        
        with patch.object(reporter.analyzer, 'extract_sections') as mock_extract:
            with patch.object(reporter.analyzer, 'calculate_section_score') as mock_score:
                mock_extract.return_value = {'Description': '<p>Only description exists</p>'}
                mock_score.return_value = 0.5
                
                result = reporter._analyze_document(content, '/test/doc.html', 0)
        
        # Should have all configured sections, even missing ones
        for header in reporter.section_headers:
            assert header in result['sections']
        
        # Missing sections should have exists=False
        assert result['sections']['Installation']['exists'] == False
        assert result['sections']['Usage']['exists'] == False
        assert result['sections']['Installation']['length'] == 0

    def test_calculate_overall_stats(self, reporter):
        """Test _calculate_overall_stats method."""
        document_metrics = [
            {
                'total_length': 1000,
                'total_words': 200,
                'code_blocks': 2,
                'links': 1,
                'sections': {
                    'Description': {'exists': True},
                    'Installation': {'exists': True},
                    'Usage': {'exists': False}
                }
            },
            {
                'total_length': 1500,
                'total_words': 300,
                'code_blocks': 3,
                'links': 2,
                'sections': {
                    'Description': {'exists': True},
                    'Installation': {'exists': False},
                    'Usage': {'exists': True}
                }
            }
        ]
        
        stats = reporter._calculate_overall_stats(document_metrics)
        
        # Verify calculations
        assert stats['total_documents'] == 2
        assert stats['average_length'] == 1250  # (1000 + 1500) / 2
        assert stats['average_words'] == 250    # (200 + 300) / 2
        assert stats['average_code_blocks'] == 2.5  # (2 + 3) / 2
        assert stats['average_links'] == 1.5    # (1 + 2) / 2
        
        # Verify section coverage
        coverage = stats['section_coverage']
        assert coverage['Description']['count'] == 2
        assert coverage['Description']['percentage'] == 100.0
        assert coverage['Installation']['count'] == 1
        assert coverage['Installation']['percentage'] == 50.0
        assert coverage['Usage']['count'] == 1
        assert coverage['Usage']['percentage'] == 50.0

    def test_calculate_overall_stats_empty(self, reporter):
        """Test _calculate_overall_stats with empty input."""
        stats = reporter._calculate_overall_stats([])
        assert stats == {}

    def test_generate_comparisons(self, reporter):
        """Test _generate_comparisons method."""
        document_metrics = [
            {
                'path': '/test/doc1.html',
                'index': 0,
                'total_words': 200,
                'sections': {
                    'Description': {'score': 0.8, 'exists': True},
                    'Installation': {'score': 0.6, 'exists': True}
                }
            },
            {
                'path': '/test/doc2.html',
                'index': 1,
                'total_words': 150,
                'sections': {
                    'Description': {'score': 0.7, 'exists': True},
                    'Installation': {'score': 0.9, 'exists': True}
                }
            }
        ]
        
        all_sections = {
            'Description': [
                {'doc_path': '/test/doc1.html', 'doc_index': 0, 'score': 0.8, 'words': 50, 'exists': True},
                {'doc_path': '/test/doc2.html', 'doc_index': 1, 'score': 0.7, 'words': 45, 'exists': True}
            ],
            'Installation': [
                {'doc_path': '/test/doc1.html', 'doc_index': 0, 'score': 0.6, 'words': 30, 'exists': True},
                {'doc_path': '/test/doc2.html', 'doc_index': 1, 'score': 0.9, 'words': 25, 'exists': True}
            ]
        }
        
        comparisons = reporter._generate_comparisons(document_metrics, all_sections)
        
        # Verify structure
        assert 'best_overall' in comparisons
        assert 'worst_overall' in comparisons
        assert 'most_comprehensive' in comparisons
        assert 'best_sections' in comparisons
        assert 'consistency_analysis' in comparisons
        
        # Most comprehensive should be doc with most words
        assert comparisons['most_comprehensive']['index'] == 0
        assert comparisons['most_comprehensive']['total_words'] == 200
        
        # Best sections
        assert comparisons['best_sections']['Description']['doc_index'] == 0  # Higher score
        assert comparisons['best_sections']['Installation']['doc_index'] == 1  # Higher score
        
        # Consistency analysis
        desc_consistency = comparisons['consistency_analysis']['Description']
        assert 'average_words' in desc_consistency
        assert 'variance' in desc_consistency
        assert 'std_dev' in desc_consistency

    def test_generate_comparisons_empty(self, reporter):
        """Test _generate_comparisons with empty input."""
        comparisons = reporter._generate_comparisons([], {})
        
        assert comparisons['best_overall'] is None
        assert comparisons['worst_overall'] is None
        assert comparisons['most_comprehensive'] is None

    @patch('doc_generator.analysis.reporter.DocumentAnalyzer')
    def test_analyze_full_workflow(self, mock_analyzer_class, reporter, sample_documents):
        """Test full analyze workflow."""
        # Setup mock analyzer
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        reporter.analyzer = mock_analyzer
        
        # Mock section extraction for different documents
        def mock_extract(content):
            if 'doc1' in content:
                return {
                    'Description': '<p>Comprehensive description</p>',
                    'Installation': '<p>Simple installation</p>',
                    'Usage': '<p>Basic usage</p>'
                }
            elif 'doc2' in content:
                return {
                    'Description': '<p>Short description</p>',
                    'Installation': '<p>Comprehensive installation</p>',
                    'Usage': '<p>Advanced usage</p>'
                }
            else:
                return {
                    'Description': '<p>Very detailed description</p>',
                    'Installation': '<p>Installation via pip</p>',
                    'Usage': '<p>Extensive usage guide</p>'
                }
        
        mock_analyzer.extract_sections.side_effect = mock_extract
        mock_analyzer.calculate_section_score.return_value = 0.75
        
        result = reporter.analyze(sample_documents, "Test Topic")
        
        # Verify structure
        assert 'document_metrics' in result
        assert 'section_analysis' in result
        assert 'overall_stats' in result
        assert 'comparisons' in result
        assert 'timestamp' in result
        
        # Verify document metrics
        assert len(result['document_metrics']) == 3
        for doc in result['document_metrics']:
            assert 'path' in doc
            assert 'index' in doc
            assert 'total_length' in doc
            assert 'total_words' in doc
            assert 'sections' in doc
        
        # Verify section analysis
        for section_name in reporter.section_headers:
            assert section_name in result['section_analysis']

    def test_analyze_empty_documents(self, reporter):
        """Test analyze with empty documents."""
        documents = [
            {'path': '/test/empty.html', 'content': ''},
            {'path': '/test/none.html', 'content': None},
            {'path': '/test/missing.html'}  # No content key
        ]
        
        result = reporter.analyze(documents, "Empty Test")
        
        # Should handle gracefully
        assert result['document_metrics'] == []
        assert result['overall_stats'] == {}

    def test_generate_report(self, reporter):
        """Test generate_report method."""
        analysis_results = {
            'timestamp': '2023-01-01T12:00:00',
            'overall_stats': {
                'total_documents': 3,
                'average_length': 1000,
                'average_words': 200,
                'average_code_blocks': 2.0,
                'average_links': 1.5,
                'section_coverage': {
                    'Description': {'count': 3, 'percentage': 100.0},
                    'Installation': {'count': 2, 'percentage': 66.7},
                    'Usage': {'count': 3, 'percentage': 100.0}
                }
            },
            'document_metrics': [
                {
                    'path': '/test/doc1.html',
                    'index': 0,
                    'total_words': 150,
                    'code_blocks': 1,
                    'links': 2,
                    'tables': 0,
                    'lists': 1,
                    'sections': {
                        'Description': {'exists': True, 'score': 0.8, 'words': 50, 'code_blocks': 0},
                        'Installation': {'exists': True, 'score': 0.6, 'words': 30, 'code_blocks': 1},
                        'Usage': {'exists': False, 'score': 0, 'words': 0, 'code_blocks': 0}
                    }
                }
            ],
            'comparisons': {
                'best_overall': {'path': '/test/doc1.html', 'index': 0, 'total_score': 1.4},
                'worst_overall': {'path': '/test/doc2.html', 'index': 1, 'total_score': 1.0},
                'most_comprehensive': {'path': '/test/doc1.html', 'index': 0, 'total_words': 200},
                'best_sections': {
                    'Description': {'doc_index': 0, 'score': 0.8}
                },
                'consistency_analysis': {
                    'Description': {'average_words': 45.0, 'variance': 25.0, 'std_dev': 5.0}
                }
            }
        }
        topic = "Test Topic"
        
        report = reporter.generate_report(analysis_results, topic)
        
        # Verify report structure
        assert '# Documentation Analysis Report: Test Topic' in report
        assert '**Generated:** 2023-01-01T12:00:00' in report
        assert '## Overall Statistics' in report
        assert '**Total Documents:** 3' in report
        assert '**Average Length:** 1000 characters' in report
        assert '**Average Word Count:** 200 words' in report
        
        # Verify section coverage
        assert '### Section Coverage' in report
        assert '**Description:** 3/3 (100%)' in report
        assert '**Installation:** 2/3 (67%)' in report
        
        # Verify document analysis
        assert '## Document Analysis' in report
        assert '### doc1.html' in report
        assert '**Total Words:** 150' in report
        
        # Verify comparisons
        assert '## Cross-Document Comparisons' in report
        assert '**Best Overall:** Document 0 (Score: 1.40)' in report

    def test_generate_report_no_comparisons(self, reporter):
        """Test generate_report with comparisons disabled."""
        reporter.include_comparisons = False
        
        analysis_results = {
            'overall_stats': {'total_documents': 1},
            'document_metrics': [],
            'comparisons': {}
        }
        
        report = reporter.generate_report(analysis_results, "Test Topic")
        
        # Should not include comparisons section
        assert '## Cross-Document Comparisons' not in report

    def test_generate_html_report(self, reporter):
        """Test generate_html_report method."""
        analysis_results = {
            'timestamp': '2023-01-01T12:00:00',
            'overall_stats': {
                'total_documents': 2,
                'average_words': 150,
                'average_code_blocks': 1.5,
                'average_links': 2.0,
                'section_coverage': {
                    'Description': {'count': 2, 'percentage': 100.0},
                    'Installation': {'count': 1, 'percentage': 50.0}
                }
            },
            'document_metrics': [
                {
                    'path': '/test/doc1.html',
                    'index': 0,
                    'total_words': 100,
                    'code_blocks': 1,
                    'links': 1,
                    'tables': 0,
                    'lists': 1,
                    'sections': {
                        'Description': {'exists': True, 'score': 0.8, 'words': 50, 'code_blocks': 0},
                        'Installation': {'exists': False, 'score': 0, 'words': 0, 'code_blocks': 0}
                    }
                }
            ],
            'comparisons': {
                'best_overall': {'index': 0, 'total_score': 1.5}
            }
        }
        topic = "Test Topic"
        
        html_report = reporter.generate_html_report(analysis_results, topic)
        
        # Verify HTML structure
        assert '<!DOCTYPE html>' in html_report
        assert '<title>Documentation Analysis Report: Test Topic</title>' in html_report
        assert '<h1>Documentation Analysis Report: Test Topic</h1>' in html_report
        assert '<h2>Overall Statistics</h2>' in html_report
        assert 'stats-grid' in html_report
        assert '<h2>Document Analysis</h2>' in html_report
        assert '<h2>Cross-Document Comparisons</h2>' in html_report
        assert '</html>' in html_report

    def test_save_artifacts_markdown(self, reporter, temp_dir):
        """Test save_artifacts with markdown format."""
        reporter.formats = ['markdown']
        results = {
            'overall_stats': {'total_documents': 1},
            'document_metrics': [],
            'comparisons': {}
        }
        topic = "Test Topic"
        
        with patch.object(reporter, 'generate_report', return_value='# Test Report'):
            saved_files = reporter.save_artifacts(results, temp_dir, topic)
        
        # Verify markdown report was saved
        assert len(saved_files) == 1
        report_file = temp_dir / 'test_topic_analysis_report.md'
        assert report_file in saved_files
        assert report_file.exists()
        content = report_file.read_text()
        assert '# Test Report' in content

    def test_save_artifacts_html(self, reporter, temp_dir):
        """Test save_artifacts with HTML format."""
        reporter.formats = ['html']
        results = {
            'overall_stats': {'total_documents': 1},
            'document_metrics': [],
            'comparisons': {}
        }
        topic = "Test Topic"
        
        with patch.object(reporter, 'generate_html_report', return_value='<html>Test</html>'):
            saved_files = reporter.save_artifacts(results, temp_dir, topic)
        
        # Verify HTML report was saved
        assert len(saved_files) == 1
        report_file = temp_dir / 'test_topic_analysis_report.html'
        assert report_file in saved_files
        assert report_file.exists()
        content = report_file.read_text()
        assert '<html>Test</html>' in content

    def test_save_artifacts_json(self, reporter, temp_dir):
        """Test save_artifacts with JSON format."""
        reporter.formats = ['json']
        results = {
            'overall_stats': {'total_documents': 1},
            'document_metrics': [
                {'path': Path('/test/doc.html'), 'score': 1.5}  # Path object for serialization test
            ],
            'comparisons': {}
        }
        topic = "Test Topic"
        
        saved_files = reporter.save_artifacts(results, temp_dir, topic)
        
        # Verify JSON data was saved
        assert len(saved_files) == 1
        json_file = temp_dir / 'test_topic_analysis_data.json'
        assert json_file in saved_files
        assert json_file.exists()
        
        # Verify JSON content
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert data['overall_stats']['total_documents'] == 1
        # Path should be converted to string
        assert isinstance(data['document_metrics'][0]['path'], str)

    def test_save_artifacts_all_formats(self, reporter, temp_dir):
        """Test save_artifacts with all formats."""
        reporter.formats = ['markdown', 'html', 'json']
        results = {
            'overall_stats': {'total_documents': 1},
            'document_metrics': [],
            'comparisons': {}
        }
        topic = "Test Topic"
        
        with patch.object(reporter, 'generate_report', return_value='# Test Report'):
            with patch.object(reporter, 'generate_html_report', return_value='<html>Test</html>'):
                saved_files = reporter.save_artifacts(results, temp_dir, topic)
        
        # Should save all three formats
        assert len(saved_files) == 3
        
        # Verify all files exist
        md_file = temp_dir / 'test_topic_analysis_report.md'
        html_file = temp_dir / 'test_topic_analysis_report.html'
        json_file = temp_dir / 'test_topic_analysis_data.json'
        
        assert md_file in saved_files and md_file.exists()
        assert html_file in saved_files and html_file.exists()
        assert json_file in saved_files and json_file.exists()

    def test_make_json_serializable(self, reporter):
        """Test _make_json_serializable method."""
        test_data = {
            'path': Path('/test/path'),
            'nested': {
                'path': Path('/nested/path'),
                'number': 42
            },
            'list': [Path('/list/path'), 'string', {'nested_path': Path('/nested/list/path')}],
            'string': 'regular_string',
            'number': 123
        }
        
        result = reporter._make_json_serializable(test_data)
        
        # Paths should be converted to strings
        assert result['path'] == '/test/path'
        assert result['nested']['path'] == '/nested/path'
        assert result['list'][0] == '/list/path'
        assert result['list'][2]['nested_path'] == '/nested/list/path'
        
        # Other types should remain unchanged
        assert result['nested']['number'] == 42
        assert result['list'][1] == 'string'
        assert result['string'] == 'regular_string'
        assert result['number'] == 123


class TestAnalysisReporterIntegration:
    """Integration tests for AnalysisReporter with minimal mocking."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_full_analysis_workflow(self, temp_dir):
        """Test complete analysis workflow with real DocumentAnalyzer."""
        logger = Mock(spec=logging.Logger)
        reporter = AnalysisReporter(logger=logger)
        
        # Sample documents with realistic content
        documents = [
            {
                'path': str(temp_dir / 'doc1.html'),
                'content': '''
                <html>
                <body>
                    <h2>Description</h2>
                    <p>This is a comprehensive description with detailed information about the library.</p>
                    <p>It provides multiple features and capabilities for users.</p>
                    <h2>Installation</h2>
                    <pre><code>pip install mylibrary</code></pre>
                    <h2>Usage</h2>
                    <p>Basic usage example:</p>
                    <pre><code>import mylibrary
mylibrary.do_something()</code></pre>
                    <a href="https://example.com">Documentation</a>
                </body>
                </html>
                '''
            },
            {
                'path': str(temp_dir / 'doc2.html'),
                'content': '''
                <html>
                <body>
                    <h2>Description</h2>
                    <p>Brief description of the library functionality.</p>
                    <h2>Installation</h2>
                    <p>Install from source:</p>
                    <pre><code>git clone https://github.com/user/repo.git</code></pre>
                    <pre><code>cd repo && python setup.py install</code></pre>
                    <h2>Usage</h2>
                    <p>Advanced usage patterns:</p>
                    <ul>
                        <li>Feature A</li>
                        <li>Feature B</li>
                    </ul>
                </body>
                </html>
                '''
            }
        ]
        
        # Run analysis
        result = reporter.analyze(documents, "Integration Test Library")
        
        # Verify results structure
        assert 'document_metrics' in result
        assert 'section_analysis' in result
        assert 'overall_stats' in result
        assert 'comparisons' in result
        
        # Should have analyzed both documents
        assert len(result['document_metrics']) == 2
        assert result['overall_stats']['total_documents'] == 2
        
        # Generate reports
        markdown_report = reporter.generate_report(result, "Integration Test Library")
        html_report = reporter.generate_html_report(result, "Integration Test Library")
        
        assert 'Documentation Analysis Report' in markdown_report
        assert '<title>Documentation Analysis Report' in html_report
        
        # Save artifacts
        saved_files = reporter.save_artifacts(result, temp_dir, "Integration Test Library")
        assert len(saved_files) >= 1
        
        # Verify report file exists and contains expected content
        report_file = temp_dir / 'integration_test_library_analysis_report.md'
        assert report_file.exists()
        report_content = report_file.read_text()
        assert 'Documentation Analysis Report' in report_content
        assert 'Overall Statistics' in report_content