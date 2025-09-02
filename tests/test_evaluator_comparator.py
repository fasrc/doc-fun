"""
Tests for the DocumentationComparator from the evaluator package.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
from datetime import datetime

from doc_generator.evaluator.comparator import DocumentationComparator


class TestDocumentationComparator:
    """Test the DocumentationComparator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_generator(self):
        """Create mock DocumentationGenerator."""
        return Mock()
    
    @pytest.fixture
    def mock_downloader(self):
        """Create mock DocumentationDownloader."""
        return Mock()
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock SimilarityMetrics."""
        mock = Mock()
        mock.sequence_similarity.return_value = 0.75
        mock.structural_similarity.return_value = 0.80
        mock.code_similarity.return_value = 0.65
        mock.semantic_similarity.return_value = 0.70
        mock.jaccard_similarity.return_value = 0.68
        mock.cosine_similarity.return_value = 0.72
        mock.calculate_composite_score.return_value = 0.73
        return mock
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create mock DocumentAnalyzer."""
        mock = Mock()
        mock.extract_sections.return_value = {
            'Description': '<p>Test description content</p>',
            'Installation': '<pre><code>pip install test</code></pre>',
            'Usage': '<p>Usage instructions</p>'
        }
        return mock
    
    @pytest.fixture
    def comparator(self, mock_generator, mock_downloader, mock_metrics, mock_analyzer):
        """Create DocumentationComparator with mocked dependencies."""
        with patch('doc_generator.evaluator.comparator.DocumentationGenerator', return_value=mock_generator):
            with patch('doc_generator.evaluator.comparator.DocumentationDownloader', return_value=mock_downloader):
                with patch('doc_generator.evaluator.comparator.SimilarityMetrics', return_value=mock_metrics):
                    with patch('doc_generator.evaluator.comparator.DocumentAnalyzer', return_value=mock_analyzer):
                        comparator = DocumentationComparator(mock_generator)
                        comparator.downloader = mock_downloader
                        comparator.metrics = mock_metrics
                        comparator.analyzer = mock_analyzer
                        return comparator
    
    @pytest.fixture
    def sample_reference_doc(self):
        """Sample reference document data."""
        return {
            'raw_text': 'This is a comprehensive guide about testing. Installation: pip install pytest. Usage: Run pytest in your terminal.',
            'sections': [
                {'title': 'Description', 'content': 'This is a comprehensive guide about testing.', 'level': 2},
                {'title': 'Installation', 'content': 'pip install pytest', 'level': 2},
                {'title': 'Usage', 'content': 'Run pytest in your terminal.', 'level': 2}
            ],
            'code_examples': [
                {'code': 'pip install pytest', 'language': 'bash'},
                {'code': 'pytest', 'language': 'bash'}
            ],
            'metadata': {'title': 'Testing Guide', 'description': 'A guide to testing'}
        }
    
    @pytest.fixture
    def sample_generated_html(self):
        """Sample generated HTML content."""
        return '''
        <html>
        <head><title>Generated Testing Guide</title></head>
        <body>
            <h2>Description</h2>
            <p>This is a guide about testing frameworks and best practices.</p>
            <h2>Installation</h2>
            <pre><code class="language-bash">pip install pytest</code></pre>
            <h2>Usage</h2>
            <p>Execute tests using: <code>pytest</code></p>
            <ul><li>Run all tests</li><li>Generate reports</li></ul>
        </body>
        </html>
        '''

    def test_init_with_generator(self, mock_generator):
        """Test initialization with provided generator."""
        with patch('doc_generator.evaluator.comparator.DocumentationDownloader'):
            with patch('doc_generator.evaluator.comparator.SimilarityMetrics'):
                with patch('doc_generator.evaluator.comparator.DocumentAnalyzer'):
                    comparator = DocumentationComparator(mock_generator)
                    assert comparator.generator == mock_generator

    def test_init_without_generator(self):
        """Test initialization without provided generator."""
        with patch('doc_generator.evaluator.comparator.DocumentationGenerator') as mock_gen_class:
            with patch('doc_generator.evaluator.comparator.DocumentationDownloader'):
                with patch('doc_generator.evaluator.comparator.SimilarityMetrics'):
                    with patch('doc_generator.evaluator.comparator.DocumentAnalyzer'):
                        mock_gen_instance = Mock()
                        mock_gen_class.return_value = mock_gen_instance
                        
                        comparator = DocumentationComparator()
                        assert comparator.generator == mock_gen_instance
                        mock_gen_class.assert_called_once()

    def test_extract_generated_content(self, comparator, sample_generated_html):
        """Test _extract_generated_content method."""
        result = comparator._extract_generated_content(sample_generated_html)
        
        # Verify structure
        assert 'raw_text' in result
        assert 'sections' in result
        assert 'code_examples' in result
        assert 'metadata' in result
        assert 'content' in result
        
        # Verify content extraction
        assert 'Generated Testing Guide' in result['metadata']['title']
        assert len(result['sections']) == 3  # Based on mock analyzer
        assert 'Description' in [s['title'] for s in result['sections']]
        
        # Verify raw text contains main content
        assert 'guide about testing' in result['raw_text']

    def test_detect_language(self, comparator):
        """Test _detect_language method."""
        # Mock code element with language class
        code_element = Mock()
        code_element.get.return_value = ['highlight', 'language-python', 'other-class']
        
        language = comparator._detect_language(code_element)
        assert language == 'python'
        
        # Test without language class
        code_element.get.return_value = ['highlight', 'other-class']
        language = comparator._detect_language(code_element)
        assert language == ''

    def test_compare_documents(self, comparator, sample_reference_doc):
        """Test _compare_documents method."""
        generated_doc = {
            'raw_text': 'This is a guide about testing frameworks. Installation: pip install pytest. Usage: Execute tests.',
            'sections': [
                {'title': 'Description', 'content': 'This is a guide about testing frameworks.', 'level': 2},
                {'title': 'Installation', 'content': 'pip install pytest', 'level': 2}
            ],
            'code_examples': [
                {'code': 'pip install pytest', 'language': 'bash'}
            ],
            'metadata': {'title': 'Testing Guide'}
        }
        
        result = comparator._compare_documents(sample_reference_doc, generated_doc)
        
        # Verify structure
        assert 'scores' in result
        assert 'details' in result
        assert 'recommendations' in result
        
        # Verify scores (based on mock returns)
        assert result['scores']['content_similarity'] == 0.75
        assert result['scores']['structural_similarity'] == 0.80
        assert result['scores']['composite_score'] == 0.73
        
        # Verify metrics methods were called
        comparator.metrics.sequence_similarity.assert_called()
        comparator.metrics.structural_similarity.assert_called()
        comparator.metrics.calculate_composite_score.assert_called()

    def test_analyze_differences(self, comparator):
        """Test _analyze_differences method."""
        reference = {
            'sections': [
                {'title': 'Description', 'content': 'Description content'},
                {'title': 'Installation', 'content': 'Installation content'},
                {'title': 'Usage', 'content': 'Usage content'}
            ],
            'code_examples': [
                {'code': 'example1', 'language': 'python'},
                {'code': 'example2', 'language': 'bash'}
            ],
            'raw_text': 'This is reference text with comprehensive content.'
        }
        
        generated = {
            'sections': [
                {'title': 'Description', 'content': 'Description content'},
                {'title': 'Examples', 'content': 'Examples content'}  # Different section
            ],
            'code_examples': [
                {'code': 'example1', 'language': 'python'}
            ],
            'raw_text': 'This is generated text.'
        }
        
        details = comparator._analyze_differences(reference, generated)
        
        # Verify section analysis
        assert 'installation' in details['missing_sections']
        assert 'usage' in details['missing_sections']
        assert 'examples' in details['extra_sections']
        assert 'description' in details['common_sections']
        
        # Verify code analysis
        assert details['reference_code_count'] == 2
        assert details['generated_code_count'] == 1
        
        # Verify length analysis
        assert details['reference_length'] == len(reference['raw_text'])
        assert details['generated_length'] == len(generated['raw_text'])
        assert 'length_ratio' in details

    def test_generate_recommendations_low_similarity(self, comparator):
        """Test _generate_recommendations with low similarity scores."""
        scores = {
            'content_similarity': 0.2,
            'structural_similarity': 0.3,
            'code_similarity': 0.1,
            'semantic_similarity': 0.25,
            'composite_score': 0.3
        }
        
        details = {
            'missing_sections': ['installation', 'usage'],
            'generated_code_count': 0,
            'reference_code_count': 3,
            'length_ratio': 0.4
        }
        
        recommendations = comparator._generate_recommendations(scores, details)
        
        # Should have multiple recommendations for improvement
        assert len(recommendations) > 3
        assert any('Low content similarity' in rec for rec in recommendations)
        assert any('Low structural similarity' in rec for rec in recommendations)
        assert any('Missing sections' in rec for rec in recommendations)
        assert any('No code examples' in rec for rec in recommendations)

    def test_generate_recommendations_high_similarity(self, comparator):
        """Test _generate_recommendations with high similarity scores."""
        scores = {
            'content_similarity': 0.85,
            'structural_similarity': 0.90,
            'code_similarity': 0.88,
            'semantic_similarity': 0.82,
            'composite_score': 0.85
        }
        
        details = {
            'missing_sections': [],
            'generated_code_count': 3,
            'reference_code_count': 3,
            'length_ratio': 1.1
        }
        
        recommendations = comparator._generate_recommendations(scores, details)
        
        # Should have positive recommendations
        assert any('Excellent similarity' in rec for rec in recommendations)

    def test_compare_with_url_success(self, comparator, sample_reference_doc, temp_dir):
        """Test compare_with_url method with successful comparison."""
        # Setup mocks
        comparator.downloader.download_and_extract.return_value = sample_reference_doc
        
        generated_file = temp_dir / "generated.html"
        generated_file.write_text('''
        <html><body>
        <h2>Description</h2><p>Test content</p>
        <h2>Installation</h2><pre><code>pip install test</code></pre>
        </body></html>
        ''')
        
        comparator.generator.generate_documentation.return_value = [str(generated_file)]
        
        # Mock comparison results
        with patch.object(comparator, '_compare_documents') as mock_compare:
            mock_compare.return_value = {
                'scores': {'composite_score': 0.75},
                'details': {'missing_sections': []},
                'recommendations': ['Good similarity']
            }
            
            result = comparator.compare_with_url(
                topic="Test Topic",
                reference_url="https://example.com/docs",
                generation_params={'runs': 1}
            )
        
        # Verify downloader was called
        comparator.downloader.download_and_extract.assert_called_once_with("https://example.com/docs")
        
        # Verify generator was called
        comparator.generator.generate_documentation.assert_called_once()
        
        # Verify result structure
        assert 'scores' in result
        assert 'details' in result
        assert 'recommendations' in result
        assert 'metadata' in result
        assert result['metadata']['topic'] == "Test Topic"
        assert result['metadata']['reference_url'] == "https://example.com/docs"

    def test_compare_with_url_generation_failure(self, comparator, sample_reference_doc):
        """Test compare_with_url with generation failure."""
        comparator.downloader.download_and_extract.return_value = sample_reference_doc
        comparator.generator.generate_documentation.return_value = []  # Empty result
        
        with pytest.raises(ValueError, match="Failed to generate documentation"):
            comparator.compare_with_url("Test Topic", "https://example.com/docs")

    def test_compare_existing_files_html_reference(self, comparator, temp_dir):
        """Test compare_existing_files with HTML reference file."""
        # Create test files
        generated_file = temp_dir / "generated.html"
        generated_file.write_text('<html><body><h1>Generated Content</h1></body></html>')
        
        reference_file = temp_dir / "reference.html"
        reference_file.write_text('<html><body><h1>Reference Content</h1></body></html>')
        
        # Mock downloader extract_content
        comparator.downloader.extract_content.return_value = {
            'raw_text': 'Reference Content',
            'sections': [],
            'code_examples': [],
            'metadata': {}
        }
        
        # Mock comparison results
        with patch.object(comparator, '_compare_documents') as mock_compare:
            mock_compare.return_value = {
                'scores': {'composite_score': 0.65},
                'details': {},
                'recommendations': []
            }
            
            result = comparator.compare_existing_files(
                str(generated_file),
                str(reference_file)
            )
        
        # Verify downloader was called for HTML processing
        comparator.downloader.extract_content.assert_called_once()
        
        # Verify comparison was performed
        mock_compare.assert_called_once()
        assert 'scores' in result

    def test_compare_existing_files_text_reference(self, comparator, temp_dir):
        """Test compare_existing_files with text reference file."""
        # Create test files
        generated_file = temp_dir / "generated.html"
        generated_file.write_text('<html><body><h1>Generated Content</h1></body></html>')
        
        reference_file = temp_dir / "reference.md"
        reference_file.write_text('# Reference Content\n\nThis is markdown content.')
        
        # Mock comparison results
        with patch.object(comparator, '_compare_documents') as mock_compare:
            mock_compare.return_value = {
                'scores': {'composite_score': 0.55},
                'details': {},
                'recommendations': []
            }
            
            result = comparator.compare_existing_files(
                str(generated_file),
                str(reference_file)
            )
        
        # Verify comparison was performed
        mock_compare.assert_called_once()
        
        # Get the reference document passed to comparison
        call_args = mock_compare.call_args[0]
        reference_doc = call_args[0]
        
        # Should be treated as plain text
        assert reference_doc['raw_text'] == '# Reference Content\n\nThis is markdown content.'
        assert reference_doc['sections'] == []

    def test_generate_report(self, comparator, temp_dir):
        """Test generate_report method."""
        comparison_results = {
            'metadata': {
                'topic': 'Test Topic',
                'reference_url': 'https://example.com',
                'generated_file': 'test.html'
            },
            'scores': {
                'composite_score': 0.75,
                'content_similarity': 0.80,
                'structural_similarity': 0.70,
                'code_similarity': 0.65,
                'semantic_similarity': 0.85,
                'jaccard_similarity': 0.72,
                'cosine_similarity': 0.78
            },
            'details': {
                'missing_sections': ['examples'],
                'extra_sections': [],
                'common_sections': ['description', 'installation'],
                'reference_length': 1000,
                'generated_length': 800,
                'length_ratio': 0.8,
                'reference_code_count': 3,
                'generated_code_count': 2,
                'reference_languages': ['python', 'bash'],
                'generated_languages': ['python']
            },
            'recommendations': [
                'Good similarity score',
                'Consider adding examples section',
                'Add more code examples'
            ]
        }
        
        # Test without saving to file
        report = comparator.generate_report(comparison_results)
        
        # Verify report structure
        assert '# Documentation Comparison Report' in report
        assert '## Metadata' in report
        assert '## Similarity Scores' in report
        assert '## Detailed Analysis' in report
        assert '## Recommendations' in report
        assert '## Quality Assessment' in report
        
        # Verify content
        assert 'Test Topic' in report
        assert '75.00%' in report  # Composite score
        assert 'Good similarity score' in report
        assert '⭐⭐⭐⭐ Good' in report  # Quality assessment for 0.75
        
        # Test with saving to file
        output_file = temp_dir / "comparison_report.md"
        report_with_file = comparator.generate_report(comparison_results, str(output_file))
        
        assert report == report_with_file
        assert output_file.exists()
        assert output_file.read_text() == report

    def test_generate_report_quality_ratings(self, comparator):
        """Test generate_report quality rating system."""
        base_results = {
            'scores': {},
            'details': {},
            'recommendations': []
        }
        
        # Test different quality levels
        quality_tests = [
            (0.9, '⭐⭐⭐⭐⭐ Excellent'),
            (0.7, '⭐⭐⭐⭐ Good'),
            (0.5, '⭐⭐⭐ Acceptable'),
            (0.3, '⭐⭐ Needs Improvement'),
            (0.1, '⭐ Poor')
        ]
        
        for score, expected_rating in quality_tests:
            results = {**base_results, 'scores': {'composite_score': score}}
            report = comparator.generate_report(results)
            assert expected_rating in report

    def test_error_handling_file_not_found(self, comparator):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            comparator.compare_existing_files(
                "/nonexistent/generated.html",
                "/nonexistent/reference.html"
            )

    def test_empty_content_handling(self, comparator, temp_dir):
        """Test handling of empty content files."""
        # Create empty files
        generated_file = temp_dir / "generated.html"
        generated_file.write_text('')
        
        reference_file = temp_dir / "reference.html"
        reference_file.write_text('')
        
        # Mock downloader to return empty structure
        comparator.downloader.extract_content.return_value = {
            'raw_text': '',
            'sections': [],
            'code_examples': [],
            'metadata': {}
        }
        
        # Mock comparison to handle empty content
        with patch.object(comparator, '_compare_documents') as mock_compare:
            mock_compare.return_value = {
                'scores': {'composite_score': 0.0},
                'details': {'missing_sections': [], 'length_ratio': 0},
                'recommendations': ['Both documents are empty']
            }
            
            result = comparator.compare_existing_files(
                str(generated_file),
                str(reference_file)
            )
        
        assert result['scores']['composite_score'] == 0.0


class TestDocumentationComparatorIntegration:
    """Integration tests for DocumentationComparator with minimal mocking."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_extract_generated_content_real_parsing(self, temp_dir):
        """Test _extract_generated_content with real HTML parsing."""
        # Create comparator with minimal mocking
        with patch('doc_generator.evaluator.comparator.DocumentationGenerator'):
            with patch('doc_generator.evaluator.comparator.DocumentationDownloader'):
                with patch('doc_generator.evaluator.comparator.SimilarityMetrics'):
                    comparator = DocumentationComparator()
        
        html_content = '''
        <!DOCTYPE html>
        <html>
        <head><title>Test Documentation</title></head>
        <body>
            <h1>Main Title</h1>
            <h2>Description</h2>
            <p>This is a comprehensive description of the library.</p>
            <p>It provides multiple features for users.</p>
            
            <h2>Installation</h2>
            <pre><code class="language-bash">pip install mylib</code></pre>
            <pre><code class="language-python">import mylib</code></pre>
            
            <h2>Usage</h2>
            <p>Basic usage example:</p>
            <code>mylib.function()</code>
            <ul>
                <li>Feature 1</li>
                <li>Feature 2</li>
            </ul>
        </body>
        </html>
        '''
        
        result = comparator._extract_generated_content(html_content)
        
        # Verify structure
        assert 'raw_text' in result
        assert 'sections' in result
        assert 'code_examples' in result
        assert 'metadata' in result
        
        # Verify metadata extraction
        assert result['metadata']['title'] == 'Test Documentation'
        
        # Verify raw text contains main content
        assert 'comprehensive description' in result['raw_text']
        assert 'pip install mylib' in result['raw_text']
        
        # Verify code extraction
        code_texts = [ex['code'] for ex in result['code_examples']]
        assert 'pip install mylib' in code_texts
        assert 'import mylib' in code_texts
        assert 'mylib.function()' in code_texts

    def test_language_detection_integration(self, temp_dir):
        """Test language detection with real HTML elements."""
        with patch('doc_generator.evaluator.comparator.DocumentationGenerator'):
            with patch('doc_generator.evaluator.comparator.DocumentationDownloader'):
                with patch('doc_generator.evaluator.comparator.SimilarityMetrics'):
                    comparator = DocumentationComparator()
        
        # Test HTML with language classes
        from bs4 import BeautifulSoup
        html = '<pre><code class="language-python highlight">print("hello")</code></pre>'
        soup = BeautifulSoup(html, 'html.parser')
        code_element = soup.find('code')
        
        language = comparator._detect_language(code_element)
        assert language == 'python'
        
        # Test without language class
        html = '<pre><code class="highlight">print("hello")</code></pre>'
        soup = BeautifulSoup(html, 'html.parser')
        code_element = soup.find('code')
        
        language = comparator._detect_language(code_element)
        assert language == ''