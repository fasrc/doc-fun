"""
Tests for the LinkValidator analysis plugin.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import logging
import requests
import time
import threading

from doc_generator.analysis.link_validator import LinkValidator


class TestLinkValidator:
    """Test the LinkValidator analysis plugin."""
    
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
        """Sample configuration for LinkValidator."""
        return {
            'timeout': 5,
            'retries': 1,
            'retry_delay': 0.1,
            'max_workers': 2,
            'check_internal': True,
            'user_agent': 'TestAgent/1.0',
            'ignore_patterns': [r'^mailto:', r'^#anchor'],
            'validate_ssl': False,
            'report_format': 'markdown'
        }
    
    @pytest.fixture
    def validator(self, mock_logger, sample_config):
        """Create LinkValidator instance."""
        return LinkValidator(logger=mock_logger, config=sample_config)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents with various types of links."""
        return [
            {
                'path': '/test/doc1.html',
                'content': '''
                <html>
                <body>
                    <h1>Test Document</h1>
                    <p>Here are some links:</p>
                    <a href="https://www.google.com">Google</a>
                    <a href="https://www.github.com">GitHub</a>
                    <a href="mailto:test@example.com">Email</a>
                    <a href="#section1">Internal Link</a>
                    <img src="https://example.com/image.jpg" alt="Image">
                    <link href="https://example.com/style.css" rel="stylesheet">
                </body>
                </html>
                '''
            },
            {
                'path': '/test/doc2.html', 
                'content': '''
                <html>
                <body>
                    <p>More links:</p>
                    <a href="https://www.google.com">Google Again</a>
                    <a href="https://nonexistent.invalid">Broken Link</a>
                    <a href="javascript:alert('test')">JavaScript</a>
                    <script src="https://example.com/script.js"></script>
                </body>
                </html>
                '''
            }
        ]

    def test_init_default_config(self, mock_logger):
        """Test LinkValidator initialization with default config."""
        validator = LinkValidator(logger=mock_logger)
        
        assert validator.logger == mock_logger
        assert validator.timeout == 10
        assert validator.retries == 2
        assert validator.retry_delay == 1
        assert validator.max_workers == 5
        assert validator.check_internal == False
        assert validator.validate_ssl == True
        assert validator.report_format == 'markdown'
        assert 'Mozilla/5.0' in validator.user_agent
        assert len(validator.ignore_patterns) > 0
        assert hasattr(validator, 'session')
        assert isinstance(validator.session, requests.Session)

    def test_init_custom_config(self, validator, sample_config):
        """Test LinkValidator initialization with custom config."""
        assert validator.timeout == sample_config['timeout']
        assert validator.retries == sample_config['retries']
        assert validator.retry_delay == sample_config['retry_delay']
        assert validator.max_workers == sample_config['max_workers']
        assert validator.check_internal == sample_config['check_internal']
        assert validator.user_agent == sample_config['user_agent']
        assert validator.ignore_patterns == sample_config['ignore_patterns']
        assert validator.validate_ssl == sample_config['validate_ssl']

    def test_get_name(self, validator):
        """Test get_name method."""
        assert validator.get_name() == 'link_validator'

    def test_get_priority(self, validator):
        """Test get_priority method."""
        assert validator.get_priority() == 50

    def test_get_supported_formats(self, validator):
        """Test get_supported_formats method."""
        assert validator.get_supported_formats() == ['markdown', 'text']

    def test_extract_links(self, validator):
        """Test _extract_links method."""
        html_content = '''
        <html>
        <body>
            <a href="https://www.example.com">Link 1</a>
            <a href="mailto:test@example.com">Email</a>
            <a href="#section">Anchor</a>
            <a href="">Empty href</a>
            <a>No href</a>
            <img src="https://example.com/image.jpg" alt="Image">
            <script src="https://example.com/script.js"></script>
            <link href="https://example.com/style.css" rel="stylesheet">
            <img src="local-image.jpg" alt="Local Image">
        </body>
        </html>
        '''
        
        links = validator._extract_links(html_content)
        
        expected_links = [
            'https://www.example.com',
            'mailto:test@example.com',
            '#section',
            'https://example.com/image.jpg',
            'https://example.com/script.js',
            'https://example.com/style.css'
        ]
        
        # Sort both lists for comparison since order might vary
        assert sorted(links) == sorted(expected_links)

    def test_should_ignore(self, validator):
        """Test _should_ignore method."""
        # Test configured ignore patterns
        assert validator._should_ignore('mailto:test@example.com') == True
        assert validator._should_ignore('#anchor') == True
        assert validator._should_ignore('https://www.example.com') == False
        
        # Test default patterns (not in sample config)
        validator.ignore_patterns = [
            r'^mailto:',
            r'^tel:',
            r'^ftp:',
            r'^javascript:',
            r'^#',
            r'^localhost',
            r'^127\.0\.0\.1',
            r'^192\.168\.',
        ]
        
        assert validator._should_ignore('tel:+1234567890') == True
        assert validator._should_ignore('ftp://example.com') == True
        assert validator._should_ignore('javascript:alert()') == True
        assert validator._should_ignore('localhost:8080') == True
        assert validator._should_ignore('127.0.0.1:3000') == True
        assert validator._should_ignore('192.168.1.1') == True
        assert validator._should_ignore('https://www.example.com') == False

    def test_should_ignore_internal_anchors(self, mock_logger):
        """Test ignoring internal anchors when check_internal is False."""
        config = {'check_internal': False, 'ignore_patterns': []}
        validator = LinkValidator(logger=mock_logger, config=config)
        
        assert validator._should_ignore('#section') == True
        
        # When check_internal is True, should not ignore
        config['check_internal'] = True
        validator = LinkValidator(logger=mock_logger, config=config)
        assert validator._should_ignore('#section') == False

    @patch('doc_generator.analysis.link_validator.LinkValidator._validate_links_concurrent')
    def test_analyze_basic_flow(self, mock_validate, validator, sample_documents):
        """Test basic analyze flow without actual network calls."""
        # Mock the concurrent validation to avoid network calls
        def mock_validation(urls, all_links):
            for url in urls:
                if 'google.com' in url:
                    all_links[url]['status'] = 'valid'
                    all_links[url]['status_code'] = 200
                    all_links[url]['message'] = 'OK'
                elif 'nonexistent.invalid' in url:
                    all_links[url]['status'] = 'broken'
                    all_links[url]['status_code'] = 404
                    all_links[url]['message'] = 'Not Found'
                else:
                    all_links[url]['status'] = 'valid'
                    all_links[url]['status_code'] = 200
                    all_links[url]['message'] = 'OK'
        
        mock_validate.side_effect = mock_validation
        
        result = validator.analyze(sample_documents, "Test Topic")
        
        # Verify structure
        assert 'total_links' in result
        assert 'unique_links' in result
        assert 'valid_links' in result
        assert 'broken_links' in result
        assert 'skipped_links' in result
        assert 'link_details' in result
        assert 'summary' in result
        
        # Verify skipped links (mailto and #anchor based on sample_config)
        skipped = result['skipped_links']
        assert any('mailto:' in link for link in skipped)
        
        # Verify summary
        summary = result['summary']
        assert 'total' in summary
        assert 'valid' in summary
        assert 'broken' in summary
        assert 'skipped' in summary
        assert 'success_rate' in summary

    def test_analyze_empty_documents(self, validator):
        """Test analyze with empty or missing content."""
        documents = [
            {'path': '/test/empty.html', 'content': ''},
            {'path': '/test/none.html', 'content': None},
            {'path': '/test/missing.html'}  # No content key
        ]
        
        result = validator.analyze(documents, "Empty Test")
        
        # Should handle gracefully
        assert result['total_links'] == 0
        assert result['unique_links'] == 0
        assert len(result['valid_links']) == 0
        assert len(result['broken_links']) == 0
        assert result['summary']['success_rate'] == 100  # No links to check

    @patch('requests.Session.head')
    def test_validate_single_link_success(self, mock_head, validator):
        """Test _validate_single_link with successful response."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response
        
        status, status_code, message, response_time = validator._validate_single_link('https://example.com')
        
        assert status == 'valid'
        assert status_code == 200
        assert message == 'OK'
        assert response_time is not None
        assert response_time >= 0

    @patch('requests.Session.head')
    @patch('requests.Session.get')
    def test_validate_single_link_method_not_allowed(self, mock_get, mock_head, validator):
        """Test _validate_single_link falling back to GET when HEAD fails."""
        # Mock HEAD returning 405, GET returning 200
        mock_head_response = Mock()
        mock_head_response.status_code = 405
        mock_head.return_value = mock_head_response
        
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.close = Mock()
        mock_get.return_value = mock_get_response
        
        status, status_code, message, response_time = validator._validate_single_link('https://example.com')
        
        assert status == 'valid'
        assert status_code == 200
        assert message == 'OK'
        mock_get.assert_called_once()
        mock_get_response.close.assert_called_once()

    @patch('requests.Session.head')
    def test_validate_single_link_not_found(self, mock_head, validator):
        """Test _validate_single_link with 404 response."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response
        
        status, status_code, message, response_time = validator._validate_single_link('https://example.com/notfound')
        
        assert status == 'broken'
        assert status_code == 404
        assert message == 'Not Found'

    @patch('requests.Session.head')
    def test_validate_single_link_server_error_with_retry(self, mock_head, validator):
        """Test _validate_single_link with server error and retries."""
        # First call returns 500, second call returns 200
        mock_responses = [Mock(), Mock()]
        mock_responses[0].status_code = 500
        mock_responses[1].status_code = 200
        mock_head.side_effect = mock_responses
        
        # Reduce retry delay for faster test
        validator.retry_delay = 0.01
        
        status, status_code, message, response_time = validator._validate_single_link('https://example.com')
        
        assert status == 'valid'
        assert status_code == 200
        assert message == 'OK'
        assert mock_head.call_count == 2

    @patch('requests.Session.head')
    def test_validate_single_link_timeout(self, mock_head, validator):
        """Test _validate_single_link with timeout."""
        mock_head.side_effect = requests.exceptions.Timeout()
        
        status, status_code, message, response_time = validator._validate_single_link('https://example.com')
        
        assert status == 'broken'
        assert status_code is None
        assert message == 'Timeout'

    @patch('requests.Session.head')
    def test_validate_single_link_ssl_error(self, mock_head, validator):
        """Test _validate_single_link with SSL error."""
        mock_head.side_effect = requests.exceptions.SSLError()
        
        status, status_code, message, response_time = validator._validate_single_link('https://example.com')
        
        assert status == 'broken'
        assert status_code is None
        assert message == 'SSL Certificate Error'

    @patch('requests.Session.head')
    def test_validate_single_link_connection_error(self, mock_head, validator):
        """Test _validate_single_link with connection error."""
        mock_head.side_effect = requests.exceptions.ConnectionError()
        
        status, status_code, message, response_time = validator._validate_single_link('https://example.com')
        
        assert status == 'broken'
        assert status_code is None
        assert message == 'Connection Failed'

    @patch('requests.Session.head')
    def test_validate_single_link_too_many_redirects(self, mock_head, validator):
        """Test _validate_single_link with too many redirects."""
        mock_head.side_effect = requests.exceptions.TooManyRedirects()
        
        status, status_code, message, response_time = validator._validate_single_link('https://example.com')
        
        assert status == 'broken'
        assert status_code is None
        assert message == 'Too Many Redirects'

    @patch('requests.Session.head')
    def test_validate_single_link_generic_exception(self, mock_head, validator):
        """Test _validate_single_link with generic exception."""
        mock_head.side_effect = Exception("Generic error")
        
        status, status_code, message, response_time = validator._validate_single_link('https://example.com')
        
        assert status == 'broken'
        assert status_code is None
        assert 'Generic error' in message

    def test_validate_links_concurrent(self, validator):
        """Test _validate_links_concurrent with mocked validation."""
        urls = ['https://example.com', 'https://test.com']
        all_links = {
            'https://example.com': {'documents': [], 'status': None},
            'https://test.com': {'documents': [], 'status': None}
        }
        
        # Mock the single link validation to avoid network calls
        def mock_validate(url):
            if 'example.com' in url:
                return ('valid', 200, 'OK', 0.1)
            else:
                return ('broken', 404, 'Not Found', 0.2)
        
        with patch.object(validator, '_validate_single_link', side_effect=mock_validate):
            validator._validate_links_concurrent(urls, all_links)
        
        # Verify results were updated
        assert all_links['https://example.com']['status'] == 'valid'
        assert all_links['https://example.com']['status_code'] == 200
        assert all_links['https://test.com']['status'] == 'broken'
        assert all_links['https://test.com']['status_code'] == 404

    def test_generate_report_with_results(self, validator):
        """Test generate_report with analysis results."""
        analysis_results = {
            'summary': {
                'total': 10,
                'valid': 7,
                'broken': 2,
                'skipped': 1,
                'success_rate': 77.8
            },
            'valid_links': [
                'https://www.google.com',
                'https://www.github.com',
                'https://example.com/valid'
            ],
            'broken_links': [
                'https://nonexistent.invalid',
                'https://broken.example.com'
            ],
            'skipped_links': [
                'mailto:test@example.com'
            ],
            'link_details': {
                'https://nonexistent.invalid': {
                    'status_code': 404,
                    'message': 'Not Found',
                    'documents': ['/test/doc1.html', '/test/doc2.html'],
                    'response_time': 0.5
                },
                'https://broken.example.com': {
                    'status_code': None,
                    'message': 'Timeout',
                    'documents': ['/test/doc1.html'],
                    'response_time': None
                },
                'https://www.google.com': {
                    'response_time': 0.1
                },
                'https://www.github.com': {
                    'response_time': 0.2
                }
            }
        }
        topic = "Test Topic"
        
        report = validator.generate_report(analysis_results, topic)
        
        # Verify report structure
        assert '# Link Validation Report: Test Topic' in report
        assert '**Total Links Found:** 10' in report
        assert '**Valid Links:** 7 ✅' in report
        assert '**Broken Links:** 2 ❌' in report
        assert '**Skipped Links:** 1 ⏭️' in report
        assert '**Success Rate:** 77.8%' in report
        
        # Verify broken links section
        assert '## ❌ Broken Links' in report
        assert 'https://nonexistent.invalid' in report
        assert '**Status:** 404 - Not Found' in report
        assert 'doc1.html, doc2.html' in report
        
        # Verify valid links section
        assert '## ✅ Valid Links' in report
        assert 'https://www.google.com' in report
        
        # Verify performance metrics
        assert '## Performance Metrics' in report
        assert 'Average Response Time:' in report
        
        # Verify skipped links
        assert '## ⏭️ Skipped Links' in report
        assert 'Email links: 1' in report

    def test_generate_html_report(self, validator):
        """Test generate_html_report method."""
        analysis_results = {
            'summary': {'total': 5, 'valid': 3, 'broken': 1, 'skipped': 1, 'success_rate': 75.0},
            'broken_links': ['https://broken.example.com'],
            'valid_links': ['https://www.google.com'],
            'link_details': {
                'https://broken.example.com': {
                    'status_code': 404,
                    'message': 'Not Found',
                    'documents': ['/test/doc1.html'],
                    'response_time': 0.5
                },
                'https://www.google.com': {
                    'response_time': 0.1
                }
            }
        }
        topic = "Test Topic"
        
        html_report = validator.generate_html_report(analysis_results, topic)
        
        # Verify HTML structure
        assert '<!DOCTYPE html>' in html_report
        assert '<title>Link Validation Report: Test Topic</title>' in html_report
        assert '<h1>Link Validation Report: Test Topic</h1>' in html_report
        assert 'summary-grid' in html_report
        assert 'https://broken.example.com' in html_report
        assert '404 - Not Found' in html_report
        assert '</html>' in html_report

    def test_save_artifacts_markdown(self, validator, temp_dir):
        """Test save_artifacts with markdown report."""
        validator.report_format = 'markdown'
        results = {
            'summary': {'total': 3, 'valid': 2, 'broken': 1, 'skipped': 0, 'success_rate': 66.7},
            'broken_links': ['https://broken.example.com'],
            'link_details': {
                'https://broken.example.com': {
                    'status_code': 404,
                    'message': 'Not Found',
                    'documents': ['/test/doc1.html']
                }
            }
        }
        topic = "Test Topic"
        
        with patch.object(validator, 'generate_report', return_value='# Test Report'):
            saved_files = validator.save_artifacts(results, temp_dir, topic)
        
        # Verify markdown report was saved
        assert len(saved_files) == 2  # Main report + broken links
        report_file = temp_dir / 'test_topic_link_validation.md'
        assert report_file in saved_files
        assert report_file.exists()
        
        # Verify broken links file was saved
        broken_file = temp_dir / 'test_topic_broken_links.txt'
        assert broken_file in saved_files
        assert broken_file.exists()
        content = broken_file.read_text()
        assert 'https://broken.example.com' in content
        assert '404 - Not Found' in content

    def test_save_artifacts_html(self, validator, temp_dir):
        """Test save_artifacts with HTML report."""
        validator.report_format = 'html'
        results = {
            'summary': {'total': 2, 'valid': 2, 'broken': 0, 'skipped': 0, 'success_rate': 100.0},
            'broken_links': [],  # No broken links
            'link_details': {}
        }
        topic = "Test Topic"
        
        with patch.object(validator, 'generate_html_report', return_value='<html><body>Test</body></html>'):
            saved_files = validator.save_artifacts(results, temp_dir, topic)
        
        # Should only save main report (no broken links file)
        assert len(saved_files) == 1
        report_file = temp_dir / 'test_topic_link_validation.html'
        assert report_file in saved_files
        assert report_file.exists()

    def test_session_cleanup(self, mock_logger):
        """Test that session is properly cleaned up."""
        validator = LinkValidator(logger=mock_logger)
        
        # Mock the session close method
        validator.session.close = Mock()
        
        # Trigger cleanup
        validator.__del__()
        
        validator.session.close.assert_called_once()

    def test_session_cleanup_no_session(self, mock_logger):
        """Test cleanup when session doesn't exist."""
        validator = LinkValidator(logger=mock_logger)
        
        # Remove session attribute
        delattr(validator, 'session')
        
        # Should not raise exception
        try:
            validator.__del__()
            success = True
        except Exception:
            success = False
        
        assert success


class TestLinkValidatorIntegration:
    """Integration tests for LinkValidator with minimal mocking."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_full_link_extraction_workflow(self, temp_dir):
        """Test complete link extraction without network validation."""
        logger = Mock(spec=logging.Logger)
        
        # Create validator with network validation disabled
        config = {
            'timeout': 1,
            'retries': 0,
            'max_workers': 1,
            'ignore_patterns': [r'.*']  # Ignore all links to avoid network calls
        }
        validator = LinkValidator(logger=logger, config=config)
        
        # Sample document with various link types
        documents = [
            {
                'path': str(temp_dir / 'test.html'),
                'content': '''
                <html>
                <body>
                    <a href="https://www.example.com">Example</a>
                    <a href="mailto:test@example.com">Email</a>
                    <a href="#section">Anchor</a>
                    <img src="https://example.com/image.jpg" alt="Image">
                    <script src="https://example.com/script.js"></script>
                    <link href="https://example.com/style.css" rel="stylesheet">
                </body>
                </html>
                '''
            }
        ]
        
        # Run analysis
        result = validator.analyze(documents, "Integration Test")
        
        # Verify results
        assert result['total_links'] > 0
        assert result['unique_links'] > 0
        assert len(result['skipped_links']) > 0  # All should be skipped due to ignore pattern
        assert result['summary']['total'] > 0
        
        # Generate report
        report = validator.generate_report(result, "Integration Test")
        assert 'Link Validation Report' in report
        
        # Save artifacts
        saved_files = validator.save_artifacts(result, temp_dir, "Integration Test")
        assert len(saved_files) >= 1
        
        # Verify report file exists and contains expected content
        report_file = temp_dir / 'integration_test_link_validation.md'
        assert report_file.exists()
        report_content = report_file.read_text()
        assert 'Link Validation Report' in report_content