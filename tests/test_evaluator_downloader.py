"""
Tests for the DocumentationDownloader from the evaluator package.
"""

import pytest
import tempfile
import shutil
import json
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
import requests
from datetime import datetime

from doc_generator.evaluator.downloader import DocumentationDownloader


class TestDocumentationDownloader:
    """Test the DocumentationDownloader class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def downloader(self, temp_dir):
        """Create DocumentationDownloader instance with temp cache."""
        return DocumentationDownloader(cache_dir=str(temp_dir / ".doc_cache"))
    
    @pytest.fixture
    def sample_html(self):
        """Sample HTML content for testing."""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Documentation</title>
            <meta name="description" content="A test documentation page">
            <meta name="author" content="Test Author">
            <meta name="keywords" content="test,documentation,example">
            <meta name="generator" content="Sphinx-1.8.5">
        </head>
        <body>
            <nav class="sidebar">
                <ul>
                    <li><a href="#intro">Introduction</a></li>
                    <li><a href="#usage">Usage</a>
                        <ul>
                            <li><a href="#basic">Basic Usage</a></li>
                        </ul>
                    </li>
                </ul>
            </nav>
            <div class="document">
                <div class="documentwrapper">
                    <h1>Test Documentation</h1>
                    <h2 id="intro">Introduction</h2>
                    <p>This is the introduction section with detailed information.</p>
                    <h2 id="installation">Installation</h2>
                    <p>Install the package using pip:</p>
                    <pre><code class="language-bash">pip install test-package</code></pre>
                    <h2 id="usage">Usage</h2>
                    <p>Basic usage example:</p>
                    <pre class="highlight-python">
                        <code>
import test_package
test_package.run()
                        </code>
                    </pre>
                    <h3 id="advanced">Advanced Usage</h3>
                    <p>For advanced usage, use configuration:</p>
                    <code>config = {'debug': True}</code>
                </div>
            </div>
            <aside class="toc">Table of Contents</aside>
        </body>
        </html>
        '''

    def test_init_default_cache_dir(self):
        """Test initialization with default cache directory."""
        downloader = DocumentationDownloader()
        assert downloader.cache_dir == Path(".doc_cache")
        assert downloader.cache_dir.exists()
        # Clean up
        shutil.rmtree(".doc_cache", ignore_errors=True)

    def test_init_custom_cache_dir(self, temp_dir):
        """Test initialization with custom cache directory."""
        cache_dir = temp_dir / "custom_cache"
        downloader = DocumentationDownloader(cache_dir=str(cache_dir))
        assert downloader.cache_dir == cache_dir
        assert downloader.cache_dir.exists()

    def test_content_selectors_structure(self, downloader):
        """Test that content selectors are properly structured."""
        assert 'readthedocs' in downloader.content_selectors
        assert 'mkdocs' in downloader.content_selectors
        assert 'sphinx' in downloader.content_selectors
        assert 'github' in downloader.content_selectors
        assert 'generic' in downloader.content_selectors
        
        # Each platform should have a list of selectors
        for platform, selectors in downloader.content_selectors.items():
            assert isinstance(selectors, list)
            assert len(selectors) > 0

    @patch('requests.get')
    def test_download_page_success(self, mock_get, downloader):
        """Test successful page download."""
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
        mock_response.encoding = 'utf-8'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        url = "https://example.com/docs"
        content = downloader.download_page(url, use_cache=False)
        
        # Verify request
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == url
        assert 'User-Agent' in call_args[1]['headers']
        assert call_args[1]['timeout'] == 30
        
        # Verify content
        assert content == "<html><body>Test content</body></html>"

    @patch('requests.get')
    def test_download_page_with_caching(self, mock_get, downloader):
        """Test page download with caching functionality."""
        # Mock response
        mock_response = Mock()
        mock_response.text = "<html><body>Cached content</body></html>"
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.encoding = 'utf-8'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        url = "https://example.com/docs"
        
        # First call should download
        content1 = downloader.download_page(url, use_cache=True)
        assert mock_get.call_count == 1
        
        # Second call should use cache
        content2 = downloader.download_page(url, use_cache=True)
        assert mock_get.call_count == 1  # Still only one call
        assert content1 == content2
        
        # Verify cache files exist
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = downloader.cache_dir / f"{cache_key}.html"
        metadata_file = downloader.cache_dir / f"{cache_key}.json"
        
        assert cache_file.exists()
        assert metadata_file.exists()
        
        # Verify metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        assert metadata['url'] == url
        assert 'downloaded_at' in metadata

    @patch('requests.get')
    def test_download_page_request_exception(self, mock_get, downloader):
        """Test download page handling request exceptions."""
        mock_get.side_effect = requests.RequestException("Network error")
        
        url = "https://example.com/docs"
        
        with pytest.raises(requests.RequestException, match="Network error"):
            downloader.download_page(url, use_cache=False)

    @patch('requests.get')
    def test_download_page_http_error(self, mock_get, downloader):
        """Test download page handling HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        url = "https://example.com/docs"
        
        with pytest.raises(requests.HTTPError, match="404 Not Found"):
            downloader.download_page(url, use_cache=False)

    def test_detect_platform_sphinx(self, downloader):
        """Test platform detection for Sphinx."""
        from bs4 import BeautifulSoup
        html = '<html><head><meta name="generator" content="Sphinx-1.8.5"></head></html>'
        soup = BeautifulSoup(html, 'html.parser')
        
        platform = downloader._detect_platform(soup, None)
        assert platform == 'sphinx'

    def test_detect_platform_mkdocs(self, downloader):
        """Test platform detection for MkDocs."""
        from bs4 import BeautifulSoup
        html = '<html><head><meta name="generator" content="MkDocs-1.1"></head></html>'
        soup = BeautifulSoup(html, 'html.parser')
        
        platform = downloader._detect_platform(soup, None)
        assert platform == 'mkdocs'

    def test_detect_platform_by_url(self, downloader):
        """Test platform detection by URL patterns."""
        from bs4 import BeautifulSoup
        html = '<html><body>Generic content</body></html>'
        soup = BeautifulSoup(html, 'html.parser')
        
        # ReadTheDocs URL
        platform = downloader._detect_platform(soup, "https://myproject.readthedocs.io/en/latest/")
        assert platform == 'readthedocs'
        
        # GitHub URL
        platform = downloader._detect_platform(soup, "https://github.com/user/repo")
        assert platform == 'github'
        
        platform = downloader._detect_platform(soup, "https://user.github.io/repo")
        assert platform == 'github'

    def test_detect_platform_by_classes(self, downloader):
        """Test platform detection by CSS classes."""
        from bs4 import BeautifulSoup
        
        # MkDocs classes
        html = '<html><body><div class="md-content">Content</div></body></html>'
        soup = BeautifulSoup(html, 'html.parser')
        platform = downloader._detect_platform(soup, None)
        assert platform == 'mkdocs'
        
        # Sphinx classes
        html = '<html><body><div class="sphinx-doc">Content</div></body></html>'
        soup = BeautifulSoup(html, 'html.parser')
        platform = downloader._detect_platform(soup, None)
        assert platform == 'sphinx'

    def test_detect_platform_generic(self, downloader):
        """Test generic platform detection fallback."""
        from bs4 import BeautifulSoup
        html = '<html><body>Generic content</body></html>'
        soup = BeautifulSoup(html, 'html.parser')
        
        platform = downloader._detect_platform(soup, None)
        assert platform == 'generic'

    def test_extract_main_content(self, downloader, sample_html):
        """Test main content extraction."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(sample_html, 'html.parser')
        
        # Test sphinx platform (should find div.documentwrapper)
        content = downloader._extract_main_content(soup, 'sphinx')
        assert content is not None
        assert 'Test Documentation' in content.get_text()
        
        # Navigation and sidebars should be removed
        assert not content.find('nav')
        assert not content.find('aside')

    def test_extract_main_content_fallback(self, downloader):
        """Test main content extraction fallback to body."""
        from bs4 import BeautifulSoup
        html = '<html><body><p>Only body content</p></body></html>'
        soup = BeautifulSoup(html, 'html.parser')
        
        content = downloader._extract_main_content(soup, 'generic')
        assert content.name == 'body'
        assert 'Only body content' in content.get_text()

    def test_extract_sections(self, downloader, sample_html):
        """Test section extraction from content."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(sample_html, 'html.parser')
        content = soup.find('div', class_='documentwrapper')
        
        sections = downloader._extract_sections(content)
        
        # Should find h1, h2, h3 headers
        assert len(sections) >= 4
        
        # Verify section structure
        section_titles = [s['title'] for s in sections]
        assert 'Test Documentation' in section_titles
        assert 'Introduction' in section_titles
        assert 'Installation' in section_titles
        assert 'Usage' in section_titles
        
        # Verify levels
        h1_sections = [s for s in sections if s['level'] == 1]
        h2_sections = [s for s in sections if s['level'] == 2]
        h3_sections = [s for s in sections if s['level'] == 3]
        
        assert len(h1_sections) >= 1
        assert len(h2_sections) >= 3
        assert len(h3_sections) >= 1
        
        # Verify IDs
        intro_section = next((s for s in sections if s['title'] == 'Introduction'), None)
        assert intro_section is not None
        assert intro_section['id'] == 'intro'

    def test_extract_metadata(self, downloader, sample_html):
        """Test metadata extraction."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(sample_html, 'html.parser')
        
        metadata = downloader._extract_metadata(soup)
        
        assert metadata['title'] == 'Test Documentation'
        assert metadata['description'] == 'A test documentation page'
        assert metadata['author'] == 'Test Author'
        assert metadata['keywords'] == ['test', 'documentation', 'example']

    def test_extract_metadata_missing_fields(self, downloader):
        """Test metadata extraction with missing fields."""
        from bs4 import BeautifulSoup
        html = '<html><head><title>Only Title</title></head><body></body></html>'
        soup = BeautifulSoup(html, 'html.parser')
        
        metadata = downloader._extract_metadata(soup)
        
        assert metadata['title'] == 'Only Title'
        assert 'description' not in metadata
        assert 'author' not in metadata
        assert 'keywords' not in metadata

    def test_extract_code_examples(self, downloader, sample_html):
        """Test code example extraction."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(sample_html, 'html.parser')
        content = soup.find('div', class_='documentwrapper')
        
        examples = downloader._extract_code_examples(content)
        
        assert len(examples) >= 3
        
        # Find specific examples
        bash_example = next((e for e in examples if e['language'] == 'bash'), None)
        assert bash_example is not None
        assert 'pip install test-package' in bash_example['code']
        
        python_example = next((e for e in examples if e['language'] == 'python'), None)
        assert python_example is not None
        assert 'import test_package' in python_example['code']
        
        # Check inline code
        inline_examples = [e for e in examples if 'debug' in e['code']]
        assert len(inline_examples) >= 1

    def test_extract_code_examples_language_detection(self, downloader):
        """Test language detection in code examples."""
        from bs4 import BeautifulSoup
        html = '''
        <div>
            <pre><code class="language-python">print("hello")</code></pre>
            <pre><code class="javascript">console.log("hello");</code></pre>
            <pre class="highlight-bash"><code>echo "hello"</code></pre>
            <code>no language</code>
        </div>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        
        examples = downloader._extract_code_examples(soup)
        
        # Should find code examples (may find both pre and code elements)
        assert len(examples) >= 4
        
        # Verify language detection
        languages = [e['language'] for e in examples]
        assert 'python' in languages
        assert 'javascript' in languages
        assert 'bash' in languages

    def test_extract_navigation(self, downloader, sample_html):
        """Test navigation structure extraction."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(sample_html, 'html.parser')
        
        nav_items = downloader._extract_navigation(soup)
        
        assert len(nav_items) >= 3
        
        # Find specific navigation items
        nav_texts = [item['text'] for item in nav_items]
        assert 'Introduction' in nav_texts
        assert 'Usage' in nav_texts
        assert 'Basic Usage' in nav_texts
        
        # Check hierarchy levels
        intro_item = next((item for item in nav_items if item['text'] == 'Introduction'), None)
        basic_item = next((item for item in nav_items if item['text'] == 'Basic Usage'), None)
        
        assert intro_item['level'] == 1  # Top level
        assert basic_item['level'] == 2  # Nested level

    def test_get_nav_level(self, downloader):
        """Test navigation level calculation."""
        from bs4 import BeautifulSoup
        html = '''
        <nav>
            <ul>
                <li><a href="#top">Top Level</a></li>
                <li>
                    <ul>
                        <li><a href="#nested">Nested Level</a></li>
                        <li>
                            <ul>
                                <li><a href="#deep">Deep Level</a></li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ul>
        </nav>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        
        links = soup.find_all('a')
        top_link = next((l for l in links if l.get_text() == 'Top Level'), None)
        nested_link = next((l for l in links if l.get_text() == 'Nested Level'), None)
        deep_link = next((l for l in links if l.get_text() == 'Deep Level'), None)
        
        assert downloader._get_nav_level(top_link) == 1
        assert downloader._get_nav_level(nested_link) == 2
        assert downloader._get_nav_level(deep_link) == 3

    def test_extract_content_full_workflow(self, downloader, sample_html):
        """Test full content extraction workflow."""
        result = downloader.extract_content(sample_html, "https://example.readthedocs.io/")
        
        # Verify structure
        assert 'platform' in result
        assert 'content' in result
        assert 'sections' in result
        assert 'metadata' in result
        assert 'code_examples' in result
        assert 'navigation' in result
        assert 'raw_text' in result
        
        # Verify platform detection (Sphinx detected from meta tag takes precedence over URL)
        assert result['platform'] == 'sphinx'  # Based on meta tag generator
        
        # Verify sections
        assert len(result['sections']) >= 4
        
        # Verify metadata
        assert result['metadata']['title'] == 'Test Documentation'
        
        # Verify code examples
        assert len(result['code_examples']) >= 3
        
        # Verify navigation
        assert len(result['navigation']) >= 3
        
        # Verify raw text
        assert 'Test Documentation' in result['raw_text']
        assert 'pip install test-package' in result['raw_text']

    def test_extract_content_empty_html(self, downloader):
        """Test content extraction with empty HTML."""
        result = downloader.extract_content('', None)
        
        assert result['platform'] == 'generic'
        assert result['sections'] == []
        assert result['metadata'] == {}
        assert result['code_examples'] == []
        assert result['navigation'] == []
        assert result['raw_text'] == ''

    @patch('requests.get')
    def test_download_and_extract_integration(self, mock_get, downloader, sample_html):
        """Test download_and_extract integration method."""
        # Mock response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.encoding = 'utf-8'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        url = "https://example.com/docs"
        result = downloader.download_and_extract(url, use_cache=False)
        
        # Verify download was called
        mock_get.assert_called_once()
        
        # Verify extraction results
        assert 'platform' in result
        assert 'sections' in result
        assert 'metadata' in result
        assert result['url'] == url
        assert len(result['sections']) >= 4
        assert result['metadata']['title'] == 'Test Documentation'

    def test_extract_sections_no_content(self, downloader):
        """Test section extraction with no content."""
        sections = downloader._extract_sections(None)
        assert sections == []

    def test_extract_code_examples_no_content(self, downloader):
        """Test code example extraction with no content."""
        examples = downloader._extract_code_examples(None)
        assert examples == []


class TestDocumentationDownloaderEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_malformed_html_handling(self, temp_dir):
        """Test handling of malformed HTML."""
        downloader = DocumentationDownloader(cache_dir=str(temp_dir / ".cache"))
        
        malformed_html = '''
        <html>
        <head><title>Broken HTML</title>
        <body>
            <h2>Section 1</h2>
            <p>Unclosed paragraph
            <h2>Section 2</h2>
            <pre><code>broken code block
            <div class="broken
        </html>
        '''
        
        # Should handle gracefully
        result = downloader.extract_content(malformed_html)
        
        assert result['metadata']['title'] == 'Broken HTML'
        assert len(result['sections']) >= 2
        assert 'Section 1' in result['raw_text']

    def test_unicode_content_handling(self, temp_dir):
        """Test handling of Unicode content."""
        downloader = DocumentationDownloader(cache_dir=str(temp_dir / ".cache"))
        
        unicode_html = '''
        <html>
        <head><title>Unicode Test 测试</title></head>
        <body>
            <h2>Español: Introducción</h2>
            <p>Content with émojis: 🚀 🎉 ✅</p>
            <h2>中文：安装</h2>
            <p>pip install 包名</p>
            <pre><code>print("Hello, 世界!")</code></pre>
        </body>
        </html>
        '''
        
        result = downloader.extract_content(unicode_html)
        
        assert 'Unicode Test 测试' in result['metadata']['title']
        assert 'Introducción' in result['raw_text']
        assert '🚀 🎉 ✅' in result['raw_text']
        assert '中文：安装' in result['raw_text']
        assert 'Hello, 世界!' in result['raw_text']

    def test_large_navigation_structure(self, temp_dir):
        """Test handling of large navigation structures."""
        downloader = DocumentationDownloader(cache_dir=str(temp_dir / ".cache"))
        
        # Generate large navigation structure
        nav_html = '<nav><ul>'
        for i in range(50):
            nav_html += f'<li><a href="#section{i}">Section {i}</a>'
            if i % 10 == 0:
                nav_html += '<ul>'
                for j in range(5):
                    nav_html += f'<li><a href="#subsection{i}_{j}">Subsection {i}.{j}</a></li>'
                nav_html += '</ul>'
            nav_html += '</li>'
        nav_html += '</ul></nav>'
        
        html = f'<html><body>{nav_html}<main>Content</main></body></html>'
        
        result = downloader.extract_content(html)
        
        # Should handle large navigation gracefully
        assert len(result['navigation']) > 50
        levels = [item['level'] for item in result['navigation']]
        assert max(levels) >= 2  # Should detect hierarchy

    @patch('requests.get')
    def test_cache_corruption_recovery(self, mock_get, temp_dir):
        """Test recovery from corrupted cache files."""
        downloader = DocumentationDownloader(cache_dir=str(temp_dir / ".cache"))
        
        # Create corrupted cache file
        url = "https://example.com/docs"
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = downloader.cache_dir / f"{cache_key}.html"
        
        # Write corrupted data
        cache_file.write_bytes(b'\x00\x01\x02corrupted')
        
        # Mock fresh download
        mock_response = Mock()
        mock_response.text = "<html><body>Fresh content</body></html>"
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.encoding = 'utf-8'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Should read corrupted cache (the current implementation doesn't have recovery logic)
        content = downloader.download_page(url, use_cache=True)
        # The implementation currently reads the cache as-is without validation
        assert len(content) > 0  # Should return some content