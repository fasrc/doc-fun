"""Tests for content extractors."""

import pytest
from src.doc_generator.extractors import HTMLContentExtractor, ExtractedContent


class TestHTMLContentExtractor:
    """Test HTMLContentExtractor functionality."""
    
    def test_can_extract_html_file(self):
        """Test HTML file detection by extension."""
        extractor = HTMLContentExtractor()
        
        # Test file extension detection
        assert extractor.can_extract("", "test.html") is True
        assert extractor.can_extract("", "test.htm") is True
        assert extractor.can_extract("", "test.txt") is False
    
    def test_can_extract_html_content(self):
        """Test HTML content detection by tags."""
        extractor = HTMLContentExtractor()
        
        # Test HTML tag detection
        html_content = "<html><body><h1>Title</h1></body></html>"
        assert extractor.can_extract(html_content) is True
        
        plain_text = "This is just plain text without HTML tags"
        assert extractor.can_extract(plain_text) is False
    
    def test_extract_basic_html(self):
        """Test basic HTML extraction."""
        extractor = HTMLContentExtractor()
        
        html_content = """
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Introduction</h1>
            <p>This is the introduction section.</p>
            <h2>Usage</h2>
            <p>This explains how to use the feature.</p>
        </body>
        </html>
        """
        
        extracted = extractor.extract(html_content)
        
        assert isinstance(extracted, ExtractedContent)
        assert extracted.title == "Test Document"
        assert extracted.format_type == "html"
        assert "introduction" in extracted.sections
        assert "usage" in extracted.sections
    
    def test_extract_complex_html(self):
        """Test extraction from complex HTML structure."""
        extractor = HTMLContentExtractor()
        
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="description" content="Test document">
            <title>Complex Document</title>
        </head>
        <body>
            <nav>Navigation menu</nav>
            <main>
                <h1>Main Title</h1>
                <section>
                    <h2>Installation</h2>
                    <p>Installation instructions here.</p>
                    <pre><code>npm install package</code></pre>
                </section>
                <section>
                    <h2>Configuration</h2>
                    <ul>
                        <li>Option 1</li>
                        <li>Option 2</li>
                    </ul>
                </section>
            </main>
            <footer>Footer content</footer>
        </body>
        </html>
        """
        
        extracted = extractor.extract(html_content)
        
        assert extracted.title == "Complex Document"
        assert "installation" in extracted.sections
        assert "configuration" in extracted.sections
        assert "npm install package" in extracted.sections["installation"]
        assert "Option 1" in extracted.sections["configuration"]
        
        # Check metadata extraction
        assert "description" in extracted.metadata
        assert extracted.metadata["description"] == "Test document"
        assert extracted.metadata["language"] == "en"
    
    def test_get_format_type(self):
        """Test format type identifier."""
        extractor = HTMLContentExtractor()
        assert extractor.get_format_type() == "html"
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        extractor = HTMLContentExtractor()
        extensions = extractor.get_supported_extensions()
        assert ".html" in extensions
        assert ".htm" in extensions
    
    def test_preprocess_content(self):
        """Test content preprocessing."""
        extractor = HTMLContentExtractor()
        
        html_with_scripts = """
        <html>
        <head>
            <script>alert('test');</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <!-- This is a comment -->
            <h1>Title</h1>
            <p>Content</p>
        </body>
        </html>
        """
        
        processed = extractor.preprocess_content(html_with_scripts)
        
        # Scripts and styles should be removed
        assert "alert('test')" not in processed
        assert "color: red" not in processed
        assert "This is a comment" not in processed
        
        # Content should remain
        assert "<h1>Title</h1>" in processed
        assert "<p>Content</p>" in processed
    
    def test_extract_no_title(self):
        """Test extraction when no title is present."""
        extractor = HTMLContentExtractor()
        
        html_content = """
        <html>
        <body>
            <p>Content without title</p>
        </body>
        </html>
        """
        
        extracted = extractor.extract(html_content)
        assert extracted.title is None
        assert "content" in extracted.sections
    
    def test_extract_empty_html(self):
        """Test extraction from minimal HTML."""
        extractor = HTMLContentExtractor()
        
        html_content = "<html><body></body></html>"
        
        extracted = extractor.extract(html_content)
        assert extracted.format_type == "html"
        assert len(extracted.sections) == 0 or extracted.sections.get("content", "") == ""