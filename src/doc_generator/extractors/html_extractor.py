"""HTML content extractor for document standardization."""

from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup, Tag, NavigableString
import re

from .base import ContentExtractor, ExtractedContent, ContentSection


class HTMLContentExtractor(ContentExtractor):
    """Extracts structured content from HTML documents."""
    
    def __init__(self):
        self.heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        self.block_tags = ['p', 'div', 'section', 'article', 'pre', 'code', 
                          'ul', 'ol', 'li', 'blockquote', 'table']
    
    def can_extract(self, content: str, file_path: Optional[str] = None) -> bool:
        """Check if content appears to be HTML."""
        # Check file extension
        if file_path and file_path.lower().endswith(('.html', '.htm')):
            return True
        
        # Check for HTML tags
        html_pattern = re.compile(r'<[^>]+>', re.IGNORECASE)
        return bool(html_pattern.search(content.strip()[:1000]))
    
    def get_format_type(self) -> str:
        """Return format type identifier."""
        return 'html'
    
    def get_supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return ['.html', '.htm']
    
    def extract(self, content: str, file_path: Optional[str] = None) -> ExtractedContent:
        """Extract structured content from HTML."""
        # Preprocess content
        content = self.preprocess_content(content)
        
        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract title
        title = self._extract_title(soup)
        
        # Extract sections
        sections = self._extract_sections(soup)
        
        # Extract metadata
        metadata = self._extract_metadata(soup)
        
        # Create extracted content
        extracted = ExtractedContent(
            title=title,
            sections=sections,
            metadata=metadata,
            raw_content=content,
            format_type=self.get_format_type()
        )
        
        return self.postprocess_content(extracted)
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract document title from HTML."""
        # Try <title> tag first
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(strip=True)
        
        # Try first h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        
        # Try page heading patterns
        for selector in ['#title', '.title', '[data-title]']:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return None
    
    def _extract_sections(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract content sections from HTML."""
        sections = {}
        current_section = "content"
        current_content = []
        
        # Find main content area
        main_content = self._find_main_content(soup)
        if not main_content:
            main_content = soup
        
        # Process elements in order
        for element in main_content.find_all(recursive=True):
            if element.name in self.heading_tags:
                # Save previous section
                if current_content:
                    sections[current_section] = self._clean_text('\n'.join(current_content))
                
                # Start new section
                current_section = element.get_text(strip=True).lower().replace(' ', '_')
                if not current_section:
                    current_section = f"section_{len(sections) + 1}"
                current_content = []
                
            elif element.name in self.block_tags:
                # Add block content
                text = self._extract_element_text(element)
                if text.strip():
                    current_content.append(text)
        
        # Save final section
        if current_content:
            sections[current_section] = self._clean_text('\n'.join(current_content))
        
        # Ensure we have at least some content
        if not sections:
            all_text = main_content.get_text(strip=True)
            if all_text:
                sections["content"] = self._clean_text(all_text)
        
        return sections
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the main content area of the HTML document."""
        # Try semantic HTML5 elements
        for tag in ['main', 'article']:
            element = soup.find(tag)
            if element:
                return element
        
        # Try common content selectors
        for selector in ['#content', '#main', '.content', '.main-content', 
                        '[role="main"]']:
            element = soup.select_one(selector)
            if element:
                return element
        
        # Try to find content by excluding common non-content areas
        body = soup.find('body')
        if body:
            # Remove navigation, header, footer, sidebar elements
            for selector in ['nav', 'header', 'footer', 'aside', '.nav', 
                           '.navbar', '.header', '.footer', '.sidebar']:
                for element in body.select(selector):
                    element.decompose()
            return body
        
        return None
    
    def _extract_element_text(self, element: Tag) -> str:
        """Extract clean text from an HTML element."""
        if element.name == 'pre':
            # Preserve whitespace in preformatted text
            return element.get_text()
        elif element.name in ['code', 'tt']:
            # Mark code elements
            return f"`{element.get_text()}`"
        elif element.name in ['ul', 'ol']:
            # Handle lists
            items = []
            for li in element.find_all('li', recursive=False):
                items.append(f"- {li.get_text(strip=True)}")
            return '\n'.join(items)
        else:
            # Regular text extraction
            return element.get_text(strip=True)
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML document."""
        metadata = {}
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # Extract language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')
        
        # Extract encoding
        charset_meta = soup.find('meta', attrs={'charset': True})
        if charset_meta:
            metadata['encoding'] = charset_meta.get('charset')
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\t+', ' ', text)  # Tabs to spaces
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess_content(self, content: str) -> str:
        """Preprocess HTML content."""
        # Remove comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Remove script and style tags
        soup = BeautifulSoup(content, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        
        return str(soup)
    
    def postprocess_content(self, extracted: ExtractedContent) -> ExtractedContent:
        """Postprocess extracted HTML content."""
        # Clean up section names
        cleaned_sections = {}
        for key, value in extracted.sections.items():
            # Normalize section names
            clean_key = re.sub(r'[^\w\s-]', '', key).strip()
            clean_key = re.sub(r'\s+', '_', clean_key).lower()
            if not clean_key:
                clean_key = f"section_{len(cleaned_sections) + 1}"
            
            cleaned_sections[clean_key] = value
        
        extracted.sections = cleaned_sections
        return extracted