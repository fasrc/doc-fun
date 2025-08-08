"""Download and process existing documentation pages for comparison."""

import re
import logging
from typing import Dict, Optional, List, Tuple
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
import hashlib
from datetime import datetime


class DocumentationDownloader:
    """Download and extract content from existing documentation pages."""
    
    def __init__(self, cache_dir: str = ".doc_cache"):
        """Initialize the downloader with optional caching.
        
        Args:
            cache_dir: Directory to cache downloaded pages
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Common documentation selectors for different platforms
        self.content_selectors = {
            'readthedocs': ['div.document', 'div.body', 'main'],
            'mkdocs': ['article', 'div.content', 'main'],
            'sphinx': ['div.documentwrapper', 'div.bodywrapper'],
            'github': ['article', 'div.markdown-body'],
            'generic': ['main', 'article', 'div.content', 'div.documentation']
        }
        
    def download_page(self, url: str, use_cache: bool = True) -> str:
        """Download a documentation page.
        
        Args:
            url: URL of the documentation page
            use_cache: Whether to use cached version if available
            
        Returns:
            HTML content of the page
        """
        # Generate cache key
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.html"
        metadata_file = self.cache_dir / f"{cache_key}.json"
        
        # Check cache
        if use_cache and cache_file.exists():
            self.logger.info(f"Using cached version of {url}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        # Download page
        self.logger.info(f"Downloading {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; DocGenerator/1.0; +https://github.com/fasrc/doc-fun)'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content = response.text
            
            # Cache the content
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Save metadata
            metadata = {
                'url': url,
                'downloaded_at': datetime.now().isoformat(),
                'content_type': response.headers.get('content-type', ''),
                'encoding': response.encoding
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            return content
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download {url}: {e}")
            raise
    
    def extract_content(self, html: str, url: str = None) -> Dict[str, any]:
        """Extract structured content from HTML documentation.
        
        Args:
            html: HTML content
            url: Optional URL for platform detection
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Detect platform
        platform = self._detect_platform(soup, url)
        
        # Extract main content
        content = self._extract_main_content(soup, platform)
        
        # Extract sections
        sections = self._extract_sections(content)
        
        # Extract metadata
        metadata = self._extract_metadata(soup)
        
        # Extract code examples
        code_examples = self._extract_code_examples(content)
        
        # Extract navigation structure
        navigation = self._extract_navigation(soup)
        
        return {
            'platform': platform,
            'content': content,
            'sections': sections,
            'metadata': metadata,
            'code_examples': code_examples,
            'navigation': navigation,
            'raw_text': content.get_text(separator='\n', strip=True) if content else ''
        }
    
    def _detect_platform(self, soup: BeautifulSoup, url: Optional[str]) -> str:
        """Detect the documentation platform."""
        # Check meta tags
        generator = soup.find('meta', attrs={'name': 'generator'})
        if generator:
            content = generator.get('content', '').lower()
            if 'sphinx' in content:
                return 'sphinx'
            elif 'mkdocs' in content:
                return 'mkdocs'
        
        # Check URL patterns
        if url:
            if 'readthedocs' in url:
                return 'readthedocs'
            elif 'github.com' in url or 'github.io' in url:
                return 'github'
        
        # Check class patterns
        if soup.find(class_=re.compile('mkdocs|md-')):
            return 'mkdocs'
        elif soup.find(class_=re.compile('sphinx|rst')):
            return 'sphinx'
        
        return 'generic'
    
    def _extract_main_content(self, soup: BeautifulSoup, platform: str) -> BeautifulSoup:
        """Extract the main content area."""
        selectors = self.content_selectors.get(platform, self.content_selectors['generic'])
        
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                # Remove navigation, sidebars, etc.
                for elem in content.select('nav, aside, .sidebar, .toc, .navigation'):
                    elem.decompose()
                return content
        
        # Fallback to body
        return soup.body if soup.body else soup
    
    def _extract_sections(self, content: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract sections with headers and content."""
        sections = []
        headers = content.find_all(['h1', 'h2', 'h3']) if content else []
        
        for i, header in enumerate(headers):
            section = {
                'level': int(header.name[1]),
                'title': header.get_text(strip=True),
                'id': header.get('id', ''),
                'content': ''
            }
            
            # Extract content between headers
            current = header.next_sibling
            content_parts = []
            
            while current and current not in headers[i+1:]:
                if hasattr(current, 'get_text'):
                    content_parts.append(current.get_text(strip=True))
                current = current.next_sibling if hasattr(current, 'next_sibling') else None
            
            section['content'] = '\n'.join(content_parts)
            sections.append(section)
        
        return sections
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract page metadata."""
        metadata = {}
        
        # Title
        title = soup.find('title')
        if title:
            metadata['title'] = title.get_text(strip=True)
        
        # Meta description
        description = soup.find('meta', attrs={'name': 'description'})
        if description:
            metadata['description'] = description.get('content', '')
        
        # Author
        author = soup.find('meta', attrs={'name': 'author'})
        if author:
            metadata['author'] = author.get('content', '')
        
        # Keywords
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords:
            metadata['keywords'] = keywords.get('content', '').split(',')
        
        return metadata
    
    def _extract_code_examples(self, content: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract code examples from the content."""
        examples = []
        
        if not content:
            return examples
        
        # Find code blocks
        for code_block in content.find_all(['pre', 'code']):
            example = {
                'code': code_block.get_text(strip=True),
                'language': ''
            }
            
            # Try to detect language
            classes = code_block.get('class', [])
            for cls in classes:
                if 'language-' in cls:
                    example['language'] = cls.replace('language-', '')
                    break
                elif cls in ['python', 'javascript', 'bash', 'shell', 'java', 'cpp']:
                    example['language'] = cls
                    break
            
            # Check parent for language hints
            if not example['language'] and code_block.parent:
                parent_classes = code_block.parent.get('class', [])
                for cls in parent_classes:
                    if 'highlight-' in cls:
                        example['language'] = cls.replace('highlight-', '')
                        break
            
            if example['code']:  # Only add non-empty examples
                examples.append(example)
        
        return examples
    
    def _extract_navigation(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract navigation structure."""
        nav_items = []
        
        # Look for common navigation patterns
        nav = soup.find('nav') or soup.find(class_=re.compile('nav|toc|sidebar'))
        
        if nav:
            for link in nav.find_all('a'):
                nav_items.append({
                    'text': link.get_text(strip=True),
                    'href': link.get('href', ''),
                    'level': self._get_nav_level(link)
                })
        
        return nav_items
    
    def _get_nav_level(self, element) -> int:
        """Determine navigation hierarchy level."""
        level = 0
        parent = element.parent
        
        while parent:
            if parent.name in ['ul', 'ol']:
                level += 1
            parent = parent.parent
        
        return level
    
    def download_and_extract(self, url: str, use_cache: bool = True) -> Dict[str, any]:
        """Download and extract content in one step.
        
        Args:
            url: URL of the documentation page
            use_cache: Whether to use cached version
            
        Returns:
            Extracted content dictionary
        """
        html = self.download_page(url, use_cache)
        content = self.extract_content(html, url)
        content['url'] = url
        return content