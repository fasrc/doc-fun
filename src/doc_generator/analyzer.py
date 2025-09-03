"""
Document Analysis Module

Provides document structure analysis and scoring functionality.
Extracted from core.py following single responsibility principle.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .config import get_settings
from .cache import cached
from .exceptions import DocGeneratorError


@dataclass
class SectionInfo:
    """Information about a document section."""
    name: str
    content: str
    start_line: int
    end_line: int
    word_count: int
    score: float


class DocumentAnalyzer:
    """
    Analyzes document structure and calculates quality scores.
    
    Provides algorithmic scoring based on section completeness,
    content quality metrics, and structural analysis.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, section_headers: Optional[List[str]] = None):
        """
        Initialize document analyzer.
        
        Args:
            logger: Optional logger instance
            section_headers: Optional list of section headers to analyze
        """
        self.settings = get_settings()
        self.logger = logger or logging.getLogger(__name__)
        
        # Use provided section headers or default ones (uppercase for backward compatibility)
        self.section_headers = section_headers or [
            'Description', 'Installation', 'Usage', 'Examples', 'References'
        ]
    
    @cached(ttl=1800)  # Cache for 30 minutes
    def extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract sections from document content (supports both HTML and Markdown).
        
        Args:
            content: Document content to analyze
            
        Returns:
            Dictionary mapping section names to content (for backward compatibility)
            
        Raises:
            DocGeneratorError: If content analysis fails
        """
        section_objects = self.extract_section_objects(content)
        
        # Convert to dictionary for backward compatibility
        return {section.name: section.content for section in section_objects}
    
    def extract_section_objects(self, content: str) -> List['SectionInfo']:
        """
        Extract sections from document content as SectionInfo objects.
        Supports both HTML and Markdown formats.
        
        Args:
            content: Document content to analyze
            
        Returns:
            List of SectionInfo objects
            
        Raises:
            DocGeneratorError: If content analysis fails
        """
        if not content or not content.strip():
            raise DocGeneratorError(
                "Cannot analyze empty content",
                error_code="EMPTY_CONTENT"
            )
        
        # Detect if content is HTML or Markdown
        content_lower = content.lower()
        is_html = '<html>' in content_lower or '<h1>' in content_lower or '<h2>' in content_lower
        
        if is_html:
            return self._extract_sections_from_html(content)
        else:
            return self._extract_sections_from_markdown(content)
    
    def _extract_sections_from_html(self, html_content: str) -> List['SectionInfo']:
        """Extract sections from HTML content."""
        if BeautifulSoup is None:
            raise DocGeneratorError(
                "BeautifulSoup is required for HTML parsing. Install with: pip install beautifulsoup4",
                error_code="MISSING_DEPENDENCY"
            )
        
        sections = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all headers (h1, h2, h3, h4)
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        
        for i, header in enumerate(headers):
            header_text = header.get_text().strip()
            
            # Check if this header matches any of our target sections
            for section_name in self.section_headers:
                if section_name.lower() in header_text.lower():
                    # Extract content between this header and the next header
                    content_parts = []
                    
                    # Get all siblings until the next header
                    for sibling in header.find_next_siblings():
                        if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                            break
                        content_parts.append(str(sibling))
                    
                    section_content = '\n'.join(content_parts)
                    word_count = len(BeautifulSoup(section_content, 'html.parser').get_text().split())
                    
                    sections.append(SectionInfo(
                        name=section_name,
                        content=section_content,
                        start_line=0,  # HTML doesn't have line numbers easily
                        end_line=0,
                        word_count=word_count,
                        score=self.calculate_section_score(section_content, section_name)
                    ))
                    break
        
        self.logger.info(f"Extracted {len(sections)} sections from HTML document")
        return sections
    
    def _extract_sections_from_markdown(self, content: str) -> List['SectionInfo']:
        """Extract sections from Markdown content."""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        start_line = 0
        
        try:
            for i, line in enumerate(lines):
                # Check for section headers (markdown style)
                header_match = re.match(r'^#+\s*(.+)$', line.strip())
                if header_match:
                    # Save previous section if exists
                    if current_section:
                        section_content = '\n'.join(current_content)
                        word_count = len(section_content.split())
                        
                        sections.append(SectionInfo(
                            name=current_section,
                            content=section_content,
                            start_line=start_line,
                            end_line=i - 1,
                            word_count=word_count,
                            score=self.calculate_section_score(section_content, current_section)
                        ))
                    
                    # Start new section
                    current_section = header_match.group(1).strip().lower()
                    current_content = []
                    start_line = i
                else:
                    # Add to current section content
                    if current_section:
                        current_content.append(line)
            
            # Handle final section
            if current_section and current_content:
                section_content = '\n'.join(current_content)
                word_count = len(section_content.split())
                
                sections.append(SectionInfo(
                    name=current_section,
                    content=section_content,
                    start_line=start_line,
                    end_line=len(lines) - 1,
                    word_count=word_count,
                    score=self.calculate_section_score(section_content, current_section)
                ))
                
        except Exception as e:
            raise DocGeneratorError(
                f"Failed to extract sections from Markdown: {e}",
                error_code="SECTION_EXTRACTION_ERROR",
                context={'content_length': len(content)}
            )
        
        self.logger.info(f"Extracted {len(sections)} sections from Markdown document")
        return sections
    
    def calculate_section_score(self, content: str, section_name: str) -> float:
        """
        Calculate quality score for a document section.
        
        Args:
            content: Section content
            section_name: Name of the section
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not content or not content.strip():
            return 0.0
        
        score = 0.0
        content_lower = content.lower()
        
        # Word count scoring (optimal range: 50-500 words)
        word_count = len(content.split())
        if word_count < 10:
            score += 0.1  # Too short
        elif word_count < 50:
            score += 0.3  # Short but acceptable
        elif word_count <= 500:
            score += 0.5  # Good length
        else:
            score += 0.4  # Long but manageable
        
        # Code examples boost score for technical sections
        code_patterns = [r'```', r'`[^`]+`', r'^\s{4,}', r'def ', r'class ', r'import ']
        code_matches = sum(len(re.findall(pattern, content, re.MULTILINE)) 
                          for pattern in code_patterns)
        if code_matches > 0:
            score += min(0.2, code_matches * 0.05)
        
        # Links and references add value
        link_patterns = [r'http[s]?://', r'\[.+\]\(.+\)', r'@\w+']
        link_matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                          for pattern in link_patterns)
        if link_matches > 0:
            score += min(0.1, link_matches * 0.02)
        
        # Section-specific scoring
        if section_name in ['examples', 'usage']:
            # Examples should have code
            if code_matches > 0:
                score += 0.1
            # Examples should have explanatory text
            if word_count > 30:
                score += 0.1
        
        elif section_name in ['installation', 'setup']:
            # Installation should have step-by-step instructions
            if re.search(r'\b(install|pip|npm|git clone)\b', content_lower):
                score += 0.1
            if re.search(r'\d+\.|step|first|then|next', content_lower):
                score += 0.1
        
        elif section_name in ['api', 'reference']:
            # API docs should be well-structured
            if re.search(r'(parameters?|returns?|raises?)', content_lower):
                score += 0.1
            if code_matches > 2:  # Multiple code examples
                score += 0.1
        
        # Formatting quality indicators
        if re.search(r'\*\*[^*]+\*\*|\*[^*]+\*', content):  # Bold/italic formatting
            score += 0.05
        
        if re.search(r'^\s*[-*+]\s', content, re.MULTILINE):  # Lists
            score += 0.05
        
        # Ensure score is within bounds
        return min(1.0, max(0.0, score))
    
    def analyze_document(self, content: str) -> Dict[str, Any]:
        """
        Perform comprehensive document analysis.
        
        Args:
            content: Full document content
            
        Returns:
            Analysis results dictionary
            
        Raises:
            DocGeneratorError: If analysis fails
        """
        try:
            sections = self.extract_sections(content)
            
            # Calculate overall metrics
            total_words = sum(section.word_count for section in sections)
            average_score = sum(section.score for section in sections) / len(sections) if sections else 0.0
            
            # Check for standard sections
            section_names = {section.name for section in sections}
            missing_sections = set(self.section_headers) - section_names
            
            # Structure analysis
            has_introduction = any('intro' in name or 'description' in name 
                                 for name in section_names)
            has_examples = any('example' in name or 'usage' in name 
                             for name in section_names)
            has_installation = any('install' in name or 'setup' in name 
                                 for name in section_names)
            
            analysis = {
                'sections': [
                    {
                        'name': section.name,
                        'word_count': section.word_count,
                        'score': section.score,
                        'start_line': section.start_line,
                        'end_line': section.end_line
                    }
                    for section in sections
                ],
                'metrics': {
                    'total_sections': len(sections),
                    'total_words': total_words,
                    'average_score': round(average_score, 3),
                    'missing_sections': list(missing_sections),
                    'section_coverage': len(section_names & set(self.section_headers)) / len(self.section_headers)
                },
                'structure': {
                    'has_introduction': has_introduction,
                    'has_examples': has_examples,
                    'has_installation': has_installation,
                    'well_structured': has_introduction and has_examples and len(sections) >= 3
                },
                'recommendations': self._generate_recommendations(sections, missing_sections)
            }
            
            self.logger.info(
                f"Document analysis complete: {len(sections)} sections, "
                f"{total_words} words, score {average_score:.2f}"
            )
            
            return analysis
            
        except Exception as e:
            if isinstance(e, DocGeneratorError):
                raise
            raise DocGeneratorError(
                f"Document analysis failed: {e}",
                error_code="ANALYSIS_ERROR",
                context={'content_length': len(content)}
            )
    
    def _generate_recommendations(self, sections: List[SectionInfo], 
                                missing_sections: set) -> List[str]:
        """Generate improvement recommendations based on analysis."""
        recommendations = []
        
        # Check for missing critical sections
        if 'description' in missing_sections:
            recommendations.append("Add a clear description section explaining what this is about")
        
        if 'examples' in missing_sections and 'usage' in missing_sections:
            recommendations.append("Include examples or usage section to help users understand how to use this")
        
        if 'installation' in missing_sections:
            recommendations.append("Add installation instructions")
        
        # Check section quality
        low_score_sections = [s for s in sections if s.score < 0.3]
        if low_score_sections:
            recommendations.append(
                f"Improve content quality in: {', '.join(s.name for s in low_score_sections)}"
            )
        
        # Check for very short sections
        short_sections = [s for s in sections if s.word_count < 20]
        if short_sections:
            recommendations.append(
                f"Expand thin sections: {', '.join(s.name for s in short_sections)}"
            )
        
        # Check for missing code examples in technical sections
        technical_sections = [s for s in sections 
                            if s.name in ['examples', 'usage', 'api'] and '```' not in s.content]
        if technical_sections:
            recommendations.append(
                f"Add code examples to: {', '.join(s.name for s in technical_sections)}"
            )
        
        return recommendations[:5]  # Limit to top 5 recommendations