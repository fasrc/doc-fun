"""Base content extractor interface for document standardization."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ExtractedContent:
    """Container for extracted document content."""
    
    title: Optional[str] = None
    sections: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    raw_content: str = ""
    format_type: str = "unknown"
    
    def __post_init__(self):
        if self.sections is None:
            self.sections = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class ContentSection:
    """Represents a section of extracted content."""
    
    title: str
    content: str
    level: int = 1
    subsections: List['ContentSection'] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []
        if self.metadata is None:
            self.metadata = {}


class ContentExtractor(ABC):
    """Abstract base class for content extractors.
    
    Content extractors parse documents in various formats and extract
    structured content for standardization processing.
    """
    
    @abstractmethod
    def can_extract(self, content: str, file_path: Optional[str] = None) -> bool:
        """Check if this extractor can handle the given content.
        
        Args:
            content: Raw content to check
            file_path: Optional file path for format detection
            
        Returns:
            True if this extractor can process the content
        """
        pass
    
    @abstractmethod
    def extract(self, content: str, file_path: Optional[str] = None) -> ExtractedContent:
        """Extract structured content from raw input.
        
        Args:
            content: Raw content to extract from
            file_path: Optional file path for context
            
        Returns:
            ExtractedContent with parsed structure
        """
        pass
    
    @abstractmethod
    def get_format_type(self) -> str:
        """Get the format type this extractor handles.
        
        Returns:
            String identifier for the format (e.g., 'html', 'markdown', 'pdf')
        """
        pass
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of file extensions this extractor supports.
        
        Returns:
            List of file extensions (including the dot)
        """
        return []
    
    def preprocess_content(self, content: str) -> str:
        """Preprocess content before extraction.
        
        Override this method to apply format-specific preprocessing.
        
        Args:
            content: Raw content
            
        Returns:
            Preprocessed content
        """
        return content
    
    def postprocess_content(self, extracted: ExtractedContent) -> ExtractedContent:
        """Postprocess extracted content.
        
        Override this method to apply format-specific postprocessing.
        
        Args:
            extracted: Extracted content
            
        Returns:
            Postprocessed extracted content
        """
        return extracted