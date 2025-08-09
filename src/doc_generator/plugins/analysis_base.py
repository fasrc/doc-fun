"""
Base classes for analysis and post-processing plugins.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging


class AnalysisPlugin(ABC):
    """
    Abstract base class for all analysis and post-processing plugins.
    
    Analysis plugins process generated documentation to provide insights,
    create compilations, generate reports, or perform validation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict] = None, **kwargs: Any):
        """
        Initialize the analysis plugin.
        
        Args:
            logger: Logger instance for debugging and error reporting
            config: Configuration dictionary for the plugin
            **kwargs: Additional configuration parameters
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Store any additional configuration
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return the unique name identifier for this analysis plugin.
        
        Returns:
            String identifier (e.g., 'compiler', 'reporter', 'validator')
        """
        pass
    
    @abstractmethod
    def analyze(self, documents: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        """
        Analyze the generated documents and return analysis results.
        
        Args:
            documents: List of document dictionaries containing:
                       - 'path': Path to the document file
                       - 'content': HTML content of the document
                       - 'metadata': Optional metadata about generation
            topic: The topic that was used for generation
            
        Returns:
            Dictionary containing analysis results specific to this plugin
        """
        pass
    
    @abstractmethod
    def generate_report(self, analysis_results: Dict[str, Any], topic: str) -> str:
        """
        Generate a human-readable report from analysis results.
        
        Args:
            analysis_results: Results from the analyze() method
            topic: The topic that was used for generation
            
        Returns:
            Formatted report as a string (format depends on plugin)
        """
        pass
    
    @abstractmethod
    def save_artifacts(self, results: Dict[str, Any], output_dir: Path, topic: str) -> List[Path]:
        """
        Save any artifacts (reports, compilations, etc.) to the output directory.
        
        Args:
            results: Combined results including analysis and any generated content
            output_dir: Directory where artifacts should be saved
            topic: The topic that was used for generation
            
        Returns:
            List of paths to saved artifact files
        """
        pass
    
    def get_priority(self) -> int:
        """
        Get the priority of this analysis plugin.
        Higher numbers = higher priority for execution order.
        
        Returns:
            Integer priority (default: 50)
        """
        return 50
    
    def is_enabled(self) -> bool:
        """
        Check if this analysis plugin is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return True
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported output formats for this plugin.
        
        Returns:
            List of format strings (e.g., ['html', 'markdown', 'json'])
        """
        return []
    
    def sanitize_filename(self, topic: str) -> str:
        """
        Convert a topic string to a safe filename.
        
        Args:
            topic: The topic string
            
        Returns:
            Sanitized filename-safe string
        """
        # Replace spaces with underscores and remove special characters
        safe_name = topic.lower()
        safe_name = safe_name.replace(' ', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
        return safe_name