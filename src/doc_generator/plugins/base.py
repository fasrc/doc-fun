"""
Base classes for recommendation engine plugins.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import logging


class RecommendationEngine(ABC):
    """
    Abstract base class for all recommendation engines.
    
    Recommendation engines analyze a topic and provide relevant recommendations
    such as HPC modules, datasets, code examples, workflows, etc.
    """
    
    def __init__(self, terminology: Optional[Dict] = None, logger: Optional[logging.Logger] = None, **kwargs: Any):
        """
        Initialize the recommendation engine.
        
        Args:
            terminology: Terminology configuration dictionary
            logger: Logger instance for debugging and error reporting
            **kwargs: Additional configuration parameters
        """
        self.terminology = terminology or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Store any additional configuration
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return the unique name identifier for this recommendation engine.
        
        Returns:
            String identifier (e.g., 'modules', 'datasets', 'workflows')
        """
        pass
    
    @abstractmethod
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        """
        Get recommendations for the given topic.
        
        Args:
            topic: The topic to generate recommendations for
            context: Optional context information (user preferences, etc.)
            
        Returns:
            List of recommendation dictionaries. Each dictionary should contain
            at minimum: name/title, description, and relevance_score fields.
        """
        pass
    
    def get_supported_types(self) -> List[str]:
        """
        Return list of supported recommendation types.
        
        Returns:
            List of strings describing what this engine recommends
            (e.g., ['hpc_modules', 'software'])
        """
        return [self.get_name()]
    
    def get_formatted_recommendations(self, topic: str, context: Optional[Dict] = None) -> str:
        """
        Get formatted recommendations for inclusion in documentation context.
        
        Args:
            topic: The topic to generate recommendations for
            context: Optional context information
            
        Returns:
            Formatted string ready for inclusion in prompt context.
            Returns empty string if no recommendations available.
        """
        recommendations = self.get_recommendations(topic, context)
        
        if not recommendations:
            return ""
        
        # Default formatting - plugins can override for custom formatting
        engine_name = self.get_name().title()
        formatted = f"\n## {engine_name} Recommendations:\n\n"
        
        for rec in recommendations:
            name = rec.get('name') or rec.get('title', 'Unknown')
            formatted += f"**{name}**\n"
            
            if 'description' in rec:
                formatted += f"- Description: {rec['description']}\n"
            
            if 'url' in rec:
                formatted += f"- URL: {rec['url']}\n"
            
            if 'relevance_score' in rec:
                formatted += f"- Relevance Score: {rec['relevance_score']}\n"
            
            formatted += "\n"
        
        return formatted
    
    def is_enabled(self) -> bool:
        """
        Check if this recommendation engine is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return True  # Default to enabled
    
    def get_priority(self) -> int:
        """
        Get the priority of this recommendation engine.
        Higher numbers = higher priority for ordering recommendations.
        
        Returns:
            Integer priority (default: 50)
        """
        return 50