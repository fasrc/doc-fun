"""
Mock plugins for testing plugin system without external dependencies.
"""

from typing import List, Dict, Optional
import sys
import os
# Add src directory to path for tests
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
sys.path.insert(0, src_path)

from doc_generator.plugins.base import RecommendationEngine


class MockRecommendationEngine(RecommendationEngine):
    """Base mock plugin for testing."""
    
    def __init__(self, recommendations=None, **kwargs):
        super().__init__(**kwargs)
        self._recommendations = recommendations or []
        self._name = kwargs.get('name', 'mock')
    
    def get_name(self) -> str:
        return self._name
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        return self._recommendations
    
    def get_supported_types(self) -> List[str]:
        return ["mock"]


class MockDatasetRecommender(MockRecommendationEngine):
    """Mock dataset recommender for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(recommendations=[
            {
                "title": "Mock Climate Dataset",
                "url": "https://example.com/dataset1",
                "source": "Mock Repository",
                "relevance_score": 8,
                "description": "Mock climate dataset for testing"
            },
            {
                "title": "Mock Biology Dataset", 
                "url": "https://example.com/dataset2",
                "source": "Mock Bio Repository",
                "relevance_score": 6,
                "description": "Mock biology dataset for testing"
            }
        ], name='datasets', **kwargs)
    
    def get_supported_types(self) -> List[str]:
        return ["datasets", "research_data"]
    
    def get_formatted_recommendations(self, topic: str, context: Optional[Dict] = None) -> str:
        recommendations = self.get_recommendations(topic, context)
        if not recommendations:
            return ""
        
        formatted = "\n## Relevant Research Datasets:\n\n"
        for rec in recommendations:
            formatted += f"**{rec['title']}**\n"
            formatted += f"- Source: {rec['source']}\n"
            formatted += f"- URL: {rec['url']}\n"
            formatted += f"- Description: {rec['description']}\n"
            formatted += f"- Relevance Score: {rec['relevance_score']}/10\n\n"
        
        return formatted


class MockWorkflowRecommender(MockRecommendationEngine):
    """Mock workflow recommender for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(recommendations=[
            {
                "name": "Parallel Processing Workflow",
                "description": "Template for parallel processing jobs",
                "type": "slurm_template",
                "relevance_score": 9
            }
        ], name='workflows', **kwargs)
    
    def get_supported_types(self) -> List[str]:
        return ["workflows", "templates", "job_scripts"]


class FailingPlugin(MockRecommendationEngine):
    """Plugin that fails for testing error handling."""
    
    def __init__(self, **kwargs):
        super().__init__(name='failing', **kwargs)
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
        raise Exception("Plugin deliberately failing for testing")
    
    def get_formatted_recommendations(self, topic: str, context: Optional[Dict] = None) -> str:
        raise Exception("Plugin deliberately failing for testing")


class DisabledPlugin(MockRecommendationEngine):
    """Plugin that reports as disabled for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(name='disabled', **kwargs)
    
    def is_enabled(self) -> bool:
        return False