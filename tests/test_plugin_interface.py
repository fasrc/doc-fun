"""
Test cases for plugin interface and base classes.
"""

import pytest
import logging
from typing import List, Dict, Optional
from doc_generator.plugins.base import RecommendationEngine


class TestRecommendationEngineInterface:
    """Test the RecommendationEngine abstract base class."""
    
    def test_recommendation_engine_is_abstract(self):
        """Test that RecommendationEngine cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RecommendationEngine()
    
    def test_incomplete_plugin_implementation(self):
        """Test that incomplete plugins raise TypeError."""
        
        class IncompletePlugin(RecommendationEngine):
            def get_name(self):
                return "incomplete"
            # Missing get_recommendations method
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompletePlugin()
    
    def test_valid_plugin_implementation(self):
        """Test that properly implemented plugins work correctly."""
        
        class ValidPlugin(RecommendationEngine):
            def get_name(self) -> str:
                return "valid"
            
            def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
                return [{"title": "Test", "relevance_score": 1}]
        
        plugin = ValidPlugin()
        assert plugin.get_name() == "valid"
        assert len(plugin.get_recommendations("test")) == 1
        assert plugin.get_recommendations("test")[0]["title"] == "Test"
    
    def test_plugin_initialization_with_parameters(self):
        """Test plugin initialization with various parameters."""
        
        class TestPlugin(RecommendationEngine):
            def get_name(self) -> str:
                return "test"
            
            def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
                return []
        
        # Test with terminology
        terminology = {"test_key": "test_value"}
        plugin = TestPlugin(terminology=terminology)
        assert plugin.terminology == terminology
        
        # Test with logger
        logger = logging.getLogger("test")
        plugin = TestPlugin(logger=logger)
        assert plugin.logger == logger
        
        # Test with additional kwargs
        plugin = TestPlugin(custom_param="custom_value")
        assert hasattr(plugin, "custom_param")
        assert plugin.custom_param == "custom_value"
    
    def test_default_methods(self):
        """Test default implementations of optional methods."""
        
        class MinimalPlugin(RecommendationEngine):
            def get_name(self) -> str:
                return "minimal"
            
            def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
                return [
                    {"name": "Test Item", "description": "Test description", "relevance_score": 5}
                ]
        
        plugin = MinimalPlugin()
        
        # Test default implementations
        assert plugin.get_supported_types() == ["minimal"]  # Default to plugin name
        assert plugin.is_enabled() == True  # Default to enabled
        assert plugin.get_priority() == 50  # Default priority
    
    def test_get_formatted_recommendations_default(self):
        """Test default formatting of recommendations."""
        
        class FormattingTestPlugin(RecommendationEngine):
            def get_name(self) -> str:
                return "formatting"
            
            def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
                return [
                    {
                        "name": "Test Item 1",
                        "description": "First test item",
                        "url": "https://example.com/1",
                        "relevance_score": 8
                    },
                    {
                        "title": "Test Item 2",  # Using title instead of name
                        "description": "Second test item",
                        "relevance_score": 6
                    }
                ]
        
        plugin = FormattingTestPlugin()
        formatted = plugin.get_formatted_recommendations("test topic")
        
        # Check that formatting includes expected elements
        assert "## Formatting Recommendations:" in formatted
        assert "**Test Item 1**" in formatted
        assert "**Test Item 2**" in formatted
        assert "Description: First test item" in formatted
        assert "URL: https://example.com/1" in formatted
        assert "Relevance Score: 8" in formatted
        assert "Relevance Score: 6" in formatted
    
    def test_get_formatted_recommendations_empty(self):
        """Test formatting when no recommendations are available."""
        
        class EmptyPlugin(RecommendationEngine):
            def get_name(self) -> str:
                return "empty"
            
            def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
                return []
        
        plugin = EmptyPlugin()
        formatted = plugin.get_formatted_recommendations("test topic")
        
        # Should return empty string when no recommendations
        assert formatted == ""
    
    def test_custom_formatted_recommendations(self):
        """Test that plugins can override the formatting method."""
        
        class CustomFormattingPlugin(RecommendationEngine):
            def get_name(self) -> str:
                return "custom"
            
            def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
                return [{"name": "Custom Item", "value": 42}]
            
            def get_formatted_recommendations(self, topic: str, context: Optional[Dict] = None) -> str:
                recs = self.get_recommendations(topic, context)
                if not recs:
                    return ""
                return f"Custom format: {recs[0]['name']} = {recs[0]['value']}"
        
        plugin = CustomFormattingPlugin()
        formatted = plugin.get_formatted_recommendations("test")
        
        assert formatted == "Custom format: Custom Item = 42"
    
    def test_plugin_context_parameter(self):
        """Test that context parameter is properly passed through."""
        
        class ContextAwarePlugin(RecommendationEngine):
            def get_name(self) -> str:
                return "context"
            
            def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
                if context and context.get("filter_high_score"):
                    return [{"name": "High Score Item", "score": 10}]
                else:
                    return [{"name": "Regular Item", "score": 5}]
        
        plugin = ContextAwarePlugin()
        
        # Test without context
        recs = plugin.get_recommendations("test")
        assert len(recs) == 1
        assert recs[0]["name"] == "Regular Item"
        
        # Test with context
        recs = plugin.get_recommendations("test", context={"filter_high_score": True})
        assert len(recs) == 1
        assert recs[0]["name"] == "High Score Item"
    
    def test_plugin_priority_and_enabled_customization(self):
        """Test customizing priority and enabled status."""
        
        class CustomPlugin(RecommendationEngine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._priority = kwargs.get('priority', 75)
                self._enabled = kwargs.get('enabled', True)
            
            def get_name(self) -> str:
                return "custom"
            
            def get_recommendations(self, topic: str, context: Optional[Dict] = None) -> List[Dict]:
                return []
            
            def get_priority(self) -> int:
                return self._priority
            
            def is_enabled(self) -> bool:
                return self._enabled
        
        # Test high priority plugin
        high_priority = CustomPlugin(priority=100)
        assert high_priority.get_priority() == 100
        assert high_priority.is_enabled() == True
        
        # Test disabled plugin
        disabled = CustomPlugin(enabled=False)
        assert disabled.get_priority() == 75  # Custom default
        assert disabled.is_enabled() == False