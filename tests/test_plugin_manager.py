"""
Test cases for PluginManager class.
"""

import pytest
import logging
from unittest.mock import patch, Mock
from doc_generator.plugin_manager import PluginManager
from doc_generator.plugins.base import RecommendationEngine
from tests.fixtures.mock_plugins import (
    MockRecommendationEngine, MockDatasetRecommender, MockWorkflowRecommender,
    FailingPlugin, DisabledPlugin
)


class TestPluginManager:
    """Test cases for PluginManager functionality."""
    
    def test_initialization(self, sample_terminology):
        """Test PluginManager initialization."""
        manager = PluginManager(terminology=sample_terminology)
        
        assert manager.terminology == sample_terminology
        assert isinstance(manager.engines, dict)
        assert len(manager.engines) == 0  # No plugins loaded yet
    
    def test_plugin_discovery_success(self, sample_terminology):
        """Test successful plugin discovery via entry points."""
        mock_entry_point = Mock()
        mock_entry_point.name = "datasets"
        mock_entry_point.load.return_value = MockDatasetRecommender
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_entry_point]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            assert "datasets" in manager.engines
            assert isinstance(manager.engines["datasets"], MockDatasetRecommender)
            assert manager.engines["datasets"].get_name() == "datasets"
    
    def test_plugin_discovery_multiple_plugins(self, sample_terminology):
        """Test discovery of multiple plugins."""
        mock_dataset_ep = Mock()
        mock_dataset_ep.name = "datasets"
        mock_dataset_ep.load.return_value = MockDatasetRecommender
        
        mock_workflow_ep = Mock()
        mock_workflow_ep.name = "workflows"
        mock_workflow_ep.load.return_value = MockWorkflowRecommender
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_dataset_ep, mock_workflow_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            assert len(manager.engines) == 2
            assert "datasets" in manager.engines
            assert "workflows" in manager.engines
    
    def test_plugin_loading_failure(self, sample_terminology, caplog):
        """Test graceful handling of plugin load failures."""
        mock_entry_point = Mock()
        mock_entry_point.name = "broken_plugin"
        mock_entry_point.load.side_effect = ImportError("Plugin broken")
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_entry_point]
            
            with caplog.at_level(logging.WARNING):
                manager = PluginManager(terminology=sample_terminology)
                manager.load_plugins()
            
            # Should not crash, and plugin should not be loaded
            assert "broken_plugin" not in manager.engines
            # Should log warning
            assert "Failed to load plugin 'broken_plugin'" in caplog.text
    
    def test_plugin_interface_validation(self, sample_terminology):
        """Test that plugins must implement RecommendationEngine interface."""
        class BadPlugin:
            """Plugin that doesn't inherit from RecommendationEngine."""
            pass
        
        mock_entry_point = Mock()
        mock_entry_point.name = "bad_plugin"
        mock_entry_point.load.return_value = BadPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_entry_point]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            # Should reject plugins that don't implement interface
            assert "bad_plugin" not in manager.engines
    
    def test_disabled_plugin_handling(self, sample_terminology):
        """Test that disabled plugins are not loaded."""
        mock_entry_point = Mock()
        mock_entry_point.name = "disabled"
        mock_entry_point.load.return_value = DisabledPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_entry_point]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            # Disabled plugin should not be in engines
            assert "disabled" not in manager.engines
    
    def test_get_engine(self, sample_terminology):
        """Test getting specific engine by name."""
        mock_ep = Mock()
        mock_ep.name = "datasets"
        mock_ep.load.return_value = MockDatasetRecommender
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            # Test existing engine
            engine = manager.get_engine("datasets")
            assert engine is not None
            assert isinstance(engine, MockDatasetRecommender)
            
            # Test non-existing engine
            none_engine = manager.get_engine("nonexistent")
            assert none_engine is None
    
    def test_get_all_engines(self, sample_terminology):
        """Test getting all loaded engines."""
        mock_dataset_ep = Mock()
        mock_dataset_ep.name = "datasets"
        mock_dataset_ep.load.return_value = MockDatasetRecommender
        
        mock_workflow_ep = Mock()
        mock_workflow_ep.name = "workflows"
        mock_workflow_ep.load.return_value = MockWorkflowRecommender
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_dataset_ep, mock_workflow_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            all_engines = manager.get_all_engines()
            
            assert len(all_engines) == 2
            assert "datasets" in all_engines
            assert "workflows" in all_engines
            assert isinstance(all_engines["datasets"], MockDatasetRecommender)
            assert isinstance(all_engines["workflows"], MockWorkflowRecommender)
    
    def test_get_recommendations(self, sample_terminology):
        """Test getting recommendations from engines."""
        mock_ep = Mock()
        mock_ep.name = "datasets"
        mock_ep.load.return_value = MockDatasetRecommender
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            results = manager.get_recommendations("Climate Science")
            
            assert "datasets" in results
            assert len(results["datasets"]) > 0
            assert results["datasets"][0]["title"] == "Mock Climate Dataset"
    
    def test_get_recommendations_specific_engines(self, sample_terminology):
        """Test getting recommendations from specific engines only."""
        mock_dataset_ep = Mock()
        mock_dataset_ep.name = "datasets"
        mock_dataset_ep.load.return_value = MockDatasetRecommender
        
        mock_workflow_ep = Mock()
        mock_workflow_ep.name = "workflows"
        mock_workflow_ep.load.return_value = MockWorkflowRecommender
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_dataset_ep, mock_workflow_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            # Request only datasets
            results = manager.get_recommendations("Topic", engine_names=["datasets"])
            
            assert "datasets" in results
            assert "workflows" not in results
    
    def test_get_recommendations_with_failing_engine(self, sample_terminology, caplog):
        """Test that failing engines don't break the whole process."""
        mock_good_ep = Mock()
        mock_good_ep.name = "datasets"
        mock_good_ep.load.return_value = MockDatasetRecommender
        
        mock_bad_ep = Mock()
        mock_bad_ep.name = "failing"
        mock_bad_ep.load.return_value = FailingPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_good_ep, mock_bad_ep]
            
            with caplog.at_level(logging.ERROR):
                manager = PluginManager(terminology=sample_terminology)
                manager.load_plugins()
                
                results = manager.get_recommendations("Topic")
            
            # Good plugin should work
            assert "datasets" in results
            assert len(results["datasets"]) > 0
            
            # Failing plugin should return empty results
            assert "failing" in results
            assert results["failing"] == []
            
            # Should log error
            assert "Error getting recommendations from engine 'failing'" in caplog.text
    
    def test_get_formatted_recommendations(self, sample_terminology):
        """Test getting formatted recommendations."""
        mock_ep = Mock()
        mock_ep.name = "datasets"
        mock_ep.load.return_value = MockDatasetRecommender
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            formatted = manager.get_formatted_recommendations("Climate Science")
            
            assert "Relevant Research Datasets:" in formatted
            assert "Mock Climate Dataset" in formatted
            assert "Mock Repository" in formatted
    
    def test_list_engines(self, sample_terminology):
        """Test listing engines with metadata."""
        mock_ep = Mock()
        mock_ep.name = "datasets"
        mock_ep.load.return_value = MockDatasetRecommender
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            engine_list = manager.list_engines()
            
            assert len(engine_list) == 1
            engine_info = engine_list[0]
            
            assert engine_info['name'] == 'datasets'
            assert engine_info['class'] == 'MockDatasetRecommender'
            assert 'supported_types' in engine_info
            assert 'priority' in engine_info
            assert 'enabled' in engine_info