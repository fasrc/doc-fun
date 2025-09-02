"""
Test cases for PluginManager class.
"""

import pytest
import logging
from unittest.mock import patch, Mock
from pathlib import Path
from doc_generator.plugin_manager import PluginManager
from doc_generator.plugins.base import RecommendationEngine
from doc_generator.plugins.analysis_base import AnalysisPlugin
from tests.fixtures.mock_plugins import (
    MockRecommendationEngine, MockDatasetRecommender, MockWorkflowRecommender,
    FailingPlugin, DisabledPlugin
)


# Mock Analysis Plugins for testing
class MockAnalysisPlugin(AnalysisPlugin):
    """Mock analysis plugin for testing."""
    
    def __init__(self, logger=None, config=None):
        self.logger = logger or Mock()
        self.config = config or {}
        
    def get_name(self):
        return "mock_analyzer"
        
    def get_priority(self):
        return 50
        
    def is_enabled(self):
        return True
        
    def analyze(self, documents, topic):
        return {
            'analyzed_documents': len(documents),
            'topic': topic,
            'findings': ['Mock finding 1', 'Mock finding 2']
        }
        
    def generate_report(self, analysis_results, topic):
        return f"Mock Analysis Report for {topic}\nFindings: {', '.join(analysis_results.get('findings', []))}"
        
    def save_artifacts(self, analysis_results, output_dir, topic):
        artifacts_file = output_dir / f"{topic.lower().replace(' ', '_')}_mock_analysis.txt"
        artifacts_file.write_text("Mock analysis artifacts")
        return [artifacts_file]


class MockReporterPlugin(AnalysisPlugin):
    """Mock reporter plugin for testing."""
    
    def __init__(self, logger=None, config=None):
        self.logger = logger or Mock()
        self.config = config or {}
        
    def get_name(self):
        return "reporter"
        
    def get_priority(self):
        return 80
        
    def is_enabled(self):
        return True
        
    def analyze(self, documents, topic):
        return {
            'report_type': 'comprehensive',
            'document_count': len(documents),
            'summary': f'Analysis report for {topic}'
        }
        
    def generate_report(self, analysis_results, topic):
        return f"# Comprehensive Report for {topic}\n\n{analysis_results.get('summary', 'No summary available')}"
        
    def save_artifacts(self, analysis_results, output_dir, topic):
        report_file = output_dir / f"{topic.lower().replace(' ', '_')}_report.md"
        html_file = output_dir / f"{topic.lower().replace(' ', '_')}_report.html"
        report_file.write_text(f"# Report for {topic}")
        html_file.write_text(f"<h1>Report for {topic}</h1>")
        return [report_file, html_file]


class DisabledAnalysisPlugin(AnalysisPlugin):
    """Disabled analysis plugin for testing."""
    
    def __init__(self, logger=None, config=None):
        self.logger = logger or Mock()
        self.config = config or {}
        
    def get_name(self):
        return "disabled_analyzer"
        
    def get_priority(self):
        return 30
        
    def is_enabled(self):
        return False
        
    def analyze(self, documents, topic):
        return {}
        
    def generate_report(self, analysis_results, topic):
        return f"Disabled Analysis Report for {topic}"
        
    def save_artifacts(self, analysis_results, output_dir, topic):
        return []


class FailingAnalysisPlugin(AnalysisPlugin):
    """Failing analysis plugin for testing."""
    
    def __init__(self, logger=None, config=None):
        self.logger = logger or Mock()
        self.config = config or {}
        
    def get_name(self):
        return "failing_analyzer"
        
    def get_priority(self):
        return 60
        
    def is_enabled(self):
        return True
        
    def analyze(self, documents, topic):
        raise RuntimeError("Mock analysis failure")
        
    def generate_report(self, analysis_results, topic):
        raise RuntimeError("Mock report generation failure")
        
    def save_artifacts(self, analysis_results, output_dir, topic):
        raise RuntimeError("Mock save failure")


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

    # Analysis Plugin Tests
    def test_load_analysis_plugins_success(self, sample_terminology):
        """Test successful analysis plugin discovery and loading."""
        mock_analyzer_ep = Mock()
        mock_analyzer_ep.name = "mock_analyzer"
        mock_analyzer_ep.load.return_value = MockAnalysisPlugin
        
        mock_reporter_ep = Mock()
        mock_reporter_ep.name = "reporter"  
        mock_reporter_ep.load.return_value = MockReporterPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_analyzer_ep, mock_reporter_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_analysis_plugins()
            
            assert len(manager.analysis_plugins) == 2
            assert "mock_analyzer" in manager.analysis_plugins
            assert "reporter" in manager.analysis_plugins
            assert isinstance(manager.analysis_plugins["mock_analyzer"], MockAnalysisPlugin)
            assert isinstance(manager.analysis_plugins["reporter"], MockReporterPlugin)

    def test_load_analysis_plugins_with_config(self, sample_terminology):
        """Test analysis plugin loading with configuration."""
        mock_ep = Mock()
        mock_ep.name = "mock_analyzer"
        mock_ep.load.return_value = MockAnalysisPlugin
        
        config = {
            "mock_analyzer": {
                "enabled": True,
                "custom_setting": "test_value"
            }
        }
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_analysis_plugins(config)
            
            plugin = manager.analysis_plugins["mock_analyzer"]
            assert plugin.config == config["mock_analyzer"]

    def test_load_analysis_plugins_failure(self, sample_terminology, caplog):
        """Test graceful handling of analysis plugin load failures."""
        mock_ep = Mock()
        mock_ep.name = "broken_analyzer"
        mock_ep.load.side_effect = ImportError("Analysis plugin broken")
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            with caplog.at_level(logging.WARNING):
                manager = PluginManager(terminology=sample_terminology)
                manager.load_analysis_plugins()
            
            assert "broken_analyzer" not in manager.analysis_plugins
            assert "Failed to load analysis plugin 'broken_analyzer'" in caplog.text

    def test_load_analysis_plugin_interface_validation(self, sample_terminology):
        """Test that analysis plugins must implement AnalysisPlugin interface."""
        class BadAnalysisPlugin:
            """Plugin that doesn't inherit from AnalysisPlugin."""
            pass
        
        mock_ep = Mock()
        mock_ep.name = "bad_analyzer"
        mock_ep.load.return_value = BadAnalysisPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_analysis_plugins()
            
            assert "bad_analyzer" not in manager.analysis_plugins

    def test_disabled_analysis_plugin_handling(self, sample_terminology):
        """Test that disabled analysis plugins are not loaded."""
        mock_ep = Mock()
        mock_ep.name = "disabled_analyzer"
        mock_ep.load.return_value = DisabledAnalysisPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_analysis_plugins()
            
            assert "disabled_analyzer" not in manager.analysis_plugins

    def test_analysis_plugin_name_mismatch_warning(self, sample_terminology, caplog):
        """Test warning when analysis plugin name doesn't match entry point."""
        class MismatchedPlugin(AnalysisPlugin):
            def get_name(self):
                return "different_name"
            def get_priority(self):
                return 50
            def is_enabled(self):
                return True
            def analyze(self, documents, topic):
                return {}
            def generate_report(self, analysis_results, topic):
                return "Mismatched plugin report"
            def save_artifacts(self, analysis_results, output_dir, topic):
                return []
        
        mock_ep = Mock()
        mock_ep.name = "entry_point_name"
        mock_ep.load.return_value = MismatchedPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            with caplog.at_level(logging.WARNING):
                manager = PluginManager(terminology=sample_terminology)
                manager.load_analysis_plugins()
            
            assert "Plugin name mismatch" in caplog.text
            assert "entry_point_name" in caplog.text
            assert "different_name" in caplog.text

    def test_get_analysis_plugin(self, sample_terminology):
        """Test getting specific analysis plugin by name."""
        mock_ep = Mock()
        mock_ep.name = "mock_analyzer"
        mock_ep.load.return_value = MockAnalysisPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_analysis_plugins()
            
            # Test existing plugin
            plugin = manager.get_analysis_plugin("mock_analyzer")
            assert plugin is not None
            assert isinstance(plugin, MockAnalysisPlugin)
            
            # Test non-existing plugin
            none_plugin = manager.get_analysis_plugin("nonexistent")
            assert none_plugin is None

    def test_get_all_analysis_plugins(self, sample_terminology):
        """Test getting all loaded analysis plugins."""
        mock_analyzer_ep = Mock()
        mock_analyzer_ep.name = "mock_analyzer"
        mock_analyzer_ep.load.return_value = MockAnalysisPlugin
        
        mock_reporter_ep = Mock()
        mock_reporter_ep.name = "reporter"
        mock_reporter_ep.load.return_value = MockReporterPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_analyzer_ep, mock_reporter_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_analysis_plugins()
            
            all_plugins = manager.get_all_analysis_plugins()
            
            assert len(all_plugins) == 2
            assert "mock_analyzer" in all_plugins
            assert "reporter" in all_plugins
            assert isinstance(all_plugins["mock_analyzer"], MockAnalysisPlugin)
            assert isinstance(all_plugins["reporter"], MockReporterPlugin)

    def test_run_analysis_pipeline_all_plugins(self, sample_terminology, tmp_path):
        """Test running analysis pipeline with all plugins."""
        mock_analyzer_ep = Mock()
        mock_analyzer_ep.name = "mock_analyzer"
        mock_analyzer_ep.load.return_value = MockAnalysisPlugin
        
        mock_reporter_ep = Mock()
        mock_reporter_ep.name = "reporter"
        mock_reporter_ep.load.return_value = MockReporterPlugin
        
        documents = [
            {'path': '/test/doc1.html', 'content': '<html>Test content</html>'},
            {'path': '/test/doc2.html', 'content': '<html>More content</html>'}
        ]
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_analyzer_ep, mock_reporter_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_analysis_plugins()
            
            results = manager.run_analysis_pipeline(documents, "Test Topic", tmp_path)
            
            # Should run reporter first (higher priority)
            assert len(results) == 2
            assert "reporter" in results
            assert "mock_analyzer" in results
            
            # Verify reporter results
            reporter_result = results["reporter"]
            assert "analysis" in reporter_result
            assert "artifacts" in reporter_result
            assert reporter_result["analysis"]["document_count"] == 2
            assert len(reporter_result["artifacts"]) == 2
            
            # Verify mock analyzer results
            analyzer_result = results["mock_analyzer"]
            assert "analysis" in analyzer_result
            assert "artifacts" in analyzer_result
            assert analyzer_result["analysis"]["analyzed_documents"] == 2
            assert len(analyzer_result["artifacts"]) == 1

    def test_run_analysis_pipeline_specific_plugins(self, sample_terminology, tmp_path):
        """Test running analysis pipeline with specific plugins only."""
        mock_analyzer_ep = Mock()
        mock_analyzer_ep.name = "mock_analyzer"
        mock_analyzer_ep.load.return_value = MockAnalysisPlugin
        
        mock_reporter_ep = Mock()
        mock_reporter_ep.name = "reporter"
        mock_reporter_ep.load.return_value = MockReporterPlugin
        
        documents = [{'path': '/test/doc.html', 'content': '<html>Content</html>'}]
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_analyzer_ep, mock_reporter_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_analysis_plugins()
            
            # Request only mock_analyzer
            results = manager.run_analysis_pipeline(
                documents, "Test Topic", tmp_path, plugin_names=["mock_analyzer"]
            )
            
            assert "mock_analyzer" in results
            assert "reporter" not in results

    def test_run_analysis_pipeline_unknown_plugin(self, sample_terminology, tmp_path, caplog):
        """Test running analysis pipeline with unknown plugin name."""
        documents = [{'path': '/test/doc.html', 'content': '<html>Content</html>'}]
        
        with caplog.at_level(logging.WARNING):
            manager = PluginManager(terminology=sample_terminology)
            results = manager.run_analysis_pipeline(
                documents, "Test Topic", tmp_path, plugin_names=["nonexistent"]
            )
        
        assert results == {}
        assert "Unknown analysis plugin: nonexistent" in caplog.text

    def test_run_analysis_pipeline_with_failing_plugin(self, sample_terminology, tmp_path, caplog):
        """Test that failing analysis plugins don't break the pipeline."""
        mock_good_ep = Mock()
        mock_good_ep.name = "mock_analyzer"
        mock_good_ep.load.return_value = MockAnalysisPlugin
        
        mock_bad_ep = Mock()
        mock_bad_ep.name = "failing_analyzer"
        mock_bad_ep.load.return_value = FailingAnalysisPlugin
        
        documents = [{'path': '/test/doc.html', 'content': '<html>Content</html>'}]
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_good_ep, mock_bad_ep]
            
            with caplog.at_level(logging.ERROR):
                manager = PluginManager(terminology=sample_terminology)
                manager.load_analysis_plugins()
                
                results = manager.run_analysis_pipeline(documents, "Test Topic", tmp_path)
            
            # Good plugin should work
            assert "mock_analyzer" in results
            assert "analysis" in results["mock_analyzer"]
            
            # Failing plugin should have error recorded
            assert "failing_analyzer" in results
            assert "error" in results["failing_analyzer"]
            assert "Mock analysis failure" in results["failing_analyzer"]["error"]
            assert results["failing_analyzer"]["artifacts"] == []
            
            # Should log error
            assert "Error running analysis plugin 'failing_analyzer'" in caplog.text

    def test_list_analysis_plugins(self, sample_terminology):
        """Test listing analysis plugins with metadata."""
        mock_ep = Mock()
        mock_ep.name = "mock_analyzer"
        mock_ep.load.return_value = MockAnalysisPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_analysis_plugins()
            
            plugin_list = manager.list_analysis_plugins()
            
            assert len(plugin_list) == 1
            plugin_info = plugin_list[0]
            
            assert plugin_info['name'] == 'mock_analyzer'
            assert plugin_info['class'] == 'MockAnalysisPlugin'
            assert plugin_info['priority'] == 50
            assert plugin_info['enabled'] == True
            assert 'module' in plugin_info

    # Error handling and edge case tests
    def test_load_plugins_discovery_error(self, sample_terminology, caplog):
        """Test handling of entry points discovery errors."""
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.side_effect = RuntimeError("Entry points discovery failed")
            
            with caplog.at_level(logging.ERROR):
                manager = PluginManager(terminology=sample_terminology)
                manager.load_plugins()
            
            assert "Error during plugin discovery" in caplog.text

    def test_load_analysis_plugins_discovery_error(self, sample_terminology, caplog):
        """Test handling of analysis plugin discovery errors."""
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.side_effect = RuntimeError("Analysis entry points failed")
            
            with caplog.at_level(logging.ERROR):
                manager = PluginManager(terminology=sample_terminology)
                manager.load_analysis_plugins()
            
            assert "Error during analysis plugin discovery" in caplog.text

    def test_get_recommendations_unknown_engine_warning(self, sample_terminology, caplog):
        """Test warning when requesting recommendations from unknown engine."""
        with caplog.at_level(logging.WARNING):
            manager = PluginManager(terminology=sample_terminology)
            results = manager.get_recommendations("Topic", engine_names=["nonexistent"])
        
        assert results == {}  # Unknown engines are skipped, not added to results
        assert "Unknown recommendation engine: nonexistent" in caplog.text

    def test_get_formatted_recommendations_error_handling(self, sample_terminology, caplog):
        """Test error handling in formatted recommendations."""
        mock_ep = Mock()
        mock_ep.name = "failing"
        mock_ep.load.return_value = FailingPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            with caplog.at_level(logging.ERROR):
                manager = PluginManager(terminology=sample_terminology)
                manager.load_plugins()
                
                # This should handle the error gracefully
                result = manager.get_formatted_recommendations("Topic")
        
        assert "Error formatting recommendations from engine 'failing'" in caplog.text
        
    def test_empty_analysis_plugins_list(self, sample_terminology):
        """Test behavior with no analysis plugins loaded."""
        manager = PluginManager(terminology=sample_terminology)
        
        assert manager.get_all_analysis_plugins() == {}
        assert manager.get_analysis_plugin("any") is None
        assert manager.list_analysis_plugins() == []
        
        # Empty pipeline should return empty results
        results = manager.run_analysis_pipeline([], "Topic", Path("/tmp"))
        assert results == {}

    def test_recommendation_plugin_name_mismatch_warning(self, sample_terminology, caplog):
        """Test warning when recommendation plugin name doesn't match entry point."""
        class MismatchedRecommendationPlugin(RecommendationEngine):
            def __init__(self, terminology=None, logger=None):
                super().__init__(terminology, logger)
                
            def get_name(self):
                return "different_name"
                
            def get_priority(self):
                return 50
                
            def is_enabled(self):
                return True
                
            def get_supported_types(self):
                return ["test"]
                
            def get_recommendations(self, topic, context=None):
                return []
                
            def get_formatted_recommendations(self, topic, context=None):
                return ""
        
        mock_ep = Mock()
        mock_ep.name = "entry_point_name"
        mock_ep.load.return_value = MismatchedRecommendationPlugin
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            with caplog.at_level(logging.WARNING):
                manager = PluginManager(terminology=sample_terminology)
                manager.load_plugins()
            
            assert "Plugin name mismatch" in caplog.text
            assert "entry_point_name" in caplog.text
            assert "different_name" in caplog.text

    def test_get_formatted_recommendations_with_nonexistent_engine(self, sample_terminology):
        """Test get_formatted_recommendations skips unknown engines."""
        mock_ep = Mock()
        mock_ep.name = "datasets"
        mock_ep.load.return_value = MockDatasetRecommender
        
        with patch('doc_generator.plugin_manager.entry_points') as mock_eps:
            mock_eps.return_value = [mock_ep]
            
            manager = PluginManager(terminology=sample_terminology)
            manager.load_plugins()
            
            # Request specific engines including nonexistent one
            result = manager.get_formatted_recommendations("Topic", engine_names=["datasets", "nonexistent"])
            
            # Should only get results from existing engine
            assert "Mock Climate Dataset" in result
            # No errors should be raised for the nonexistent engine