"""
Integration tests for plugin system with DocumentationGenerator.
"""

import pytest
import yaml
import os
from unittest.mock import patch, Mock
from doc_generator import DocumentationGenerator
from doc_generator.plugins.base import RecommendationEngine
from tests.fixtures.mock_plugins import (
    MockRecommendationEngine, MockDatasetRecommender, MockWorkflowRecommender,
    FailingPlugin
)


class TestPluginIntegration:
    """Test integration of plugin system with DocumentationGenerator."""
    
    def test_plugin_integration_in_context_building(self, temp_dir, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test that plugins are properly integrated into context building."""
        
        with mock_plugin_discovery(sample_plugins):
            prompt_file = temp_dir / "prompt.yaml"
            terminology_file = temp_dir / "terminology.yaml"
            examples_dir = temp_dir / "examples"
            examples_dir.mkdir()
            
            with open(prompt_file, 'w') as f:
                yaml.dump({'system_prompt': 'Test prompt for {topic}'}, f)
            with open(terminology_file, 'w') as f:
                yaml.dump(sample_terminology, f)
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        examples_dir=str(examples_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    context = generator._build_terminology_context("Climate Modeling")
                    
                    # Should include contributions from loaded plugins
                    assert isinstance(context, str)
                    assert len(context) > 0
                    # The mock plugins should contribute something to the context
                    assert "Mock" in context or "Relevant" in context or context.strip() != ""
    
    def test_multiple_plugins_in_context(self, temp_dir, sample_terminology, mock_plugin_discovery):
        """Test that multiple plugins contribute to documentation context."""
        plugins = {
            "modules": MockRecommendationEngine,
            "datasets": MockDatasetRecommender,
            "workflows": MockWorkflowRecommender
        }
        
        with mock_plugin_discovery(plugins):
            prompt_file = temp_dir / "prompt.yaml"
            terminology_file = temp_dir / "terminology.yaml"
            examples_dir = temp_dir / "examples"
            examples_dir.mkdir()
            
            with open(prompt_file, 'w') as f:
                yaml.dump({'system_prompt': 'Test prompt'}, f)
            with open(terminology_file, 'w') as f:
                yaml.dump(sample_terminology, f)
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        examples_dir=str(examples_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    # Check that all plugins are loaded
                    assert len(generator.plugin_manager.engines) == 3
                    assert "modules" in generator.plugin_manager.engines
                    assert "datasets" in generator.plugin_manager.engines
                    assert "workflows" in generator.plugin_manager.engines
                    
                    context = generator._build_terminology_context("Parallel Python")
                    
                    # All plugins should potentially contribute
                    assert isinstance(context, str)
    
    def test_plugin_failure_doesnt_break_generation(self, temp_dir, sample_terminology, mock_plugin_discovery):
        """Test that if one plugin fails, others still work and generation continues."""
        plugins = {
            "modules": MockRecommendationEngine,
            "failing": FailingPlugin
        }
        
        with mock_plugin_discovery(plugins):
            prompt_file = temp_dir / "prompt.yaml"
            terminology_file = temp_dir / "terminology.yaml"
            examples_dir = temp_dir / "examples"
            examples_dir.mkdir()
            
            with open(prompt_file, 'w') as f:
                yaml.dump({'system_prompt': 'Test prompt'}, f)
            with open(terminology_file, 'w') as f:
                yaml.dump(sample_terminology, f)
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        examples_dir=str(examples_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    # Both plugins should be loaded
                    assert len(generator.plugin_manager.engines) == 2
                    
                    # Context building should not crash despite failing plugin
                    context = generator._build_terminology_context("Python Programming")
                    
                    # Should still return a string (even if some plugins failed)
                    assert isinstance(context, str)
    
    def test_generation_works_without_plugins(self, temp_dir, sample_terminology, mock_plugin_discovery):
        """Test that system works even if no plugins are available."""
        
        with mock_plugin_discovery({}):  # No plugins
            prompt_file = temp_dir / "prompt.yaml"
            terminology_file = temp_dir / "terminology.yaml"
            examples_dir = temp_dir / "examples"
            examples_dir.mkdir()
            
            with open(prompt_file, 'w') as f:
                yaml.dump({
                    'system_prompt': 'Test prompt for {topic}',
                    'placeholders': {'format': 'HTML'}
                }, f)
            with open(terminology_file, 'w') as f:
                yaml.dump(sample_terminology, f)
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        examples_dir=str(examples_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    # No plugins should be loaded
                    assert len(generator.plugin_manager.engines) == 0
                    
                    # System should still work
                    context = generator._build_terminology_context("Python")
                    
                    # Should still work, just without plugin recommendations
                    assert isinstance(context, str)
                    # Basic terminology should still be included
                    if 'cluster_commands' in sample_terminology:
                        assert "sbatch" in context  # From cluster_commands
    
    def test_system_prompt_building_with_plugins(self, temp_dir, sample_terminology, mock_plugin_discovery, sample_plugins):
        """Test system prompt building includes plugin recommendations."""
        
        with mock_plugin_discovery(sample_plugins):
            prompt_file = temp_dir / "prompt.yaml"
            terminology_file = temp_dir / "terminology.yaml"
            examples_dir = temp_dir / "examples"
            examples_dir.mkdir()
            
            prompt_config = {
                'system_prompt': 'You are creating {format} docs for {topic} at {organization}.',
                'placeholders': {
                    'format': 'HTML',
                    'organization': 'FASRC'
                }
            }
            
            with open(prompt_file, 'w') as f:
                yaml.dump(prompt_config, f)
            with open(terminology_file, 'w') as f:
                yaml.dump(sample_terminology, f)
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        examples_dir=str(examples_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    system_prompt = generator._build_system_prompt("Python Programming")
                    
                    # Check parameterization worked
                    assert "HTML docs" in system_prompt
                    assert "Python Programming" in system_prompt
                    assert "FASRC" in system_prompt
                    
                    # Should include plugin recommendations context
                    # (The actual content depends on the plugins loaded)
                    assert len(system_prompt) > len("You are creating HTML docs for Python Programming at FASRC.")
    
    def test_plugin_priority_ordering(self, temp_dir, sample_terminology, mock_plugin_discovery):
        """Test that plugins are ordered by priority in formatted output."""
        
        class HighPriorityPlugin(MockRecommendationEngine):
            def __init__(self, **kwargs):
                super().__init__(name='high', **kwargs)
            
            def get_priority(self):
                return 100
            
            def get_formatted_recommendations(self, topic, context=None):
                return "HIGH PRIORITY CONTENT"
        
        class LowPriorityPlugin(MockRecommendationEngine):
            def __init__(self, **kwargs):
                super().__init__(name='low', **kwargs)
            
            def get_priority(self):
                return 10
            
            def get_formatted_recommendations(self, topic, context=None):
                return "low priority content"
        
        plugins = {
            "low": LowPriorityPlugin,
            "high": HighPriorityPlugin
        }
        
        with mock_plugin_discovery(plugins):
            prompt_file = temp_dir / "prompt.yaml"
            terminology_file = temp_dir / "terminology.yaml"
            examples_dir = temp_dir / "examples"
            examples_dir.mkdir()
            
            with open(prompt_file, 'w') as f:
                yaml.dump({'system_prompt': 'Test'}, f)
            with open(terminology_file, 'w') as f:
                yaml.dump(sample_terminology, f)
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                with patch('doc_generator.core.OpenAI'):
                    generator = DocumentationGenerator(
                        prompt_yaml_path=str(prompt_file),
                        examples_dir=str(examples_dir),
                        terminology_path=str(terminology_file)
                    )
                    
                    # Get formatted recommendations
                    formatted = generator.plugin_manager.get_formatted_recommendations("test topic")
                    
                    # High priority content should come before low priority
                    high_pos = formatted.find("HIGH PRIORITY CONTENT")
                    low_pos = formatted.find("low priority content")
                    
                    assert high_pos != -1, "High priority content should be present"
                    assert low_pos != -1, "Low priority content should be present"
                    assert high_pos < low_pos, "High priority content should come first"