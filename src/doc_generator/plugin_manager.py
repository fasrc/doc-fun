"""
Plugin manager for discovering and loading recommendation engine and analysis plugins.
"""

import logging
from typing import Dict, List, Optional, Any
from importlib.metadata import entry_points
from pathlib import Path
from .plugins.base import RecommendationEngine
from .plugins.analysis_base import AnalysisPlugin


class PluginManager:
    """
    Manages discovery, loading, and execution of recommendation engine and analysis plugins.
    """
    
    def __init__(self, terminology: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the plugin manager.
        
        Args:
            terminology: Terminology configuration to pass to plugins
            logger: Logger instance
        """
        self.terminology = terminology or {}
        self.logger = logger or logging.getLogger(__name__)
        self.engines: Dict[str, RecommendationEngine] = {}
        self.analysis_plugins: Dict[str, AnalysisPlugin] = {}
        
    def load_plugins(self) -> None:
        """
        Discover and load all available recommendation engine plugins.
        
        Looks for plugins registered under the 'doc_generator.plugins' entry point group.
        """
        self.logger.info("Loading recommendation engine plugins...")
        
        try:
            # Discover plugins via entry points
            eps = entry_points(group='doc_generator.plugins')
            
            for entry_point in eps:
                try:
                    self._load_plugin(entry_point)
                except Exception as e:
                    self.logger.warning(f"Failed to load plugin '{entry_point.name}': {e}")
            
            self.logger.info(f"Successfully loaded {len(self.engines)} plugins: {list(self.engines.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error during plugin discovery: {e}")
    
    def _load_plugin(self, entry_point) -> None:
        """
        Load a single plugin from an entry point.
        
        Args:
            entry_point: Entry point object to load
        """
        plugin_name = entry_point.name
        self.logger.debug(f"Loading plugin: {plugin_name}")
        
        # Load the plugin class
        plugin_class = entry_point.load()
        
        # Verify it implements RecommendationEngine interface
        if not issubclass(plugin_class, RecommendationEngine):
            raise TypeError(f"Plugin '{plugin_name}' must inherit from RecommendationEngine")
        
        # Instantiate the plugin
        plugin_instance = plugin_class(
            terminology=self.terminology,
            logger=self.logger
        )
        
        # Verify the plugin reports the correct name
        if plugin_instance.get_name() != plugin_name:
            self.logger.warning(f"Plugin name mismatch: entry point '{plugin_name}' vs plugin name '{plugin_instance.get_name()}'")
        
        # Check if plugin is enabled
        if not plugin_instance.is_enabled():
            self.logger.info(f"Plugin '{plugin_name}' is disabled, skipping")
            return
        
        # Store the plugin
        self.engines[plugin_name] = plugin_instance
        self.logger.info(f"✓ Loaded plugin: {plugin_name}")
    
    def get_engine(self, name: str) -> Optional[RecommendationEngine]:
        """
        Get a specific recommendation engine by name.
        
        Args:
            name: Name of the recommendation engine
            
        Returns:
            RecommendationEngine instance or None if not found
        """
        return self.engines.get(name)
    
    def get_all_engines(self) -> Dict[str, RecommendationEngine]:
        """
        Get all loaded recommendation engines.
        
        Returns:
            Dictionary mapping engine names to engine instances
        """
        return self.engines.copy()
    
    def get_recommendations(self, topic: str, context: Optional[Dict] = None, 
                          engine_names: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Get recommendations from one or more engines.
        
        Args:
            topic: Topic to get recommendations for
            context: Optional context information
            engine_names: List of specific engines to use (None = all engines)
            
        Returns:
            Dictionary mapping engine names to their recommendation lists
        """
        if engine_names is None:
            engine_names = list(self.engines.keys())
        
        results = {}
        
        for engine_name in engine_names:
            if engine_name not in self.engines:
                self.logger.warning(f"Unknown recommendation engine: {engine_name}")
                continue
            
            try:
                engine = self.engines[engine_name]
                recommendations = engine.get_recommendations(topic, context)
                results[engine_name] = recommendations
                self.logger.debug(f"Engine '{engine_name}' returned {len(recommendations)} recommendations")
                
            except Exception as e:
                self.logger.error(f"Error getting recommendations from engine '{engine_name}': {e}")
                results[engine_name] = []
        
        return results
    
    def get_formatted_recommendations(self, topic: str, context: Optional[Dict] = None,
                                    engine_names: Optional[List[str]] = None) -> str:
        """
        Get formatted recommendations from all engines for documentation context.
        
        Args:
            topic: Topic to get recommendations for
            context: Optional context information
            engine_names: List of specific engines to use (None = all engines)
            
        Returns:
            Formatted string with all recommendations
        """
        if engine_names is None:
            # Sort engines by priority (highest first)
            sorted_engines = sorted(
                self.engines.items(),
                key=lambda x: x[1].get_priority(),
                reverse=True
            )
            engine_names = [name for name, _ in sorted_engines]
        
        formatted_parts = []
        
        for engine_name in engine_names:
            if engine_name not in self.engines:
                continue
            
            try:
                engine = self.engines[engine_name]
                formatted = engine.get_formatted_recommendations(topic, context)
                if formatted.strip():
                    formatted_parts.append(formatted)
                    
            except Exception as e:
                self.logger.error(f"Error formatting recommendations from engine '{engine_name}': {e}")
        
        return "\n".join(formatted_parts)
    
    def list_engines(self) -> List[Dict[str, any]]:
        """
        List all loaded engines with their metadata.
        
        Returns:
            List of dictionaries with engine information
        """
        engine_info = []
        
        for name, engine in self.engines.items():
            info = {
                'name': name,
                'class': engine.__class__.__name__,
                'module': engine.__class__.__module__,
                'supported_types': engine.get_supported_types(),
                'priority': engine.get_priority(),
                'enabled': engine.is_enabled()
            }
            engine_info.append(info)
        
        return engine_info
    
    def load_analysis_plugins(self, config: Optional[Dict] = None) -> None:
        """
        Discover and load all available analysis plugins.
        
        Args:
            config: Optional configuration dictionary for analysis plugins
        
        Looks for plugins registered under the 'doc_generator.analysis' entry point group.
        """
        self.logger.info("Loading analysis plugins...")
        
        try:
            # Discover analysis plugins via entry points
            eps = entry_points(group='doc_generator.analysis')
            
            for entry_point in eps:
                try:
                    self._load_analysis_plugin(entry_point, config)
                except Exception as e:
                    self.logger.warning(f"Failed to load analysis plugin '{entry_point.name}': {e}")
            
            self.logger.info(f"Successfully loaded {len(self.analysis_plugins)} analysis plugins: {list(self.analysis_plugins.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error during analysis plugin discovery: {e}")
    
    def _load_analysis_plugin(self, entry_point, config: Optional[Dict] = None) -> None:
        """
        Load a single analysis plugin from an entry point.
        
        Args:
            entry_point: Entry point object to load
            config: Optional configuration for the plugin
        """
        plugin_name = entry_point.name
        self.logger.debug(f"Loading analysis plugin: {plugin_name}")
        
        # Load the plugin class
        plugin_class = entry_point.load()
        
        # Verify it implements AnalysisPlugin interface
        if not issubclass(plugin_class, AnalysisPlugin):
            raise TypeError(f"Analysis plugin '{plugin_name}' must inherit from AnalysisPlugin")
        
        # Get plugin-specific config
        plugin_config = {}
        if config and plugin_name in config:
            plugin_config = config[plugin_name]
        
        # Instantiate the plugin
        plugin_instance = plugin_class(
            logger=self.logger,
            config=plugin_config
        )
        
        # Verify the plugin reports the correct name
        if plugin_instance.get_name() != plugin_name:
            self.logger.warning(f"Plugin name mismatch: entry point '{plugin_name}' vs plugin name '{plugin_instance.get_name()}'")
        
        # Check if plugin is enabled
        if not plugin_instance.is_enabled():
            self.logger.info(f"Analysis plugin '{plugin_name}' is disabled, skipping")
            return
        
        # Store the plugin
        self.analysis_plugins[plugin_name] = plugin_instance
        self.logger.info(f"✓ Loaded analysis plugin: {plugin_name}")
    
    def get_analysis_plugin(self, name: str) -> Optional[AnalysisPlugin]:
        """
        Get a specific analysis plugin by name.
        
        Args:
            name: Name of the analysis plugin
            
        Returns:
            AnalysisPlugin instance or None if not found
        """
        return self.analysis_plugins.get(name)
    
    def get_all_analysis_plugins(self) -> Dict[str, AnalysisPlugin]:
        """
        Get all loaded analysis plugins.
        
        Returns:
            Dictionary mapping plugin names to plugin instances
        """
        return self.analysis_plugins.copy()
    
    def run_analysis_pipeline(self, documents: List[Dict[str, Any]], topic: str, 
                            output_dir: Path, plugin_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run analysis plugins on generated documents.
        
        Args:
            documents: List of document dictionaries with 'path' and 'content'
            topic: The topic used for generation
            output_dir: Directory to save analysis artifacts
            plugin_names: Optional list of specific plugins to run (None = all)
            
        Returns:
            Dictionary mapping plugin names to their results
        """
        if plugin_names is None:
            # Sort plugins by priority (highest first)
            sorted_plugins = sorted(
                self.analysis_plugins.items(),
                key=lambda x: x[1].get_priority(),
                reverse=True
            )
            plugin_names = [name for name, _ in sorted_plugins]
        
        results = {}
        
        for plugin_name in plugin_names:
            if plugin_name not in self.analysis_plugins:
                self.logger.warning(f"Unknown analysis plugin: {plugin_name}")
                continue
            
            try:
                plugin = self.analysis_plugins[plugin_name]
                self.logger.info(f"Running analysis plugin: {plugin_name}")
                
                # Run analysis
                analysis_results = plugin.analyze(documents, topic)
                
                # Save artifacts
                saved_files = plugin.save_artifacts(analysis_results, output_dir, topic)
                
                # Store results
                results[plugin_name] = {
                    'analysis': analysis_results,
                    'artifacts': saved_files
                }
                
                self.logger.info(f"✓ {plugin_name}: Generated {len(saved_files)} artifacts")
                
            except Exception as e:
                self.logger.error(f"Error running analysis plugin '{plugin_name}': {e}")
                results[plugin_name] = {
                    'error': str(e),
                    'artifacts': []
                }
        
        return results
    
    def list_analysis_plugins(self) -> List[Dict[str, Any]]:
        """
        List all loaded analysis plugins with their metadata.
        
        Returns:
            List of dictionaries with plugin information
        """
        plugin_info = []
        
        for name, plugin in self.analysis_plugins.items():
            info = {
                'name': name,
                'class': plugin.__class__.__name__,
                'module': plugin.__class__.__module__,
                'priority': plugin.get_priority(),
                'enabled': plugin.is_enabled(),
                'supported_formats': plugin.get_supported_formats()
            }
            plugin_info.append(info)
        
        return plugin_info