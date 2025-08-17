"""
Custom validation logic for configuration settings.
"""

from pathlib import Path
from typing import Any, Dict
import re


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_path_exists(path: Path) -> bool:
        """
        Validate that a path exists.
        
        Args:
            path: Path to validate
            
        Returns:
            True if path exists, False otherwise
        """
        return path.exists()
    
    @staticmethod
    def validate_log_level(level: str) -> bool:
        """
        Validate log level string.
        
        Args:
            level: Log level to validate
            
        Returns:
            True if valid log level, False otherwise
        """
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        return level.upper() in valid_levels
    
    @staticmethod
    def validate_model_name(model: str) -> bool:
        """
        Validate model name format.
        
        Args:
            model: Model name to validate
            
        Returns:
            True if valid model name format, False otherwise
        """
        # Basic validation - alphanumeric, hyphens, dots, underscores
        pattern = r'^[a-zA-Z0-9._-]+$'
        return bool(re.match(pattern, model))
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Basic API key format validation.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid format, False otherwise
        """
        if not api_key:
            return False
        
        # Basic checks - non-empty, reasonable length
        return len(api_key.strip()) >= 10
    
    @staticmethod
    def validate_temperature(temp: float) -> bool:
        """
        Validate temperature parameter.
        
        Args:
            temp: Temperature value to validate
            
        Returns:
            True if valid temperature, False otherwise
        """
        return 0.0 <= temp <= 2.0
    
    @staticmethod
    def validate_plugin_list(plugins: list) -> Dict[str, Any]:
        """
        Validate plugin list configuration.
        
        Args:
            plugins: List of plugin names to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        if not isinstance(plugins, list):
            results['valid'] = False
            results['errors'].append("Plugins must be a list")
            return results
        
        # Check for valid plugin names
        valid_plugins = {
            'modules', 'compiler', 'reporter', 'link_validator'
        }
        
        for plugin in plugins:
            if not isinstance(plugin, str):
                results['warnings'].append(f"Plugin name must be string: {plugin}")
            elif plugin not in valid_plugins:
                results['warnings'].append(f"Unknown plugin: {plugin}")
        
        return results