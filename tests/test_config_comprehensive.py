"""
Comprehensive tests for the config module including validators and settings.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from doc_generator.config.validators import ConfigValidator
from doc_generator.config.settings import (
    ProviderSettings, PathSettings, PerformanceSettings, Settings, get_settings
)


class TestConfigValidator:
    """Comprehensive tests for ConfigValidator class."""

    def test_validate_path_exists_valid_path(self):
        """Test path validation with existing path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            assert ConfigValidator.validate_path_exists(path) is True

    def test_validate_path_exists_invalid_path(self):
        """Test path validation with non-existing path."""
        path = Path("/nonexistent/path/that/does/not/exist")
        assert ConfigValidator.validate_path_exists(path) is False

    def test_validate_path_exists_file(self):
        """Test path validation with existing file."""
        with tempfile.NamedTemporaryFile() as temp_file:
            path = Path(temp_file.name)
            assert ConfigValidator.validate_path_exists(path) is True

    def test_validate_log_level_valid_levels(self):
        """Test log level validation with valid levels."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for level in valid_levels:
            assert ConfigValidator.validate_log_level(level) is True

    def test_validate_log_level_case_insensitive(self):
        """Test log level validation is case insensitive."""
        levels = ['debug', 'Info', 'WARNING', 'error', 'Critical']
        for level in levels:
            assert ConfigValidator.validate_log_level(level) is True

    def test_validate_log_level_invalid_levels(self):
        """Test log level validation with invalid levels."""
        invalid_levels = ['TRACE', 'VERBOSE', 'NOTICE', 'ALERT', '', 'invalid']
        for level in invalid_levels:
            assert ConfigValidator.validate_log_level(level) is False

    def test_validate_model_name_valid_names(self):
        """Test model name validation with valid names."""
        valid_names = [
            'gpt-4',
            'gpt-3.5-turbo',
            'claude-3-sonnet',
            'text-davinci-003',
            'gpt4o_mini',
            'model.name',
            'model_name_123',
            'simple'
        ]
        for name in valid_names:
            assert ConfigValidator.validate_model_name(name) is True

    def test_validate_model_name_invalid_names(self):
        """Test model name validation with invalid names."""
        invalid_names = [
            'model with spaces',
            'model/with/slashes',
            'model@with@symbols',
            'model+plus',
            'model=equals',
            '',
            'model\nwith\nnewlines'
        ]
        for name in invalid_names:
            assert ConfigValidator.validate_model_name(name) is False

    def test_validate_api_key_valid_keys(self):
        """Test API key validation with valid keys."""
        valid_keys = [
            'sk-1234567890abcdefghij',
            'anthropic_api_key_12345',
            'a' * 50,
            '1234567890',
            'valid-api-key-123'
        ]
        for key in valid_keys:
            assert ConfigValidator.validate_api_key(key) is True

    def test_validate_api_key_invalid_keys(self):
        """Test API key validation with invalid keys."""
        invalid_keys = [
            '',
            '   ',
            'short',
            '123456789',  # 9 chars, need at least 10
            None
        ]
        for key in invalid_keys:
            assert ConfigValidator.validate_api_key(key) is False

    def test_validate_api_key_whitespace_handling(self):
        """Test API key validation handles whitespace correctly."""
        # Key with leading/trailing whitespace should be valid if long enough
        key_with_whitespace = '  valid_api_key_123  '
        assert ConfigValidator.validate_api_key(key_with_whitespace) is True

    def test_validate_temperature_valid_values(self):
        """Test temperature validation with valid values."""
        valid_temps = [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0]
        for temp in valid_temps:
            assert ConfigValidator.validate_temperature(temp) is True

    def test_validate_temperature_invalid_values(self):
        """Test temperature validation with invalid values."""
        invalid_temps = [-0.1, -1.0, 2.1, 3.0, 10.0]
        for temp in invalid_temps:
            assert ConfigValidator.validate_temperature(temp) is False

    def test_validate_temperature_edge_cases(self):
        """Test temperature validation edge cases."""
        # Exactly at boundaries
        assert ConfigValidator.validate_temperature(0.0) is True
        assert ConfigValidator.validate_temperature(2.0) is True
        
        # Just outside boundaries
        assert ConfigValidator.validate_temperature(-0.000001) is False
        assert ConfigValidator.validate_temperature(2.000001) is False

    def test_validate_plugin_list_valid_list(self):
        """Test plugin list validation with valid list."""
        valid_plugins = ['modules', 'compiler', 'reporter']
        result = ConfigValidator.validate_plugin_list(valid_plugins)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_plugin_list_empty_list(self):
        """Test plugin list validation with empty list."""
        result = ConfigValidator.validate_plugin_list([])
        
        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_plugin_list_all_valid_plugins(self):
        """Test plugin list validation with all known valid plugins."""
        all_valid = ['modules', 'compiler', 'reporter', 'link_validator']
        result = ConfigValidator.validate_plugin_list(all_valid)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert len(result['warnings']) == 0

    def test_validate_plugin_list_unknown_plugins(self):
        """Test plugin list validation with unknown plugins."""
        plugins_with_unknown = ['modules', 'unknown_plugin', 'another_unknown']
        result = ConfigValidator.validate_plugin_list(plugins_with_unknown)
        
        assert result['valid'] is True  # Still valid, just warnings
        assert len(result['errors']) == 0
        assert len(result['warnings']) == 2
        assert 'Unknown plugin: unknown_plugin' in result['warnings']
        assert 'Unknown plugin: another_unknown' in result['warnings']

    def test_validate_plugin_list_non_string_items(self):
        """Test plugin list validation with non-string items."""
        plugins_with_non_strings = ['modules', 123, {'plugin': 'name'}]
        result = ConfigValidator.validate_plugin_list(plugins_with_non_strings)
        
        assert result['valid'] is True  # Still valid, just warnings
        assert len(result['errors']) == 0
        assert len(result['warnings']) == 2
        assert 'Plugin name must be string: 123' in result['warnings']

    def test_validate_plugin_list_not_a_list(self):
        """Test plugin list validation with non-list input."""
        non_list_inputs = [
            'not a list',
            {'plugins': ['modules']},
            123,
            None
        ]
        
        for invalid_input in non_list_inputs:
            result = ConfigValidator.validate_plugin_list(invalid_input)
            assert result['valid'] is False
            assert 'Plugins must be a list' in result['errors']

    def test_validate_plugin_list_mixed_valid_invalid(self):
        """Test plugin list validation with mixed valid and invalid items."""
        mixed_plugins = ['modules', 'invalid_plugin', 123, 'compiler']
        result = ConfigValidator.validate_plugin_list(mixed_plugins)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert len(result['warnings']) == 2


class TestProviderSettings:
    """Test ProviderSettings pydantic model."""

    def test_provider_settings_defaults(self):
        """Test ProviderSettings with default values."""
        settings = ProviderSettings()
        
        assert settings.default_provider == 'auto'
        assert settings.default_model is None
        assert settings.temperature == 0.3
        assert settings.max_tokens == 2000
        # API keys may be set from environment, so just check they're strings or None
        assert settings.openai_api_key is None or isinstance(settings.openai_api_key, str)
        assert settings.anthropic_api_key is None or isinstance(settings.anthropic_api_key, str)

    def test_provider_settings_custom_values(self):
        """Test ProviderSettings with custom values."""
        custom_settings = ProviderSettings(
            default_provider='openai',
            default_model='gpt-4',
            temperature=0.7,
            max_tokens=4000,
            openai_api_key='test-key'
        )
        
        assert custom_settings.default_provider == 'openai'
        assert custom_settings.default_model == 'gpt-4'
        assert custom_settings.temperature == 0.7
        assert custom_settings.max_tokens == 4000
        assert custom_settings.openai_api_key == 'test-key'

    def test_provider_settings_validation(self):
        """Test ProviderSettings validation."""
        # Valid temperature range
        valid_settings = ProviderSettings(temperature=0.5)
        assert valid_settings.temperature == 0.5
        
        # Test edge cases that should be valid
        edge_settings1 = ProviderSettings(temperature=0.0)
        assert edge_settings1.temperature == 0.0
        
        edge_settings2 = ProviderSettings(temperature=2.0)
        assert edge_settings2.temperature == 2.0


class TestPathSettings:
    """Test PathSettings pydantic model."""

    def test_path_settings_defaults(self):
        """Test PathSettings with default values."""
        settings = PathSettings()
        
        assert settings.output_dir.name == 'output'
        assert settings.shots_dir.name == 'shots'
        assert settings.prompts_dir.name == 'prompts'
        assert settings.terminology_path.name == 'terminology.yaml'

    def test_path_settings_custom_values(self):
        """Test PathSettings with custom values."""
        custom_settings = PathSettings(
            output_dir='/custom/output',
            shots_dir='/custom/shots',
            prompts_dir='/custom/prompts',
            terminology_path='/custom/terminology.yaml'
        )
        
        assert str(custom_settings.output_dir).endswith('custom/output')
        assert str(custom_settings.shots_dir).endswith('custom/shots')
        assert str(custom_settings.prompts_dir).endswith('custom/prompts')
        assert str(custom_settings.terminology_path).endswith('custom/terminology.yaml')

    def test_path_settings_string_conversion(self):
        """Test that PathSettings converts strings to Path objects."""
        settings = PathSettings(output_dir='./test/output')
        assert isinstance(settings.output_dir, Path)
        assert 'test/output' in str(settings.output_dir)


class TestPerformanceSettings:
    """Test PerformanceSettings pydantic model."""

    def test_performance_settings_defaults(self):
        """Test PerformanceSettings with default values."""
        settings = PerformanceSettings()
        
        assert settings.cache_enabled is True
        assert settings.cache_ttl == 3600
        assert settings.max_workers == 4
        assert settings.request_timeout == 30
        assert settings.retry_max_attempts == 3
        assert settings.retry_backoff_factor == 2.0

    def test_performance_settings_custom_values(self):
        """Test PerformanceSettings with custom values."""
        custom_settings = PerformanceSettings(
            cache_enabled=False,
            cache_ttl=7200,
            max_workers=8,
            request_timeout=60,
            retry_max_attempts=5
        )
        
        assert custom_settings.cache_enabled is False
        assert custom_settings.cache_ttl == 7200
        assert custom_settings.max_workers == 8
        assert custom_settings.request_timeout == 60
        assert custom_settings.retry_max_attempts == 5


class TestSettings:
    """Test main Settings pydantic model."""

    def test_settings_defaults(self):
        """Test Settings with default values."""
        settings = Settings()
        
        assert isinstance(settings.providers, ProviderSettings)
        assert isinstance(settings.paths, PathSettings)
        assert isinstance(settings.performance, PerformanceSettings)
        assert settings.log_level == 'INFO'
        assert settings.debug is False
        assert settings.app_name == 'doc-generator'
        assert settings.version == '2.5.0'

    def test_settings_custom_values(self):
        """Test Settings with custom nested values."""
        custom_settings = Settings(
            log_level='DEBUG',
            debug=True,
            app_name='custom-app'
        )
        
        assert custom_settings.log_level == 'DEBUG'
        assert custom_settings.debug is True
        assert custom_settings.app_name == 'custom-app'
        assert isinstance(custom_settings.providers, ProviderSettings)
        assert isinstance(custom_settings.paths, PathSettings)
        assert isinstance(custom_settings.performance, PerformanceSettings)

    def test_settings_log_level_validation(self):
        """Test Settings log level validation."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in valid_levels:
            settings = Settings(log_level=level)
            assert settings.log_level == level

    def test_settings_nested_model_creation(self):
        """Test that Settings properly creates nested model instances."""
        settings = Settings()
        
        # Check that nested models are proper instances
        assert hasattr(settings.providers, 'default_provider')
        assert hasattr(settings.paths, 'output_dir')
        assert hasattr(settings.performance, 'cache_enabled')
        
        # Verify they work as expected
        assert settings.providers.temperature == 0.3
        assert isinstance(settings.paths.output_dir, Path)
        assert settings.performance.cache_ttl == 3600
        assert len(settings.enabled_plugins) > 0


class TestGetSettings:
    """Test get_settings function."""

    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_singleton_behavior(self):
        """Test that get_settings returns the same instance (singleton pattern)."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_settings_with_environment_variables(self):
        """Test get_settings respects environment variables."""
        with patch.dict(os.environ, {
            'LOG_LEVEL': 'DEBUG',
            'DEBUG': 'true',
            'CACHE_ENABLED': 'false'
        }):
            # Clear cache
            get_settings.cache_clear()
            
            settings = get_settings()
            assert settings.log_level == 'DEBUG'
            assert settings.debug is True
            assert settings.performance.cache_enabled is False

    def test_get_settings_environment_variable_types(self):
        """Test that get_settings properly converts environment variable types."""
        with patch.dict(os.environ, {
            'MAX_WORKERS': '8',
            'TEMPERATURE': '0.9',
            'CACHE_TTL': '7200'
        }):
            # Clear cache
            get_settings.cache_clear()
                
            settings = get_settings()
            assert settings.performance.max_workers == 8
            assert settings.providers.temperature == 0.9
            assert settings.performance.cache_ttl == 7200

    def test_get_settings_invalid_environment_values(self):
        """Test get_settings handles invalid environment values gracefully."""
        with patch.dict(os.environ, {
            'LOG_LEVEL': 'INVALID_LEVEL',
            'TEMPERATURE': 'not_a_number',
            'MAX_WORKERS': 'invalid'
        }):
            # Clear cache
            get_settings.cache_clear()
            
            # Should not raise an error, should fall back to defaults or handle gracefully
            try:
                settings = get_settings()
                # If it doesn't raise an error, verify it has reasonable defaults
                assert isinstance(settings, Settings)
            except Exception as e:
                # If it does raise an error, make sure it's a validation error
                assert 'validation' in str(e).lower() or 'invalid' in str(e).lower()

    def test_get_settings_path_environment_variables(self):
        """Test get_settings with path environment variables."""
        with patch.dict(os.environ, {
            'OUTPUT_DIR': '/tmp/test_output',
            'SHOTS_DIR': '/tmp/test_shots'
        }):
            # Clear cache
            get_settings.cache_clear()
                
            settings = get_settings()
            assert str(settings.paths.output_dir).endswith('/tmp/test_output')
            assert str(settings.paths.shots_dir).endswith('/tmp/test_shots')


class TestConfigIntegration:
    """Integration tests for config validation with settings."""

    def test_settings_with_validator_methods(self):
        """Test that settings work with validator methods."""
        settings = get_settings()
        
        # Test log level validation
        assert ConfigValidator.validate_log_level(settings.log_level) is True
        
        # Test temperature validation
        assert ConfigValidator.validate_temperature(settings.providers.temperature) is True
        
        # Test path validation (some paths may not exist, but method should work)
        path_exists = ConfigValidator.validate_path_exists(settings.paths.output_dir)
        assert isinstance(path_exists, bool)

    def test_settings_validation_with_custom_values(self):
        """Test settings validation with custom values."""
        # Create settings with values that should pass validation
        custom_settings = Settings(
            provider={
                'default_model': 'gpt-4',
                'temperature': 0.7
            },
            log_level='WARNING'
        )
        
        # Create custom settings with proper structure
        custom_settings = Settings(
            log_level='WARNING'
        )
        
        # Validate using ConfigValidator methods
        assert ConfigValidator.validate_temperature(custom_settings.providers.temperature) is True
        assert ConfigValidator.validate_log_level(custom_settings.log_level) is True

    def test_plugin_validation_integration(self):
        """Test plugin validation integration."""
        # Test with typical plugin configuration
        plugins = ['modules', 'compiler', 'reporter']
        result = ConfigValidator.validate_plugin_list(plugins)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
        # This could be used in actual settings validation
        assert all(isinstance(plugin, str) for plugin in plugins)

    def test_comprehensive_settings_validation(self):
        """Test comprehensive validation of settings object."""
        settings = Settings(
            log_level='INFO',
            debug=False
        )
        
        # Validate all configurable values
        validations = {
            'log_level': ConfigValidator.validate_log_level(settings.log_level),
            'temperature': ConfigValidator.validate_temperature(settings.providers.temperature),
        }
        
        # All validations should pass
        assert all(validations.values()), f"Failed validations: {validations}"

    def test_edge_case_configurations(self):
        """Test edge case configurations."""
        # Test minimum/maximum values
        edge_settings = Settings()
        
        assert ConfigValidator.validate_temperature(edge_settings.providers.temperature) is True
        assert edge_settings.performance.retry_max_attempts >= 1
        assert edge_settings.performance.request_timeout >= 1

        # Test maximum values  
        max_settings = Settings()
        
        assert ConfigValidator.validate_temperature(max_settings.providers.temperature) is True