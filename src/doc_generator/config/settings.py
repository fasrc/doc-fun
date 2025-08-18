"""
Settings module using pydantic for configuration management.
Handles environment variables, YAML files, and defaults.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
import yaml


class ProviderSettings(BaseSettings):
    """Provider-specific configuration."""
    
    openai_api_key: Optional[str] = Field(None, env='OPENAI_API_KEY')
    anthropic_api_key: Optional[str] = Field(None, env='ANTHROPIC_API_KEY')
    default_provider: str = Field('auto', env='DEFAULT_PROVIDER')
    default_model: Optional[str] = Field(None, env='DEFAULT_MODEL')
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=128000)
    
    @field_validator('default_provider')
    @classmethod
    def validate_provider(cls, v):
        """Validate provider selection."""
        valid_providers = {'openai', 'claude', 'auto'}
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )


class PathSettings(BaseSettings):
    """Path configuration with validation."""
    
    prompts_dir: Path = Field(Path('./prompts'), env='PROMPTS_DIR')
    shots_dir: Path = Field(Path('./shots'), env='SHOTS_DIR')
    output_dir: Path = Field(Path('./output'), env='OUTPUT_DIR')
    terminology_path: Path = Field(Path('./terminology.yaml'), env='TERMINOLOGY_PATH')
    
    @field_validator('prompts_dir', 'shots_dir', 'output_dir', 'terminology_path', mode='before')
    @classmethod
    def resolve_path(cls, v):
        """Resolve and validate paths."""
        if isinstance(v, str):
            v = Path(v)
        if isinstance(v, Path):
            return v.resolve()
        return v
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )


class PerformanceSettings(BaseSettings):
    """Performance tuning configuration."""
    
    cache_enabled: bool = Field(True, env='CACHE_ENABLED')
    cache_ttl: int = Field(3600, env='CACHE_TTL')  # seconds
    max_workers: int = Field(4, env='MAX_WORKERS')
    request_timeout: int = Field(30, env='REQUEST_TIMEOUT')
    retry_max_attempts: int = Field(3, env='RETRY_MAX_ATTEMPTS')
    retry_backoff_factor: float = Field(2.0, env='RETRY_BACKOFF_FACTOR')
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )


class Settings(BaseSettings):
    """
    Main settings class combining all configuration sections.
    Follows PEP-8 naming conventions and provides comprehensive validation.
    """
    
    # Application metadata
    app_name: str = Field('doc-generator', env='APP_NAME')
    version: str = Field('2.5.0', env='APP_VERSION')
    debug: bool = Field(False, env='DEBUG')
    log_level: str = Field('INFO', env='LOG_LEVEL')
    
    # Nested configuration sections
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    # Plugin configuration
    enabled_plugins: List[str] = Field(
        default_factory=lambda: ['modules', 'compiler', 'reporter', 'link_validator'],
        env='ENABLED_PLUGINS'
    )
    
    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False,
        'extra': 'ignore'  # Ignore extra fields from environment
    }
        
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Settings':
        """
        Load settings from YAML file with error handling.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Settings instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # Merge with environment variables
            return cls(**data)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def validate_runtime(self) -> Dict[str, Any]:
        """
        Perform runtime validation checks.
        
        Returns:
            Dictionary of validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check API keys
        if not self.providers.openai_api_key and not self.providers.anthropic_api_key:
            results['warnings'].append(
                "No API keys configured. At least one provider key required."
            )
            
        # Check paths exist
        for path_name, path_value in [
            ('prompts_dir', self.paths.prompts_dir),
            ('shots_dir', self.paths.shots_dir),
        ]:
            if not path_value.exists():
                results['warnings'].append(
                    f"{path_name} does not exist: {path_value}"
                )
                
        # Create output directory if missing
        if not self.paths.output_dir.exists():
            try:
                self.paths.output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                results['errors'].append(
                    f"Cannot create output directory: {e}"
                )
                results['valid'] = False
                
        return results


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance (singleton pattern).
    
    Returns:
        Cached Settings instance
    """
    settings = Settings()
    
    # Perform runtime validation
    validation = settings.validate_runtime()
    if not validation['valid']:
        import logging
        logger = logging.getLogger(__name__)
        for error in validation['errors']:
            logger.error(error)
        for warning in validation['warnings']:
            logger.warning(warning)
            
    return settings