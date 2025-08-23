"""
Custom exception hierarchy following PEP-8 standards.
Provides structured error handling with proper context.
"""

from typing import Optional, Dict, Any


class DocGeneratorError(Exception):
    """
    Base exception for all doc-generator errors.
    Provides context and error codes for better debugging.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize exception with context.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            context: Additional context for debugging
        """
        super().__init__(message)
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            'error': self.error_code,
            'message': str(self),
            'context': self.context
        }


class ConfigurationError(DocGeneratorError):
    """Raised when configuration is invalid or missing."""
    pass


class ProviderError(DocGeneratorError):
    """Base class for provider-related errors."""
    pass


class ProviderNotAvailableError(ProviderError):
    """Raised when requested provider is not available."""
    pass


class ProviderAPIError(ProviderError):
    """Raised when provider API returns an error."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class PluginError(DocGeneratorError):
    """Base class for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when plugin fails to load."""
    pass


class PluginExecutionError(PluginError):
    """Raised when plugin execution fails."""
    pass


class ValidationError(DocGeneratorError):
    """Raised when input validation fails."""
    pass


class FileOperationError(DocGeneratorError):
    """Raised when file operations fail."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if file_path:
            self.context['file_path'] = file_path


class DocumentStandardizerError(DocGeneratorError):
    """Raised when document standardization fails."""
    pass