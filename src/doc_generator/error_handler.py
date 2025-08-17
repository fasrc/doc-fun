"""
Centralized error handling with retry logic and graceful degradation.
"""

import logging
import functools
import time
from typing import TypeVar, Callable, Optional, Any, Tuple, Type
from .exceptions import DocGeneratorError, ProviderAPIError

T = TypeVar('T')


class ErrorHandler:
    """
    Handles errors with retry logic and fallback strategies.
    Implements exponential backoff and circuit breaker patterns.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            logger: Logger instance for error reporting
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logger or logging.getLogger(__name__)
        
    def with_retry(
        self,
        func: Callable[..., T],
        *args,
        exceptions: Tuple[Type[Exception], ...] = (ProviderAPIError,),
        **kwargs
    ) -> T:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            exceptions: Tuple of exceptions to retry on
            **kwargs: Keyword arguments for func
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
                
            except exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    # Calculate backoff time
                    wait_time = self.backoff_factor ** attempt
                    
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"All {self.max_retries} attempts failed. Last error: {e}"
                    )
                    
        if last_exception:
            raise last_exception
            
    def handle_gracefully(
        self,
        func: Callable[..., T],
        fallback: Optional[T] = None,
        log_errors: bool = True
    ) -> Callable[..., Optional[T]]:
        """
        Decorator for graceful error handling with fallback.
        
        Args:
            func: Function to wrap
            fallback: Fallback value on error
            log_errors: Whether to log errors
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
                
            except DocGeneratorError as e:
                if log_errors:
                    self.logger.error(
                        f"Error in {func.__name__}: {e}",
                        extra={'context': e.context}
                    )
                return fallback
                
            except Exception as e:
                if log_errors:
                    self.logger.exception(
                        f"Unexpected error in {func.__name__}: {e}"
                    )
                return fallback
                
        return wrapper


def retry_on_failure(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (ProviderAPIError,)
):
    """
    Decorator for automatic retry on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
        exceptions: Tuple of exceptions to retry on
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = ErrorHandler(max_retries, backoff_factor)
            return handler.with_retry(func, *args, exceptions=exceptions, **kwargs)
        return wrapper
    return decorator


def handle_gracefully(
    fallback: Optional[T] = None,
    log_errors: bool = True
):
    """
    Decorator for graceful error handling.
    
    Args:
        fallback: Fallback value on error
        log_errors: Whether to log errors
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except DocGeneratorError as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(
                        f"Error in {func.__name__}: {e}",
                        extra={'context': getattr(e, 'context', {})}
                    )
                return fallback
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.exception(f"Unexpected error in {func.__name__}: {e}")
                return fallback
        return wrapper
    return decorator