"""
Caching utilities with TTL and memory management.
Implements both in-memory and disk-based caching.
"""

import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Callable, TypeVar, Tuple
from functools import wraps
from collections import OrderedDict
import threading

T = TypeVar('T')


class CacheManager:
    """
    Thread-safe cache manager with TTL and size limits.
    Uses LRU eviction policy for memory management.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time-to-live for cache entries (seconds)
            cache_dir: Directory for disk-based cache
        """
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._cache_dir = cache_dir
        self._lock = threading.RLock()
        
        # Create cache directory if specified
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from function arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Hashed cache key
        """
        # Create a stable string representation
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        
        # Hash for consistent key length
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if expired/missing
        """
        with self._lock:
            # Check memory cache first
            if key in self._cache:
                value, timestamp = self._cache[key]
                
                # Check if expired
                if time.time() - timestamp < self._ttl:
                    # Move to end (LRU)
                    self._cache.move_to_end(key)
                    return value
                else:
                    # Remove expired entry
                    del self._cache[key]
                    
            # Check disk cache if configured
            if self._cache_dir:
                cache_file = self._cache_dir / f"{key}.cache"
                if cache_file.exists():
                    try:
                        # Check file age
                        file_age = time.time() - cache_file.stat().st_mtime
                        if file_age < self._ttl:
                            with open(cache_file, 'rb') as f:
                                value = pickle.load(f)
                                
                            # Add to memory cache
                            self._cache[key] = (value, time.time())
                            self._evict_if_needed()
                            
                            return value
                        else:
                            # Remove expired file
                            cache_file.unlink()
                            
                    except (pickle.PickleError, IOError) as e:
                        # Log error but don't fail
                        import logging
                        logging.warning(f"Failed to load cache file {cache_file}: {e}")
                        
        return None
        
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Add to memory cache
            self._cache[key] = (value, time.time())
            self._evict_if_needed()
            
            # Write to disk if configured
            if self._cache_dir:
                cache_file = self._cache_dir / f"{key}.cache"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(value, f)
                except (pickle.PickleError, IOError) as e:
                    # Log error but don't fail
                    import logging
                    logging.warning(f"Failed to write cache file {cache_file}: {e}")
                    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size."""
        while len(self._cache) > self._max_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            
            # Remove from disk if exists
            if self._cache_dir:
                cache_file = self._cache_dir / f"{oldest_key}.cache"
                if cache_file.exists():
                    cache_file.unlink()
                    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            
            # Clear disk cache if configured
            if self._cache_dir:
                for cache_file in self._cache_dir.glob("*.cache"):
                    cache_file.unlink()
                    
    def cached(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to cache function results.
        
        Args:
            func: Function to cache
            
        Returns:
            Wrapped function with caching
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            cache_key = self._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_value = self.get(cache_key)
            if cached_value is not None:
                return cached_value
                
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            self.set(cache_key, result)
            
            return result
            
        # Add cache control methods
        wrapper.cache_clear = lambda: self.clear()
        wrapper.cache_info = lambda: {
            'size': len(self._cache),
            'max_size': self._max_size,
            'ttl': self._ttl
        }
        
        return wrapper


# Global cache instance (can be configured via settings)
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get global cache manager instance.
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        from .config import get_settings
        settings = get_settings()
        
        cache_dir = None
        if settings.performance.cache_enabled:
            cache_dir = settings.paths.output_dir / '.cache'
            
        _cache_manager = CacheManager(
            max_size=1000,
            ttl=settings.performance.cache_ttl,
            cache_dir=cache_dir
        )
        
    return _cache_manager


def cached(ttl: Optional[int] = None):
    """
    Decorator factory for caching with optional TTL override.
    
    Args:
        ttl: Optional TTL override for this specific cache
        
    Returns:
        Cache decorator
    """
    cache_manager = get_cache_manager()
    
    if ttl:
        # Create custom cache manager with different TTL
        from .config import get_settings
        settings = get_settings()
        
        cache_dir = None
        if settings.performance.cache_enabled:
            cache_dir = settings.paths.output_dir / '.cache'
            
        custom_cache = CacheManager(
            max_size=1000,
            ttl=ttl,
            cache_dir=cache_dir
        )
        return custom_cache.cached
        
    return cache_manager.cached