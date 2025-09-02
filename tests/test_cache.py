"""
Comprehensive tests for the Cache system with edge cases and failure scenarios.
"""

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import logging

from doc_generator.cache import CacheManager, get_cache_manager, cached


class TestCacheManager:
    """Test CacheManager functionality including edge cases and failure scenarios."""

    def setup_method(self):
        """Setup for each test method."""
        # Create a fresh cache manager for each test
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default_values(self):
        """Test CacheManager initialization with default values."""
        manager = CacheManager()
        
        assert manager._max_size == 1000
        assert manager._ttl == 3600
        assert manager._cache == {}
        assert manager._cache_dir is None
        
    def test_init_custom_values(self):
        """Test CacheManager initialization with custom values."""
        manager = CacheManager(max_size=500, ttl=1800, cache_dir=self.cache_dir)
        
        assert manager._max_size == 500
        assert manager._ttl == 1800
        assert manager._cache_dir == self.cache_dir

    def test_generate_key_basic(self):
        """Test basic key generation."""
        manager = CacheManager()
        
        key = manager._generate_key("test", "arg1", "arg2", kwarg1="value1")
        assert isinstance(key, str)
        assert len(key) > 0
        
        # Same inputs should generate same key
        key2 = manager._generate_key("test", "arg1", "arg2", kwarg1="value1")
        assert key == key2
        
        # Different inputs should generate different keys
        key3 = manager._generate_key("test", "arg1", "arg3", kwarg1="value1")
        assert key != key3

    def test_generate_key_with_unhashable_args(self):
        """Test key generation with unhashable arguments."""
        manager = CacheManager()
        
        # Lists and dicts should be handled
        key = manager._generate_key("test", ["list", "arg"], {"dict": "arg"})
        assert isinstance(key, str)
        
        # Nested structures
        complex_arg = {"nested": {"list": [1, 2, {"deep": "value"}]}}
        key2 = manager._generate_key("test", complex_arg)
        assert isinstance(key2, str)

    def test_set_and_get_basic(self):
        """Test basic set and get operations."""
        manager = CacheManager()
        
        manager.set("test_key", "test_value")
        result = manager.get("test_key")
        
        assert result == "test_value"

    def test_get_nonexistent_key(self):
        """Test getting a non-existent key returns None."""
        manager = CacheManager()
        
        result = manager.get("nonexistent")
        assert result is None

    def test_ttl_expiration(self):
        """Test TTL expiration functionality."""
        manager = CacheManager(ttl=0.1)  # 100ms TTL
        
        manager.set("test_key", "test_value")
        
        # Should be available immediately
        assert manager.get("test_key") == "test_value"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be None after expiration
        assert manager.get("test_key") is None

    def test_max_size_eviction(self):
        """Test eviction when max size is exceeded."""
        manager = CacheManager(max_size=2)
        
        manager.set("key1", "value1")
        manager.set("key2", "value2")
        manager.set("key3", "value3")  # Should trigger eviction
        
        # Should have exactly 2 items
        assert len(manager._cache) == 2
        
        # First item should be evicted
        assert manager.get("key1") is None
        assert manager.get("key2") == "value2"
        assert manager.get("key3") == "value3"

    def test_evict_if_needed_multiple_evictions(self):
        """Test multiple evictions when cache is significantly oversized."""
        manager = CacheManager(max_size=5)
        
        # Add many items to trigger multiple evictions
        for i in range(10):
            manager.set(f"key{i}", f"value{i}")
        
        # Should have exactly max_size items
        assert len(manager._cache) == 5
        
        # Earliest items should be evicted
        assert manager.get("key0") is None
        assert manager.get("key1") is None
        assert manager.get("key5") == "value5"
        assert manager.get("key9") == "value9"

    def test_clear_cache(self):
        """Test clearing the cache."""
        manager = CacheManager()
        
        manager.set("key1", "value1")
        manager.set("key2", "value2")
        
        assert len(manager._cache) == 2
        
        manager.clear()
        
        assert len(manager._cache) == 0
        assert manager.get("key1") is None

    def test_clear_cache_with_directory_errors(self):
        """Test clearing cache with file system errors."""
        manager = CacheManager(cache_dir=self.cache_dir)
        
        # Create some cache files
        cache_file = self.cache_dir / "test_file.cache"
        cache_file.write_text("test content")
        
        # Mock shutil.rmtree to raise an exception
        with patch('shutil.rmtree') as mock_rmtree:
            mock_rmtree.side_effect = OSError("Permission denied")
            
            # Should not raise an exception
            manager.clear()
            
            # Memory cache should still be cleared
            assert len(manager._cache) == 0

    def test_file_cache_persistence(self):
        """Test file cache persistence functionality."""
        cache_file = self.cache_dir / "test.json"
        
        manager = CacheManager(cache_dir=self.cache_dir)
        
        # Test file-based caching (requires implementation details)
        # This tests the persistence layer when cache_dir is set
        manager.set("persistent_key", "persistent_value")
        
        # Verify in-memory cache works
        assert manager.get("persistent_key") == "persistent_value"

    def test_concurrent_access(self):
        """Test thread safety of cache operations."""
        manager = CacheManager()
        results = []
        errors = []
        
        def cache_worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    manager.set(key, value)
                    result = manager.get(key)
                    results.append((key, result == value))
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50  # 5 workers * 10 operations each
        assert all(success for _, success in results)

    def test_cached_decorator_basic(self):
        """Test basic cached decorator functionality."""
        manager = CacheManager()
        call_count = 0
        
        @manager.cached
        def expensive_function(x, y=1):
            nonlocal call_count
            call_count += 1
            return x * y + call_count
        
        # First call
        result1 = expensive_function(5, y=2)
        assert call_count == 1
        assert result1 == 11  # 5*2 + 1
        
        # Second call with same args - should use cache
        result2 = expensive_function(5, y=2)
        assert call_count == 1  # No additional calls
        assert result2 == 11
        
        # Different args - should call function
        result3 = expensive_function(3, y=2)
        assert call_count == 2
        assert result3 == 8  # 3*2 + 2

    def test_cached_decorator_with_ttl_expiration(self):
        """Test cached decorator with TTL expiration."""
        manager = CacheManager(ttl=0.1)  # 100ms TTL for all operations
        call_count = 0
        
        @manager.cached
        def fast_expiring_function(x):
            nonlocal call_count
            call_count += 1
            return x * call_count
        
        # First call
        result1 = fast_expiring_function(5)
        assert call_count == 1
        assert result1 == 5
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should call function again
        result2 = fast_expiring_function(5)
        assert call_count == 2
        assert result2 == 10  # 5 * 2

    def test_cached_decorator_with_exception(self):
        """Test cached decorator behavior when function raises exception."""
        manager = CacheManager()
        call_count = 0
        
        @manager.cached
        def failing_function(should_fail=True):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # First call - should raise exception
        with pytest.raises(ValueError, match="Test error"):
            failing_function(True)
        assert call_count == 1
        
        # Second call with same args - should raise exception again (not cached)
        with pytest.raises(ValueError, match="Test error"):
            failing_function(True)
        assert call_count == 2
        
        # Call with different args that succeeds
        result = failing_function(False)
        assert result == "success"
        assert call_count == 3

    def test_cached_decorator_with_unhashable_args(self):
        """Test cached decorator with unhashable arguments."""
        manager = CacheManager()
        call_count = 0
        
        @manager.cached
        def function_with_list_arg(data):
            nonlocal call_count
            call_count += 1
            return sum(data) + call_count
        
        # Should work with list arguments
        result1 = function_with_list_arg([1, 2, 3])
        assert call_count == 1
        assert result1 == 7  # 1+2+3+1
        
        # Same list should use cache
        result2 = function_with_list_arg([1, 2, 3])
        assert call_count == 1
        assert result2 == 7

    def test_get_cache_manager_singleton(self):
        """Test that get_cache_manager returns singleton instance."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, CacheManager)

    def test_cache_key_collision_resistance(self):
        """Test that cache keys are resistant to collisions."""
        manager = CacheManager()
        
        # These should generate different keys despite similar structure
        key1 = manager._generate_key("func", "a", "b")
        key2 = manager._generate_key("func", "ab")
        key3 = manager._generate_key("func", "a", "", "b")
        
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_with_none_values(self):
        """Test caching None values (should work)."""
        manager = CacheManager()
        
        manager.set("none_key", None)
        result = manager.get("none_key")
        
        assert result is None
        
        # Should distinguish between cached None and missing key
        assert "none_key" in manager._cache

    def test_cache_with_complex_objects(self):
        """Test caching complex objects."""
        manager = CacheManager()
        
        complex_obj = {
            "data": [1, 2, {"nested": "value"}],
            "metadata": {"timestamp": 123456789}
        }
        
        manager.set("complex", complex_obj)
        result = manager.get("complex")
        
        assert result == complex_obj

    def test_cache_memory_efficiency_with_large_objects(self):
        """Test cache behavior with large objects."""
        manager = CacheManager(max_size=3)
        
        # Create some large-ish objects
        large_obj1 = {"data": list(range(1000))}
        large_obj2 = {"data": list(range(1000, 2000))}
        large_obj3 = {"data": list(range(2000, 3000))}
        large_obj4 = {"data": list(range(3000, 4000))}
        
        manager.set("large1", large_obj1)
        manager.set("large2", large_obj2)
        manager.set("large3", large_obj3)
        manager.set("large4", large_obj4)  # Should evict large1
        
        assert manager.get("large1") is None
        assert manager.get("large2") == large_obj2
        assert manager.get("large3") == large_obj3
        assert manager.get("large4") == large_obj4

    def test_cached_decorator_with_class_methods(self):
        """Test cached decorator with class methods."""
        manager = CacheManager()
        
        class TestClass:
            def __init__(self):
                self.call_count = 0
            
            @manager.cached
            def instance_method(self, x):
                self.call_count += 1
                return x * self.call_count
        
        obj = TestClass()
        
        # First call
        result1 = obj.instance_method(5)
        assert obj.call_count == 1
        assert result1 == 5
        
        # Second call - should use cache
        result2 = obj.instance_method(5)
        assert obj.call_count == 1  # No additional calls
        assert result2 == 5

    def test_global_cached_decorator(self):
        """Test the global cached decorator function without TTL."""
        call_count = 0
        
        # Clear global cache manager state first
        import doc_generator.cache
        doc_generator.cache._cache_manager = None
        
        # Mock the settings to avoid configuration issues
        with patch('doc_generator.config.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.performance.cache_enabled = False
            mock_settings.performance.cache_ttl = 3600
            mock_get_settings.return_value = mock_settings
            
            @cached()
            def global_cached_function(x):
                nonlocal call_count
                call_count += 1
                return x * call_count
            
            # Should use global cache manager
            result1 = global_cached_function(10)
            assert call_count == 1
            assert result1 == 10
            
            # Should use cache
            result2 = global_cached_function(10)
            assert call_count == 1
            assert result2 == 10

    def test_global_cached_decorator_with_ttl_override(self):
        """Test the global cached decorator with TTL override."""
        call_count = 0
        
        # Mock the settings for TTL override test
        with patch('doc_generator.config.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.performance.cache_enabled = False
            mock_settings.paths.output_dir = Path("/tmp")
            mock_get_settings.return_value = mock_settings
            
            @cached(ttl=0.1)  # 100ms TTL override
            def ttl_override_function(x):
                nonlocal call_count
                call_count += 1
                return x * call_count
            
            # First call
            result1 = ttl_override_function(5)
            assert call_count == 1
            assert result1 == 5
            
            # Wait for expiration
            time.sleep(0.2)
            
            # Should call function again due to TTL override
            result2 = ttl_override_function(5)
            assert call_count == 2
            assert result2 == 10  # 5 * 2

    def test_cache_entry_timestamp_update_on_access(self):
        """Test that cache entries are updated on access for LRU behavior."""
        manager = CacheManager(max_size=3)
        
        manager.set("key1", "value1")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        manager.set("key2", "value2")
        time.sleep(0.01)
        manager.set("key3", "value3")
        
        # Access key1 to update its timestamp
        manager.get("key1")
        
        # Add new item - should evict key2 (oldest unaccessed)
        manager.set("key4", "value4")
        
        assert manager.get("key1") == "value1"  # Should still exist
        assert manager.get("key2") is None      # Should be evicted
        assert manager.get("key3") == "value3"  # Should still exist
        assert manager.get("key4") == "value4"  # Should exist

    def test_cache_edge_case_zero_ttl(self):
        """Test cache behavior with zero TTL."""
        manager = CacheManager(ttl=0)
        
        manager.set("key", "value")
        
        # With zero TTL, items should expire immediately
        result = manager.get("key")
        assert result is None

    def test_cache_edge_case_zero_max_size(self):
        """Test cache behavior with zero max size."""
        manager = CacheManager(max_size=0)
        
        manager.set("key", "value")
        
        # With zero max size, nothing should be cached
        assert len(manager._cache) == 0
        assert manager.get("key") is None

    def test_cache_stress_test_rapid_operations(self):
        """Stress test with rapid cache operations."""
        manager = CacheManager(max_size=50, ttl=1)
        
        # Rapid set operations
        for i in range(100):
            manager.set(f"key_{i}", f"value_{i}")
        
        # Should respect max_size
        assert len(manager._cache) <= 50
        
        # Recent items should be available
        assert manager.get("key_99") == "value_99"
        assert manager.get("key_95") == "value_95"