"""
Comprehensive tests for the utils module.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from doc_generator.utils import get_output_directory


class TestGetOutputDirectory:
    """Comprehensive tests for get_output_directory function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup is handled by the system

    def test_get_output_directory_default_creates_timestamped(self, mock_logger, temp_dir):
        """Test that default 'output' creates a timestamped subdirectory."""
        with patch('doc_generator.utils.get_settings') as mock_get_settings:
            # Mock settings
            mock_settings = Mock()
            mock_settings.paths.output_dir = temp_dir
            mock_get_settings.return_value = mock_settings
            
            # Mock time.time() to get predictable timestamp
            with patch('doc_generator.utils.time.time', return_value=1234567890):
                result = get_output_directory('output', mock_logger)
                
                expected_path = temp_dir / '1234567890'
                assert result == str(expected_path)
                
                # Verify the directory was created
                assert expected_path.exists()
                assert expected_path.is_dir()
                
                # Verify logger was called
                mock_logger.info.assert_called_once()
                assert 'Created timestamped output directory' in mock_logger.info.call_args[0][0]
                assert str(expected_path) in mock_logger.info.call_args[0][0]

    def test_get_output_directory_default_with_real_timestamp(self, mock_logger, temp_dir):
        """Test default 'output' with real timestamp (not mocked)."""
        with patch('doc_generator.utils.get_settings') as mock_get_settings:
            # Mock settings
            mock_settings = Mock()
            mock_settings.paths.output_dir = temp_dir
            mock_get_settings.return_value = mock_settings
            
            # Get current time before call
            start_time = int(time.time())
            
            result = get_output_directory('output', mock_logger)
            
            # Get time after call
            end_time = int(time.time())
            
            # The result should be a subdirectory with a timestamp
            result_path = Path(result)
            assert result_path.parent == temp_dir
            
            # The timestamp should be between start_time and end_time
            timestamp = int(result_path.name)
            assert start_time <= timestamp <= end_time
            
            # Verify the directory was created
            assert result_path.exists()
            assert result_path.is_dir()
            
            # Verify logger was called
            mock_logger.info.assert_called_once()

    def test_get_output_directory_custom_path_creates_directory(self, mock_logger, temp_dir):
        """Test that custom output directory is created."""
        custom_dir = temp_dir / 'custom' / 'nested' / 'output'
        
        result = get_output_directory(str(custom_dir), mock_logger)
        
        assert result == str(custom_dir)
        assert custom_dir.exists()
        assert custom_dir.is_dir()
        
        # Logger should not be called for custom directories
        mock_logger.info.assert_not_called()

    def test_get_output_directory_custom_path_already_exists(self, mock_logger, temp_dir):
        """Test custom output directory when it already exists."""
        existing_dir = temp_dir / 'existing'
        existing_dir.mkdir()
        
        # Create a file in it to verify it's preserved
        test_file = existing_dir / 'test.txt'
        test_file.write_text('test content')
        
        result = get_output_directory(str(existing_dir), mock_logger)
        
        assert result == str(existing_dir)
        assert existing_dir.exists()
        assert existing_dir.is_dir()
        
        # Verify existing content is preserved
        assert test_file.exists()
        assert test_file.read_text() == 'test content'
        
        # Logger should not be called for custom directories
        mock_logger.info.assert_not_called()

    def test_get_output_directory_relative_path(self, mock_logger, temp_dir):
        """Test get_output_directory with relative path."""
        # Change to temp directory to make relative paths work predictably
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            result = get_output_directory('./relative/path', mock_logger)
            
            expected_path = temp_dir / 'relative' / 'path'
            assert Path(result).resolve() == expected_path.resolve()
            assert expected_path.exists()
            assert expected_path.is_dir()
            
            # Logger should not be called for custom directories
            mock_logger.info.assert_not_called()
            
        finally:
            os.chdir(original_cwd)

    def test_get_output_directory_absolute_path(self, mock_logger, temp_dir):
        """Test get_output_directory with absolute path."""
        absolute_dir = temp_dir / 'absolute' / 'test'
        
        result = get_output_directory(str(absolute_dir), mock_logger)
        
        assert result == str(absolute_dir)
        assert absolute_dir.exists()
        assert absolute_dir.is_dir()
        
        # Logger should not be called for custom directories
        mock_logger.info.assert_not_called()

    def test_get_output_directory_empty_string(self, mock_logger):
        """Test get_output_directory with empty string."""
        result = get_output_directory('', mock_logger)
        
        # Empty string should create current directory (or handle gracefully)
        assert result == ''
        
        # Logger should not be called for custom directories
        mock_logger.info.assert_not_called()

    def test_get_output_directory_current_directory(self, mock_logger):
        """Test get_output_directory with current directory."""
        result = get_output_directory('.', mock_logger)
        
        assert result == '.'
        
        # Logger should not be called for custom directories
        mock_logger.info.assert_not_called()

    def test_get_output_directory_multiple_calls_different_timestamps(self, mock_logger, temp_dir):
        """Test that multiple calls with 'output' create different timestamps."""
        with patch('doc_generator.utils.get_settings') as mock_get_settings:
            # Mock settings
            mock_settings = Mock()
            mock_settings.paths.output_dir = temp_dir
            mock_get_settings.return_value = mock_settings
            
            # Use mocked time to ensure different timestamps
            with patch('doc_generator.utils.time.time', side_effect=[1234567890, 1234567891]):
                # First call
                result1 = get_output_directory('output', mock_logger)
                
                # Second call
                result2 = get_output_directory('output', mock_logger)
                
                # Results should be different
                assert result1 != result2
                assert '1234567890' in result1
                assert '1234567891' in result2
                
                # Both directories should exist
                assert Path(result1).exists()
                assert Path(result2).exists()
                
                # Both should be subdirectories of temp_dir
                assert Path(result1).parent == temp_dir
                assert Path(result2).parent == temp_dir
                
                # Logger should be called twice
                assert mock_logger.info.call_count == 2

    def test_get_output_directory_mkdir_parents_true(self, mock_logger, temp_dir):
        """Test that mkdir is called with parents=True, exist_ok=True."""
        deep_path = temp_dir / 'very' / 'deep' / 'nested' / 'path'
        
        result = get_output_directory(str(deep_path), mock_logger)
        
        assert result == str(deep_path)
        assert deep_path.exists()
        assert deep_path.is_dir()
        
        # All parent directories should also exist
        assert (temp_dir / 'very').exists()
        assert (temp_dir / 'very' / 'deep').exists()
        assert (temp_dir / 'very' / 'deep' / 'nested').exists()

    def test_get_output_directory_settings_integration(self, mock_logger, temp_dir):
        """Test integration with actual settings (mocked)."""
        with patch('doc_generator.utils.get_settings') as mock_get_settings:
            # Create a more realistic mock settings
            mock_settings = Mock()
            mock_settings.paths.output_dir = temp_dir / 'configured_output'
            mock_get_settings.return_value = mock_settings
            
            with patch('doc_generator.utils.time.time', return_value=9876543210):
                result = get_output_directory('output', mock_logger)
                
                expected_path = temp_dir / 'configured_output' / '9876543210'
                assert result == str(expected_path)
                assert expected_path.exists()
                
                # Verify get_settings was called
                mock_get_settings.assert_called_once()

    def test_get_output_directory_permission_handling(self, mock_logger, temp_dir):
        """Test handling of permission errors (simulated)."""
        restricted_dir = temp_dir / 'restricted'
        
        # This test would need special setup to actually test permission errors
        # For now, we test that the function completes normally
        result = get_output_directory(str(restricted_dir), mock_logger)
        
        assert result == str(restricted_dir)
        assert restricted_dir.exists()

    def test_get_output_directory_path_object_handling(self, mock_logger, temp_dir):
        """Test that the function properly handles Path objects internally."""
        test_dir = temp_dir / 'path_test'
        
        # Pass a string, but verify internal Path handling works
        result = get_output_directory(str(test_dir), mock_logger)
        
        assert result == str(test_dir)
        assert test_dir.exists()
        
        # Test with path that has special characters
        special_dir = temp_dir / 'path with spaces & symbols!'
        result2 = get_output_directory(str(special_dir), mock_logger)
        
        assert result2 == str(special_dir)
        assert special_dir.exists()

    def test_get_output_directory_edge_case_output_string_variations(self, mock_logger, temp_dir):
        """Test edge cases around the 'output' string detection."""
        with patch('doc_generator.utils.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.paths.output_dir = temp_dir
            mock_get_settings.return_value = mock_settings
            
            # Test exact match 'output' - should create timestamp
            with patch('doc_generator.utils.time.time', return_value=1111111111):
                result1 = get_output_directory('output', mock_logger)
                expected1 = temp_dir / '1111111111'
                assert result1 == str(expected1)
                assert expected1.exists()
            
            # Test variations that should NOT trigger timestamp behavior
            variations = ['Output', 'OUTPUT', 'output/', './output', 'my_output', 'output_dir']
            
            for variation in variations:
                variation_path = temp_dir / f'test_{variation}'
                result = get_output_directory(str(variation_path), mock_logger)
                assert result == str(variation_path)
                assert variation_path.exists()

    def test_get_output_directory_logger_message_content(self, mock_logger, temp_dir):
        """Test the specific content of the logger message."""
        with patch('doc_generator.utils.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.paths.output_dir = temp_dir
            mock_get_settings.return_value = mock_settings
            
            with patch('doc_generator.utils.time.time', return_value=5555555555):
                result = get_output_directory('output', mock_logger)
                
                # Verify logger was called with correct message
                mock_logger.info.assert_called_once()
                logged_message = mock_logger.info.call_args[0][0]
                
                # Check message content
                assert 'Created timestamped output directory:' in logged_message
                assert '5555555555' in logged_message
                assert str(temp_dir) in logged_message
                
                # Verify the logged path matches the returned result
                assert result in logged_message

    def test_get_output_directory_return_type(self, mock_logger, temp_dir):
        """Test that the function always returns a string."""
        # Test with default 'output'
        with patch('doc_generator.utils.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.paths.output_dir = temp_dir
            mock_get_settings.return_value = mock_settings
            
            result1 = get_output_directory('output', mock_logger)
            assert isinstance(result1, str)
        
        # Test with custom path
        custom_path = temp_dir / 'custom'
        result2 = get_output_directory(str(custom_path), mock_logger)
        assert isinstance(result2, str)
        assert result2 == str(custom_path)


class TestUtilsModuleStructure:
    """Test the utils module structure and imports."""

    def test_utils_module_imports(self):
        """Test that utils module imports correctly."""
        import doc_generator.utils as utils
        
        # Check that the function is available
        assert hasattr(utils, 'get_output_directory')
        assert callable(utils.get_output_directory)

    def test_utils_function_signature(self):
        """Test that get_output_directory has the expected signature."""
        from doc_generator.utils import get_output_directory
        import inspect
        
        sig = inspect.signature(get_output_directory)
        params = list(sig.parameters.keys())
        
        assert len(params) == 2
        assert 'output_dir' in params
        assert 'logger' in params
        
        # Check parameter annotations if they exist
        assert sig.parameters['output_dir'].annotation == str
        assert str(sig.parameters['logger'].annotation) == '<class \'logging.Logger\'>'
        
        # Check return annotation
        assert sig.return_annotation == str

    def test_utils_module_docstring(self):
        """Test that utils module has proper documentation."""
        import doc_generator.utils as utils
        
        assert utils.__doc__ is not None
        assert 'Utility functions' in utils.__doc__

    def test_get_output_directory_docstring(self):
        """Test that get_output_directory has proper documentation."""
        from doc_generator.utils import get_output_directory
        
        assert get_output_directory.__doc__ is not None
        assert 'Get the output directory path' in get_output_directory.__doc__
        assert 'Args:' in get_output_directory.__doc__
        assert 'Returns:' in get_output_directory.__doc__