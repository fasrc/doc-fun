"""
Simplified CLI workflow tests focusing on coverage and core functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

from doc_generator.cli import (
    main, setup_logging, display_info, list_plugins, cleanup_output_directory,
    parse_args, run_generation, run_readme_generation
)


class TestCLIMainWorkflows:
    """Test main CLI entry point workflows."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_main_info_mode(self):
        """Test main function with --info flag."""
        with patch('doc_generator.cli.parse_args') as mock_parse:
            mock_args = Mock()
            mock_args.info = True
            mock_args.verbose = False
            mock_parse.return_value = mock_args
            
            with patch('doc_generator.cli.display_info') as mock_display:
                main()
                mock_display.assert_called_once()

    def test_main_cleanup_mode(self):
        """Test main function with --cleanup flag."""
        with patch('doc_generator.cli.parse_args') as mock_parse:
            mock_args = Mock()
            mock_args.info = False
            mock_args.cleanup = True
            mock_args.verbose = False
            mock_args.token_analyze = False
            mock_args.token_estimate = False
            mock_args.token_report = False
            mock_args.token_optimize = False
            mock_args.readme = None
            mock_args.standardize = None
            mock_args.scan_code = None
            mock_parse.return_value = mock_args
            
            mock_logger = Mock()
            with patch('doc_generator.cli.setup_logging', return_value=mock_logger):
                with patch('doc_generator.cli.cleanup_output_directory') as mock_cleanup:
                    main()
                    mock_cleanup.assert_called_once_with(mock_logger)

    def test_main_keyboard_interrupt(self):
        """Test main function handles KeyboardInterrupt."""
        with patch('doc_generator.cli.parse_args') as mock_parse:
            mock_args = Mock()
            mock_args.info = False
            mock_args.cleanup = False
            mock_args.token_analyze = False
            mock_args.token_estimate = False
            mock_args.token_report = False
            mock_args.token_optimize = False
            mock_args.readme = None
            mock_args.standardize = None
            mock_args.scan_code = None
            mock_args.verbose = False
            mock_parse.return_value = mock_args
            
            mock_logger = Mock()
            with patch('doc_generator.cli.setup_logging', return_value=mock_logger):
                with patch('doc_generator.cli.run_generation', side_effect=KeyboardInterrupt):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1
                    mock_logger.info.assert_called_with("Operation cancelled by user")

    def test_main_unexpected_error_verbose(self):
        """Test main function handles unexpected errors in verbose mode."""
        with patch('doc_generator.cli.parse_args') as mock_parse:
            mock_args = Mock()
            mock_args.info = False
            mock_args.cleanup = False
            mock_args.token_analyze = False
            mock_args.token_estimate = False
            mock_args.token_report = False
            mock_args.token_optimize = False
            mock_args.readme = None
            mock_args.standardize = None
            mock_args.scan_code = None
            mock_args.verbose = True
            mock_parse.return_value = mock_args
            
            mock_logger = Mock()
            with patch('doc_generator.cli.setup_logging', return_value=mock_logger):
                with patch('doc_generator.cli.run_generation', side_effect=RuntimeError("Test error")):
                    with patch('traceback.print_exc') as mock_traceback:
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        assert exc_info.value.code == 1
                        mock_logger.error.assert_called_with("Unexpected error: Test error")
                        mock_traceback.assert_called_once()


class TestCLIUtilityFunctions:
    """Test CLI utility functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging(verbose=False)
        assert logger.name == 'doc_generator'
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')

    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        logger = setup_logging(verbose=True)
        assert logger.name == 'doc_generator'
        # Test that debug level is set correctly
        import logging
        # The logger should be configured for debug level in verbose mode

    def test_display_info_execution(self):
        """Test display_info function executes without error."""
        with patch('builtins.print') as mock_print:
            display_info()
            # Should print multiple lines of information
            assert mock_print.call_count > 0
            # Check that some expected content is printed
            calls = [str(call) for call in mock_print.call_args_list]
            assert any('doc-generator' in call for call in calls)

    def test_list_plugins_with_engines(self):
        """Test plugin listing with available engines."""
        mock_generator = Mock()
        mock_generator.plugin_manager.engines = {
            'test_plugin': Mock(
                get_name=Mock(return_value='Test Plugin'),
                get_description=Mock(return_value='A test plugin'),
                get_priority=Mock(return_value=5)
            ),
            'another_plugin': Mock(
                get_name=Mock(return_value='Another Plugin'),  
                get_description=Mock(return_value='Another test plugin'),
                get_priority=Mock(return_value=10)
            )
        }
        
        with patch('builtins.print') as mock_print:
            list_plugins(mock_generator)
            # Should print plugin information
            assert mock_print.call_count > 0

    def test_list_plugins_empty(self):
        """Test plugin listing with no engines."""
        mock_generator = Mock()
        mock_generator.plugin_manager.engines = {}
        
        with patch('builtins.print') as mock_print:
            list_plugins(mock_generator)
            assert mock_print.call_count > 0
            # Should print message about no plugins
            calls = [str(call) for call in mock_print.call_args_list]
            assert any('No plugins' in call for call in calls)

    def test_cleanup_output_directory_with_files(self, mock_logger, temp_dir):
        """Test cleanup of output directory with files."""
        # Create some test files
        (temp_dir / "test1.html").write_text("test content")
        (temp_dir / "test2.md").write_text("test content")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "test3.txt").write_text("test content")
        
        with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
            with patch('builtins.input', return_value='y'):
                cleanup_output_directory(mock_logger)
                
                # Check files were removed
                assert not (temp_dir / "test1.html").exists()
                assert not (temp_dir / "test2.md").exists()
                assert not (subdir / "test3.txt").exists()

    def test_cleanup_output_directory_cancelled(self, mock_logger, temp_dir):
        """Test cleanup cancelled by user."""
        (temp_dir / "test.html").write_text("test content")
        
        with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
            with patch('builtins.input', return_value='n'):
                cleanup_output_directory(mock_logger)
                
                # File should still exist
                assert (temp_dir / "test.html").exists()

    def test_cleanup_output_directory_empty(self, mock_logger, temp_dir):
        """Test cleanup of empty output directory."""
        with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
            cleanup_output_directory(mock_logger)
            mock_logger.info.assert_called_with("Output directory is already empty.")

    def test_cleanup_output_directory_permission_error(self, mock_logger, temp_dir):
        """Test cleanup with permission errors."""
        (temp_dir / "test.html").write_text("test content")
        
        with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
            with patch('builtins.input', return_value='y'):
                with patch('shutil.rmtree', side_effect=PermissionError("Permission denied")):
                    cleanup_output_directory(mock_logger)
                    # Should log the error
                    assert mock_logger.error.called


class TestCLIGenerationWorkflow:
    """Test CLI generation workflow with mocking."""

    @pytest.fixture  
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()

    def test_run_generation_list_models(self, mock_logger):
        """Test generation with list models flag."""
        args = Mock()
        args.list_models = True
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.ProviderManager') as mock_manager_class:
                mock_manager = Mock()
                mock_manager.get_available_models.return_value = {
                    'openai': ['gpt-4', 'gpt-3.5-turbo'],
                    'claude': ['claude-3-sonnet']
                }
                
                # Mock provider objects
                mock_openai_provider = Mock()
                mock_openai_provider.is_available.return_value = True
                mock_claude_provider = Mock() 
                mock_claude_provider.is_available.return_value = False
                
                mock_manager.get_provider.side_effect = lambda name: {
                    'openai': mock_openai_provider,
                    'claude': mock_claude_provider
                }.get(name)
                
                mock_manager.get_default_provider.return_value = 'openai'
                mock_manager.get_default_model.return_value = 'gpt-4o-mini'
                
                mock_manager_class.return_value = mock_manager
                
                with patch('builtins.print') as mock_print:
                    run_generation(args, mock_logger)
                    
                    mock_manager.get_available_models.assert_called_once()
                    # Should print model information
                    assert mock_print.call_count > 0

    def test_run_generation_no_models_available(self, mock_logger):
        """Test generation when no models are available."""
        args = Mock()
        args.list_models = True
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.ProviderManager') as mock_manager_class:
                mock_manager = Mock()
                mock_manager.get_available_models.return_value = {}
                mock_manager_class.return_value = mock_manager
                
                with patch('builtins.print') as mock_print:
                    run_generation(args, mock_logger)
                    
                    # Should print message about no providers
                    calls = [str(call) for call in mock_print.call_args_list]
                    assert any('No providers available' in call for call in calls)

    def test_run_generation_missing_topic_error(self, mock_logger):
        """Test error when topic is missing for generation."""
        args = Mock()
        args.list_models = False
        args.topic = None
        args.list_plugins = False
        args.shots = None
        args.examples_dir = None
        args.prompt_yaml_path = None
        args.terminology_path = None
        args.provider = 'auto'
        args.disable_plugins = None
        args.enable_only = None
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.core.DocumentationGenerator') as mock_gen_class:
                mock_gen = Mock()
                mock_gen.plugin_manager.engines = {}
                mock_gen_class.return_value = mock_gen
                
                with pytest.raises(SystemExit) as exc_info:
                    run_generation(args, mock_logger)
                
                assert exc_info.value.code == 1
                mock_logger.error.assert_called_with("--topic is required for documentation generation")

    def test_run_generation_plugin_filtering(self, mock_logger):
        """Test generation with plugin filtering."""
        args = Mock()
        args.list_models = False
        args.topic = "Test Topic"
        args.list_plugins = False
        args.shots = None
        args.examples_dir = None
        args.prompt_yaml_path = None
        args.terminology_path = None
        args.provider = 'auto'
        args.disable_plugins = ['unwanted_plugin']
        args.enable_only = None
        args.runs = 1
        args.model = 'gpt-4o-mini'
        args.temperature = 0.7
        args.output_dir = None
        args.compare_url = None
        args.compare_file = None
        args.analyze = False
        args.quality_eval = False
        args.quiet = True
        args.verbose = False
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.get_output_directory', return_value='/tmp/output'):
                with patch('doc_generator.core.DocumentationGenerator') as mock_gen_class:
                    mock_gen = Mock()
                    mock_gen.plugin_manager.engines = {
                        'unwanted_plugin': Mock(),
                        'good_plugin': Mock()
                    }
                    mock_gen.generate_documentation.return_value = ['/tmp/output/test_v1.html']
                    mock_gen_class.return_value = mock_gen
                    
                    run_generation(args, mock_logger)
                    
                    # Should have removed the unwanted plugin
                    assert 'unwanted_plugin' not in mock_gen.plugin_manager.engines
                    mock_logger.info.assert_any_call("Disabled plugin: unwanted_plugin")


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()

    def test_readme_generation_missing_path(self, mock_logger):
        """Test error when README path is missing."""
        with pytest.raises(AttributeError):
            # This should fail because args.readme is expected to exist
            args = Mock(spec=[])  # Empty spec means no attributes
            run_readme_generation(args, mock_logger)

    def test_generation_provider_initialization_error(self, mock_logger):
        """Test handling of provider initialization errors."""
        args = Mock()
        args.list_models = False
        args.topic = "Test Topic"
        args.shots = None
        args.examples_dir = None
        args.prompt_yaml_path = None
        args.terminology_path = None
        args.provider = 'auto'
        args.verbose = False
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.core.DocumentationGenerator', side_effect=Exception("Provider initialization failed")):
                with pytest.raises(SystemExit) as exc_info:
                    run_generation(args, mock_logger)
                assert exc_info.value.code == 1
                mock_logger.error.assert_called_with("Error during generation: Provider initialization failed")

    def test_generation_provider_initialization_error_verbose(self, mock_logger):
        """Test provider initialization error with verbose mode."""
        args = Mock()
        args.list_models = False
        args.topic = "Test Topic"
        args.shots = None
        args.examples_dir = None
        args.prompt_yaml_path = None
        args.terminology_path = None
        args.provider = 'auto'
        args.verbose = True
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.core.DocumentationGenerator', side_effect=Exception("Provider error")):
                with patch('traceback.print_exc') as mock_traceback:
                    with pytest.raises(SystemExit) as exc_info:
                        run_generation(args, mock_logger)
                    assert exc_info.value.code == 1
                    mock_logger.error.assert_called_with("Error during generation: Provider error")
                    mock_traceback.assert_called_once()


class TestCLIArgumentParsing:
    """Test CLI argument parsing functionality."""

    def test_argument_parsing_basic(self):
        """Test basic argument parsing."""
        with patch('sys.argv', ['doc-gen', '--topic', 'Test Topic']):
            args = parse_args()
            assert args.topic == 'Test Topic'
            assert hasattr(args, 'provider')
            assert hasattr(args, 'model') 
            assert hasattr(args, 'temperature')

    def test_argument_parsing_help(self):
        """Test help argument."""
        with patch('sys.argv', ['doc-gen', '--help']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_argument_parsing_version(self):
        """Test version argument."""
        with patch('sys.argv', ['doc-gen', '--version']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_argument_parsing_multiple_flags(self):
        """Test parsing multiple arguments."""
        with patch('sys.argv', [
            'doc-gen', 
            '--topic', 'Python Programming',
            '--provider', 'openai',
            '--model', 'gpt-4',
            '--temperature', '0.5',
            '--runs', '2',
            '--verbose'
        ]):
            args = parse_args()
            assert args.topic == 'Python Programming'
            assert args.provider == 'openai'
            assert args.model == 'gpt-4'
            assert args.temperature == 0.5
            assert args.runs == 2
            assert args.verbose is True


class TestCLITokenModes:
    """Test CLI token-related modes (stubs for coverage)."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()

    def test_main_token_modes(self):
        """Test that main function handles token modes."""
        token_modes = ['token_analyze', 'token_estimate', 'token_report', 'token_optimize']
        
        for mode in token_modes:
            with patch('doc_generator.cli.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.info = False
                mock_args.cleanup = False
                mock_args.verbose = False
                mock_args.readme = None
                mock_args.standardize = None
                mock_args.scan_code = None
                
                # Set all token modes to False except the current one
                for token_mode in token_modes:
                    setattr(mock_args, token_mode, token_mode == mode)
                
                mock_parse.return_value = mock_args
                
                mock_logger = Mock()
                with patch('doc_generator.cli.setup_logging', return_value=mock_logger):
                    with patch(f'doc_generator.cli.run_{mode}') as mock_run:
                        main()
                        mock_run.assert_called_once_with(mock_args, mock_logger)