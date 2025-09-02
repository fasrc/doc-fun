"""
Simplified CLI integration tests to verify core workflows work.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

from doc_generator.cli import run_generation, parse_args


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_args():
    """Create mock args for testing."""
    args = Mock()
    args.verbose = False
    args.quiet = False
    args.temperature = 0.7
    args.runs = 1
    args.provider = 'auto'
    args.model = 'gpt-4o-mini'
    args.output_dir = './test_output'  # Provide actual directory
    args.topic = "Test Topic"
    args.list_models = False
    args.list_plugins = False
    args.analyze = False
    args.quality_eval = False
    args.compare_url = None
    args.compare_file = None
    args.disable_plugins = []
    args.enable_only = []
    args.prompt_yaml_path = './prompts/generator/default.yaml'
    args.terminology_path = './terminology.yaml'
    args.shots = None
    args.examples_dir = './shots'
    args.report_format = 'markdown'
    args.analysis_prompt_path = './prompts/analysis/default.yaml'
    return args


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return Mock(spec=logging.Logger)


class TestBasicCLIIntegration:
    """Test basic CLI integration functionality."""
    
    @patch('doc_generator.cli.DocumentationGenerator')
    @patch('doc_generator.utils.get_output_directory')
    @patch('doc_generator.cli.load_dotenv')
    def test_basic_topic_generation_flow(self, mock_dotenv, mock_get_output, 
                                        mock_generator_class, mock_args, mock_logger, temp_dir):
        """Test the basic flow of topic generation without complex mocking."""
        # Setup
        mock_get_output.return_value = str(temp_dir)
        
        # Mock generator instance
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.plugin_manager.engines = {}
        
        # Mock successful generation
        result_files = [str(temp_dir / "test_topic_v1.html")]
        mock_generator.generate_documentation.return_value = result_files
        
        # Create the result file
        Path(result_files[0]).write_text("<html><body>Generated content</body></html>")
        
        # Execute
        run_generation(mock_args, mock_logger)
        
        # Verify the flow worked
        mock_dotenv.assert_called_once()
        mock_generator_class.assert_called_once()
        mock_generator.generate_documentation.assert_called_once()
        
        # Verify logging
        assert mock_logger.info.call_count >= 2

    @patch('doc_generator.providers.manager.ProviderManager')
    @patch('doc_generator.cli.load_dotenv')
    def test_list_models_basic_flow(self, mock_dotenv, mock_provider_class, 
                                   mock_args, mock_logger, capsys):
        """Test --list-models workflow without complex mocking."""
        # Setup
        mock_args.list_models = True
        mock_args.topic = None
        
        # Mock provider manager
        mock_manager = Mock()
        mock_provider_class.return_value = mock_manager
        mock_manager.get_available_models.return_value = {
            'openai': ['gpt-4'],
            'claude': ['claude-3-sonnet']
        }
        
        # Mock providers
        mock_openai = Mock()
        mock_openai.is_available.return_value = True
        mock_claude = Mock()
        mock_claude.is_available.return_value = False
        
        mock_manager.get_provider.side_effect = lambda name: {
            'openai': mock_openai,
            'claude': mock_claude
        }.get(name)
        
        mock_manager.get_default_provider.return_value = 'openai'
        mock_manager.get_default_model.return_value = 'gpt-4'
        
        # Execute
        run_generation(mock_args, mock_logger)
        
        # Verify output
        captured = capsys.readouterr()
        assert "Available LLM Providers and Models:" in captured.out
        assert "OPENAI: ✅ CONFIGURED" in captured.out

    def test_argument_parsing_basic(self):
        """Test basic argument parsing functionality."""
        with patch('sys.argv', ['doc-gen', '--topic', 'Test Topic']):
            args = parse_args()
            assert args.topic == 'Test Topic'
            assert hasattr(args, 'provider')
            assert hasattr(args, 'model')


class TestCLIErrorHandling:
    """Test error handling in CLI functions."""
    
    @patch('doc_generator.cli.DocumentationGenerator')
    @patch('doc_generator.cli.load_dotenv')
    def test_generation_error_handling(self, mock_dotenv, mock_generator_class, 
                                      mock_args, mock_logger):
        """Test error handling when generation fails."""
        # Mock generator to raise exception
        mock_generator_class.side_effect = Exception("Test error")
        
        # Execute and verify SystemExit is raised
        with pytest.raises(SystemExit) as exc_info:
            run_generation(mock_args, mock_logger)
        
        assert exc_info.value.code == 1
        mock_logger.error.assert_called_with("Error during generation: Test error")

    def test_missing_topic_error(self, mock_logger):
        """Test error when topic is missing for generation."""
        args = Mock()
        args.topic = None
        args.list_models = False
        args.list_plugins = False
        args.verbose = False
        
        # Add all required attributes to prevent early returns
        args.token_analyze = None
        args.token_estimate = None
        args.token_report = False
        args.token_optimize = None
        args.readme = None
        args.standardize = None
        args.scan_code = None
        args.cleanup = False
        args.info = False
        
        # Add attributes for plugin filtering
        args.disable_plugins = []
        args.enable_only = []
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.DocumentationGenerator') as mock_gen_class:
                mock_gen = Mock()
                mock_gen.plugin_manager.engines = {}
                mock_gen_class.return_value = mock_gen
                
                with pytest.raises(SystemExit) as exc_info:
                    run_generation(args, mock_logger)
                
                assert exc_info.value.code == 1
                mock_logger.error.assert_called_with("--topic is required for documentation generation")