"""
Integration tests for CLI workflows.

These tests focus on end-to-end testing of CLI functions with mocked
external dependencies but real internal logic flow.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import sys
import logging
from io import StringIO

from doc_generator.cli import (
    run_generation, run_readme_generation, run_standardization,
    parse_args, main, setup_logging
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_args():
    """Create sample args namespace for testing."""
    args = Mock()
    # Common defaults
    args.verbose = False
    args.quiet = False
    args.temperature = 0.7
    args.runs = 1
    args.provider = 'auto'
    args.model = 'gpt-4o-mini'
    args.output_dir = None
    args.prompt_yaml_path = './prompts/generator/default.yaml'
    args.terminology_path = './terminology.yaml'
    args.shots = None
    args.examples_dir = './shots'
    args.analyze = False
    args.quality_eval = False
    args.compare_url = None
    args.compare_file = None
    args.report_format = 'markdown'
    args.analysis_prompt_path = './prompts/analysis/default.yaml'
    args.disable_plugins = []
    args.enable_only = []
    args.list_plugins = False
    args.list_models = False
    
    # Token analysis related attributes
    args.token_analyze = None
    args.content = None
    args.content_type = 'text'
    args.context_size = None
    args.expected_output = None
    args.analysis_depth = 'standard'
    
    # Additional CLI attributes that might be checked
    args.token_estimate = None
    args.token_report = False
    args.token_optimize = None
    args.max_cost = None
    args.min_quality = None
    args.period = 30
    
    return args


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    logger = Mock(spec=logging.Logger)
    return logger


class TestTopicGenerationWorkflow:
    """Test CLI topic generation workflow."""
    
    @patch('doc_generator.cli.DocumentationGenerator')
    @patch('doc_generator.cli.get_output_directory')
    @patch('doc_generator.cli.load_dotenv')
    def test_basic_topic_generation(self, mock_dotenv, mock_get_output, 
                                   mock_generator_class, sample_args, mock_logger, temp_dir):
        """Test basic topic documentation generation workflow."""
        # Setup
        sample_args.topic = "Python Programming"
        sample_args.list_models = False
        sample_args.list_plugins = False
        
        mock_get_output.return_value = str(temp_dir)
        
        # Mock generator instance and methods
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.plugin_manager.engines = {}
        
        # Mock successful generation
        result_files = [str(temp_dir / "python_programming_v1.html")]
        mock_generator.generate_documentation.return_value = result_files
        
        # Create the mock result file
        Path(result_files[0]).write_text("<html><body>Test content</body></html>")
        
        # Execute
        run_generation(sample_args, mock_logger)
        
        # Verify
        mock_dotenv.assert_called_once()
        mock_generator_class.assert_called_once()
        mock_generator.generate_documentation.assert_called_once_with(
            query="Python Programming",
            runs=1,
            model='gpt-4o-mini',
            temperature=0.7,
            topic_filename='python_programming',
            output_dir=str(temp_dir),
            provider=None
        )
        
        # Check logging
        assert mock_logger.info.call_count >= 2
        mock_logger.info.assert_any_call("Initializing documentation generator...")
        mock_logger.info.assert_any_call("Generating documentation for topic: 'Python Programming'")

    @patch('doc_generator.providers.ProviderManager')
    @patch('doc_generator.cli.load_dotenv')
    def test_list_models_workflow(self, mock_dotenv, mock_provider_class, 
                                  sample_args, mock_logger, capsys):
        """Test --list-models workflow."""
        # Setup
        sample_args.list_models = True
        sample_args.topic = None
        
        mock_manager = Mock()
        mock_provider_class.return_value = mock_manager
        
        # Mock available models
        mock_manager.get_available_models.return_value = {
            'openai': ['gpt-4', 'gpt-3.5-turbo'],
            'claude': ['claude-3-sonnet', 'claude-3-haiku']
        }
        
        # Mock providers
        mock_openai_provider = Mock()
        mock_openai_provider.is_available.return_value = True
        mock_claude_provider = Mock() 
        mock_claude_provider.is_available.return_value = False
        
        mock_manager.get_provider.side_effect = lambda name: {
            'openai': mock_openai_provider,
            'claude': mock_claude_provider
        }.get(name)
        
        mock_manager.get_default_provider.return_value = 'openai'
        mock_manager.get_default_model.return_value = 'gpt-4'
        
        # Execute
        run_generation(sample_args, mock_logger)
        
        # Verify output
        captured = capsys.readouterr()
        assert "Available LLM Providers and Models:" in captured.out
        assert "OPENAI: ✅ CONFIGURED" in captured.out
        assert "CLAUDE: ❌ NOT CONFIGURED" in captured.out
        assert "gpt-4" in captured.out
        assert "claude-3-sonnet" in captured.out
        assert "Default provider: openai" in captured.out

    @patch('doc_generator.cli.DocumentationGenerator')
    @patch('doc_generator.cli.get_output_directory')
    @patch('doc_generator.cli.load_dotenv')
    def test_multiple_runs_with_analysis(self, mock_dotenv, mock_get_output, 
                                        mock_generator_class, sample_args, mock_logger, temp_dir):
        """Test generation with multiple runs and analysis enabled."""
        # Setup
        sample_args.topic = "Machine Learning"
        sample_args.runs = 3
        sample_args.analyze = True
        sample_args.list_models = False
        sample_args.list_plugins = False
        
        mock_get_output.return_value = str(temp_dir)
        
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.plugin_manager.engines = {}
        
        # Mock multiple results
        result_files = [
            str(temp_dir / "machine_learning_v1.html"),
            str(temp_dir / "machine_learning_v2.html"),
            str(temp_dir / "machine_learning_v3.html")
        ]
        mock_generator.generate_documentation.return_value = result_files
        
        # Create mock result files
        for file_path in result_files:
            Path(file_path).write_text(f"<html><body>Content for {Path(file_path).name}</body></html>")
        
        # Mock analysis pipeline
        mock_generator.plugin_manager.load_analysis_plugins.return_value = None
        mock_generator.plugin_manager.run_analysis_pipeline.return_value = {
            'document_compiler': {
                'artifacts': [str(temp_dir / 'machine_learning_best_compilation.html')]
            },
            'link_validator': {
                'artifacts': [str(temp_dir / 'machine_learning_link_validation.md')]
            }
        }
        
        # Execute
        run_generation(sample_args, mock_logger)
        
        # Verify analysis was triggered
        mock_generator.plugin_manager.load_analysis_plugins.assert_called_once()
        mock_generator.plugin_manager.run_analysis_pipeline.assert_called_once()
        
        # Check that proper configuration was passed
        call_args = mock_generator.plugin_manager.load_analysis_plugins.call_args
        assert 'config' in call_args.kwargs
        config = call_args.kwargs['config']
        assert 'reporter' in config
        assert 'link_validator' in config

    @patch('doc_generator.cli.DocumentationGenerator')
    @patch('doc_generator.cli.get_output_directory')
    @patch('doc_generator.cli.load_dotenv')
    def test_generation_with_plugin_filtering(self, mock_dotenv, mock_get_output, 
                                             mock_generator_class, sample_args, mock_logger, temp_dir):
        """Test generation with plugin filtering."""
        # Setup
        sample_args.topic = "Data Science"
        sample_args.disable_plugins = ['module_recommender']
        sample_args.enable_only = []
        sample_args.list_models = False
        sample_args.list_plugins = False
        
        mock_get_output.return_value = str(temp_dir)
        
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.plugin_manager.engines = {
            'module_recommender': Mock(),
            'datasets': Mock()
        }
        
        result_files = [str(temp_dir / "data_science_v1.html")]
        mock_generator.generate_documentation.return_value = result_files
        Path(result_files[0]).write_text("<html><body>Test</body></html>")
        
        # Execute
        run_generation(sample_args, mock_logger)
        
        # Verify plugin was disabled
        assert 'module_recommender' not in mock_generator.plugin_manager.engines
        assert 'datasets' in mock_generator.plugin_manager.engines
        mock_logger.info.assert_any_call("Disabled plugin: module_recommender")

    @patch('doc_generator.cli.DocumentationGenerator')
    @patch('doc_generator.cli.load_dotenv')
    def test_generation_error_handling(self, mock_dotenv, mock_generator_class, 
                                      sample_args, mock_logger):
        """Test error handling in generation workflow."""
        # Setup
        sample_args.topic = "Test Topic"
        sample_args.list_models = False
        sample_args.list_plugins = False
        
        # Mock generator to raise exception
        mock_generator_class.side_effect = Exception("Test error")
        
        # Execute and verify SystemExit is raised
        with pytest.raises(SystemExit) as exc_info:
            run_generation(sample_args, mock_logger)
        
        assert exc_info.value.code == 1
        mock_logger.error.assert_called_with("Error during generation: Test error")


class TestReadmeGenerationWorkflow:
    """Test CLI README generation workflow."""
    
    @patch('doc_generator.readme_documentation_generator.ReadmeDocumentationGenerator')
    @patch('doc_generator.cli.get_output_directory')
    @patch('doc_generator.cli.load_dotenv')
    def test_basic_readme_generation(self, mock_dotenv, mock_get_output, 
                                    mock_generator_class, sample_args, mock_logger, temp_dir):
        """Test basic README generation workflow."""
        # Setup
        test_directory = temp_dir / "test_project"
        test_directory.mkdir()
        (test_directory / "main.py").write_text("print('hello')")
        
        sample_args.readme = str(test_directory)
        sample_args.recursive = False
        
        mock_get_output.return_value = str(temp_dir)
        
        # Mock generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        # Mock successful generation results
        mock_results = {
            'generated_files': [
                str(temp_dir / 'test_project_readme_v1.md'),
                str(temp_dir / 'test_project_readme_v2.md'),
                str(temp_dir / 'test_project_best_compilation.md')
            ],
            'directory_info': {
                'name': 'test_project',
                'path': str(test_directory),
                'languages': ['python'],
                'subdirectories': [],
                'files': ['main.py']
            },
            'depth_level': 1,
            'analysis_results': {
                'overall_best': {
                    'file': str(temp_dir / 'test_project_best_compilation.md'),
                    'total_score': 8.5
                },
                'section_winners': {
                    'Overview': {'version': 0, 'score': 9.0},
                    'Installation': {'version': 1, 'score': 8.2}
                }
            }
        }
        mock_generator.generate_readme_documentation.return_value = mock_results
        
        # Execute
        run_readme_generation(sample_args, mock_logger)
        
        # Verify
        mock_dotenv.assert_called_once()
        mock_generator_class.assert_called_once()
        mock_generator.generate_readme_documentation.assert_called_once_with(
            directory_path=str(test_directory),
            runs=3,  # Default for README
            model='gpt-4o-mini',
            temperature=0.7,
            analyze=False,  # Analyze not enabled by default
            provider=None,
            output_dir=str(temp_dir)
        )
        
        mock_logger.info.assert_any_call(f"Generating README for directory: {test_directory}")

    @patch('doc_generator.readme_documentation_generator.ReadmeDocumentationGenerator')
    @patch('doc_generator.cli.get_output_directory')  
    @patch('doc_generator.cli.load_dotenv')
    def test_recursive_readme_generation(self, mock_dotenv, mock_get_output,
                                        mock_generator_class, sample_args, mock_logger, temp_dir):
        """Test recursive README generation workflow."""
        # Setup directory structure
        root_dir = temp_dir / "project"
        root_dir.mkdir()
        sub_dir1 = root_dir / "module1"
        sub_dir1.mkdir()
        sub_dir2 = root_dir / "module2" 
        sub_dir2.mkdir()
        
        (root_dir / "main.py").write_text("print('main')")
        (sub_dir1 / "utils.py").write_text("def helper(): pass")
        (sub_dir2 / "core.py").write_text("class Core: pass")
        
        sample_args.readme = str(root_dir)
        sample_args.recursive = True
        
        mock_get_output.return_value = str(temp_dir)
        
        # Mock generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        # Mock main directory results
        main_results = {
            'generated_files': [str(temp_dir / 'project_readme_v1.md')],
            'directory_info': {
                'name': 'project',
                'path': str(root_dir),
                'languages': ['python'],
                'subdirectories': [
                    {'name': 'module1', 'path': str(sub_dir1)},
                    {'name': 'module2', 'path': str(sub_dir2)}
                ],
                'files': ['main.py']
            },
            'depth_level': 1
        }
        
        # Mock subdirectory results  
        sub_results = {
            'generated_files': [str(sub_dir1 / 'README.md')],
            'directory_info': {
                'name': 'module1',
                'path': str(sub_dir1),
                'languages': ['python'],
                'subdirectories': [],
                'files': ['utils.py']
            },
            'depth_level': 2
        }
        
        mock_generator.generate_readme_documentation.side_effect = [main_results, sub_results, sub_results]
        
        # Execute
        run_readme_generation(sample_args, mock_logger)
        
        # Verify main call + 2 subdirectory calls
        assert mock_generator.generate_readme_documentation.call_count == 3
        
        # Verify recursive processing logged
        mock_logger.info.assert_any_call("Processing subdirectories recursively...")


class TestStandardizationWorkflow:
    """Test CLI document standardization workflow."""
    
    @patch('doc_generator.standardizers.DocumentStandardizer')
    @patch('doc_generator.cli.get_output_directory')
    @patch('doc_generator.cli.load_dotenv')
    def test_file_standardization(self, mock_dotenv, mock_get_output, 
                                 mock_standardizer_class, sample_args, mock_logger, temp_dir):
        """Test file-based document standardization."""
        # Setup input file
        input_file = temp_dir / "input.html"
        input_file.write_text("<html><body><h1>Test</h1><p>Content</p></body></html>")
        
        sample_args.standardize = str(input_file)
        sample_args.target_format = 'markdown'
        sample_args.format = 'markdown'
        sample_args.template = 'default'
        
        mock_get_output.return_value = str(temp_dir)
        
        # Mock standardizer
        mock_standardizer = Mock()
        mock_standardizer_class.return_value = mock_standardizer
        
        mock_result = {
            'standardized_content': '# Test\n\nContent',
            'original_format': 'html',
            'target_format': 'markdown',
            'sections_processed': ['Introduction', 'Content'],
            'metadata': {
                'provider': 'openai',
                'model': 'gpt-4o-mini',
                'tokens_used': 150
            }
        }
        mock_standardizer.standardize_document.return_value = mock_result
        
        # Execute
        run_standardization(sample_args, mock_logger)
        
        # Verify
        mock_dotenv.assert_called_once()
        mock_standardizer_class.assert_called_once_with(
            provider=None,
            model='gpt-4o-mini', 
            temperature=0.7,
            output_format='markdown'
        )
        
        mock_standardizer.standardize_document.assert_called_once_with(
            content="<html><body><h1>Test</h1><p>Content</p></body></html>",
            file_path=str(input_file),
            target_format='markdown'
        )
        
        # Check output file was created
        output_file = temp_dir / "input_standardized.md"
        assert output_file.exists()
        assert output_file.read_text() == '# Test\n\nContent'

    @patch('doc_generator.standardizers.DocumentStandardizer')
    @patch('doc_generator.cli.get_output_directory')
    @patch('doc_generator.cli.load_dotenv')
    @patch('urllib.request.urlopen')
    def test_url_standardization(self, mock_urlopen, mock_dotenv, mock_get_output,
                                mock_standardizer_class, sample_args, mock_logger, temp_dir):
        """Test URL-based document standardization."""
        # Setup
        sample_args.standardize = "https://example.com/document.html"
        sample_args.target_format = 'auto'
        sample_args.format = 'auto'
        sample_args.template = 'default'
        
        mock_get_output.return_value = str(temp_dir)
        
        # Mock URL response
        mock_response = Mock()
        mock_response.read.return_value = b"<html><body><h1>Web Content</h1></body></html>"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response
        
        # Mock standardizer
        mock_standardizer = Mock()
        mock_standardizer_class.return_value = mock_standardizer
        
        mock_result = {
            'standardized_content': '<html><h1>Web Content</h1></html>',
            'original_format': 'html',
            'target_format': 'html',
            'sections_processed': ['Title'],
            'metadata': {
                'provider': 'openai',
                'model': 'gpt-4o-mini'
            }
        }
        mock_standardizer.standardize_document.return_value = mock_result
        
        # Execute
        run_standardization(sample_args, mock_logger)
        
        # Verify URL was fetched
        mock_urlopen.assert_called_once_with("https://example.com/document.html")
        
        # Verify standardization was called with URL content
        mock_standardizer.standardize_document.assert_called_once_with(
            content="<html><body><h1>Web Content</h1></body></html>",
            file_path="https://example.com/document.html",
            target_format='html'  # auto-detected for URLs
        )

    @patch('doc_generator.cli.load_dotenv')
    def test_standardization_file_not_found(self, mock_dotenv, sample_args, mock_logger):
        """Test standardization error handling for missing files."""
        # Setup
        sample_args.standardize = "/nonexistent/file.html"
        sample_args.target_format = 'markdown'
        sample_args.format = 'markdown'
        sample_args.template = 'default'
        
        # Execute and verify SystemExit is raised
        with pytest.raises(SystemExit) as exc_info:
            run_standardization(sample_args, mock_logger)
        
        assert exc_info.value.code == 1
        mock_logger.error.assert_called_with("Input file not found: /nonexistent/file.html")


class TestCLIMainFunction:
    """Test the main CLI entry point function."""
    
    @patch('doc_generator.cli.run_generation')
    @patch('doc_generator.cli.parse_args')
    @patch('doc_generator.cli.setup_logging')
    def test_main_topic_mode(self, mock_setup_logging, mock_parse_args, mock_run_generation):
        """Test main function routing to topic generation."""
        # Setup
        mock_args = Mock()
        mock_args.topic = "Test Topic"
        mock_args.readme = None
        mock_args.standardize = None
        mock_args.token_analyze = False
        mock_args.token_estimate = False
        mock_args.token_report = False
        mock_args.token_optimize = False
        mock_args.scan_code = None
        mock_args.cleanup = False
        mock_args.info = False
        mock_args.verbose = False
        
        mock_parse_args.return_value = mock_args
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        
        # Execute
        main()
        
        # Verify
        mock_parse_args.assert_called_once()
        mock_setup_logging.assert_called_once_with(mock_args.verbose)
        mock_run_generation.assert_called_once_with(mock_args, mock_logger)

    @patch('doc_generator.cli.run_readme_generation')
    @patch('doc_generator.cli.parse_args')
    @patch('doc_generator.cli.setup_logging')
    def test_main_readme_mode(self, mock_setup_logging, mock_parse_args, mock_run_readme):
        """Test main function routing to README generation."""
        # Setup
        mock_args = Mock()
        mock_args.topic = None
        mock_args.readme = "/path/to/directory"
        mock_args.standardize = None
        mock_args.token_analyze = False
        mock_args.token_estimate = False
        mock_args.token_report = False
        mock_args.token_optimize = False
        mock_args.scan_code = None
        mock_args.cleanup = False
        mock_args.info = False
        mock_args.verbose = False
        
        mock_parse_args.return_value = mock_args
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        
        # Execute
        main()
        
        # Verify
        mock_run_readme.assert_called_once_with(mock_args, mock_logger)

    @patch('doc_generator.cli.run_standardization')
    @patch('doc_generator.cli.parse_args')
    @patch('doc_generator.cli.setup_logging')
    def test_main_standardization_mode(self, mock_setup_logging, mock_parse_args, mock_run_std):
        """Test main function routing to document standardization."""
        # Setup
        mock_args = Mock()
        mock_args.topic = None
        mock_args.readme = None
        mock_args.standardize = "document.html"
        mock_args.token_analyze = False
        mock_args.token_estimate = False
        mock_args.token_report = False
        mock_args.token_optimize = False
        mock_args.scan_code = None
        mock_args.cleanup = False
        mock_args.info = False
        mock_args.verbose = False
        
        mock_parse_args.return_value = mock_args
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        
        # Execute  
        main()
        
        # Verify
        mock_run_std.assert_called_once_with(mock_args, mock_logger)

    @patch('doc_generator.cli.display_info')
    @patch('doc_generator.cli.parse_args')
    @patch('doc_generator.cli.setup_logging')
    def test_main_info_mode(self, mock_setup_logging, mock_parse_args, mock_display_info):
        """Test main function routing to info display."""
        # Setup
        mock_args = Mock()
        mock_args.topic = None
        mock_args.readme = None
        mock_args.standardize = None
        mock_args.token_analyze = False
        mock_args.token_estimate = False
        mock_args.token_report = False
        mock_args.token_optimize = False
        mock_args.scan_code = None
        mock_args.cleanup = False
        mock_args.info = True
        mock_args.verbose = False
        
        mock_parse_args.return_value = mock_args
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        
        # Execute
        main()
        
        # Verify
        mock_display_info.assert_called_once()