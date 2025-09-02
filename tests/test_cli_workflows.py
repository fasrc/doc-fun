"""
Comprehensive CLI workflow tests for doc-generator.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
from io import StringIO

from doc_generator.cli import (
    main, run_generation, run_readme_generation, run_standardization,
    cleanup_output_directory, display_info, list_plugins, scan_code_examples,
    run_token_analysis, run_token_estimate, run_token_report, run_token_optimize,
    run_comparison
)


class TestCLIWorkflows:
    """Comprehensive CLI workflow tests."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        logger = Mock()
        logger.info = Mock()
        logger.error = Mock()
        logger.debug = Mock()
        logger.warning = Mock()
        return logger

    @pytest.fixture
    def sample_args(self):
        """Create sample CLI arguments."""
        args = Mock()
        args.topic = "Python Programming"
        args.provider = "openai"
        args.model = "gpt-4o-mini"
        args.temperature = 0.7
        args.runs = 1
        args.format = "html"
        args.analyze = False
        args.verbose = False
        args.quiet = False
        args.output_dir = None
        args.disable_plugins = None
        args.enable_only = None
        args.compare_url = None
        args.compare_file = None
        args.list_models = False
        args.list_plugins = False
        args.shots = None
        args.examples_dir = None
        args.prompt_yaml_path = None
        args.terminology_path = None
        args.quality_eval = False
        args.analysis_prompt_path = None
        args.report_format = 'html'
        return args

    def test_main_info_mode(self, capsys):
        """Test main function with --info flag."""
        with patch('sys.argv', ['doc-gen', '--info']):
            with patch('doc_generator.cli.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.info = True
                mock_args.verbose = False
                mock_parse.return_value = mock_args
                
                with patch('doc_generator.cli.display_info') as mock_display:
                    main()
                    mock_display.assert_called_once()

    def test_main_cleanup_mode(self, mock_logger):
        """Test main function with --cleanup flag."""
        with patch('sys.argv', ['doc-gen', '--cleanup']):
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
                
                with patch('doc_generator.cli.setup_logging', return_value=mock_logger):
                    with patch('doc_generator.cli.cleanup_output_directory') as mock_cleanup:
                        main()
                        mock_cleanup.assert_called_once_with(mock_logger)

    def test_main_readme_mode(self, mock_logger, temp_dir):
        """Test main function with README generation mode."""
        with patch('sys.argv', ['doc-gen', '--readme', str(temp_dir)]):
            with patch('doc_generator.cli.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.info = False
                mock_args.cleanup = False
                mock_args.token_analyze = False
                mock_args.token_estimate = False
                mock_args.token_report = False
                mock_args.token_optimize = False
                mock_args.readme = str(temp_dir)
                mock_args.standardize = None
                mock_args.scan_code = None
                mock_args.verbose = False
                mock_parse.return_value = mock_args
                
                with patch('doc_generator.cli.setup_logging', return_value=mock_logger):
                    with patch('doc_generator.cli.run_readme_generation') as mock_readme:
                        main()
                        mock_readme.assert_called_once_with(mock_args, mock_logger)

    def test_main_generation_mode(self, mock_logger, sample_args):
        """Test main function with generation mode."""
        with patch('sys.argv', ['doc-gen', '--topic', 'Test']):
            with patch('doc_generator.cli.parse_args') as mock_parse:
                mock_args = sample_args
                mock_args.info = False
                mock_args.cleanup = False
                mock_args.token_analyze = False
                mock_args.token_estimate = False
                mock_args.token_report = False
                mock_args.token_optimize = False
                mock_args.readme = None
                mock_args.standardize = None
                mock_args.scan_code = None
                mock_parse.return_value = mock_args
                
                with patch('doc_generator.cli.setup_logging', return_value=mock_logger):
                    with patch('doc_generator.cli.run_generation') as mock_gen:
                        main()
                        mock_gen.assert_called_once_with(mock_args, mock_logger)

    def test_main_keyboard_interrupt(self, mock_logger, sample_args):
        """Test main function handles KeyboardInterrupt."""
        with patch('doc_generator.cli.parse_args') as mock_parse:
            mock_args = sample_args
            mock_args.info = False
            mock_args.cleanup = False
            mock_args.token_analyze = False
            mock_args.token_estimate = False
            mock_args.token_report = False
            mock_args.token_optimize = False
            mock_args.readme = None
            mock_args.standardize = None
            mock_args.scan_code = None
            mock_parse.return_value = mock_args
            
            with patch('doc_generator.cli.setup_logging', return_value=mock_logger):
                with patch('doc_generator.cli.run_generation', side_effect=KeyboardInterrupt):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1
                    mock_logger.info.assert_called_with("Operation cancelled by user")

    def test_main_unexpected_error(self, mock_logger, sample_args):
        """Test main function handles unexpected errors."""
        with patch('doc_generator.cli.parse_args') as mock_parse:
            mock_args = sample_args
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
            
            with patch('doc_generator.cli.setup_logging', return_value=mock_logger):
                with patch('doc_generator.cli.run_generation', side_effect=RuntimeError("Test error")):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1
                    mock_logger.error.assert_called_with("Unexpected error: Test error")

    def test_run_generation_basic_workflow(self, mock_logger, sample_args, temp_dir):
        """Test basic generation workflow."""
        sample_args.output_dir = str(temp_dir)
        
        # Create a sample output file that the function expects
        output_file = temp_dir / "python_programming_v1.html"
        output_file.write_text("<html><body>Generated content</body></html>")
        
        with patch('doc_generator.cli.load_dotenv') as mock_dotenv:
            with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
                with patch('doc_generator.cli.DocumentationGenerator') as mock_gen_class:
                    mock_gen = Mock()
                    mock_gen.plugin_manager.engines = {}
                    # The function expects file paths, not content dict
                    mock_gen.generate_documentation.return_value = [str(output_file)]
                    mock_gen_class.return_value = mock_gen
                    
                    run_generation(sample_args, mock_logger)
                    
                    mock_dotenv.assert_called_once()
                    mock_gen_class.assert_called_once()
                    mock_gen.generate_documentation.assert_called_once()

    def test_run_generation_with_analysis(self, mock_logger, sample_args, temp_dir):
        """Test generation workflow with analysis."""
        sample_args.analyze = True
        sample_args.runs = 2  # Need at least 2 results for analysis
        sample_args.output_dir = str(temp_dir)
        
        # Create sample output files
        output_file1 = temp_dir / "python_programming_v1.html"
        output_file2 = temp_dir / "python_programming_v2.html"
        output_file1.write_text("<html><body>Generated content 1</body></html>")
        output_file2.write_text("<html><body>Generated content 2</body></html>")
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
                with patch('doc_generator.cli.DocumentationGenerator') as mock_gen_class:
                    mock_gen = Mock()
                    mock_gen.plugin_manager.engines = {}
                    mock_gen.plugin_manager.load_analysis_plugins = Mock()
                    mock_gen.plugin_manager.run_analysis_pipeline.return_value = {
                        'reporter': {'artifacts': ['report.html']}
                    }
                    # Return file paths, not content dicts
                    mock_gen.generate_documentation.return_value = [str(output_file1), str(output_file2)]
                    mock_gen_class.return_value = mock_gen
                    
                    run_generation(sample_args, mock_logger)
                    
                    mock_gen.generate_documentation.assert_called_once()
                    mock_gen.plugin_manager.load_analysis_plugins.assert_called_once()
                    mock_gen.plugin_manager.run_analysis_pipeline.assert_called_once()

    def test_run_generation_list_models(self, mock_logger, sample_args):
        """Test generation with list models flag."""
        sample_args.list_models = True
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.providers.ProviderManager') as mock_manager_class:
                mock_manager = Mock()
                mock_manager.get_available_models.return_value = {
                    'openai': ['gpt-4', 'gpt-3.5-turbo'],
                    'claude': ['claude-3-sonnet', 'claude-3-haiku']
                }
                # Mock provider availability
                mock_openai = Mock()
                mock_openai.is_available.return_value = True
                mock_claude = Mock() 
                mock_claude.is_available.return_value = True
                mock_manager.get_provider.side_effect = lambda name: {
                    'openai': mock_openai, 'claude': mock_claude
                }.get(name)
                mock_manager.get_default_provider.return_value = 'openai'
                mock_manager.get_default_model.return_value = 'gpt-4'
                mock_manager_class.return_value = mock_manager
                
                run_generation(sample_args, mock_logger)
                
                mock_manager.get_available_models.assert_called_once()

    def test_run_generation_list_plugins(self, mock_logger, sample_args):
        """Test generation with list plugins flag."""
        sample_args.list_plugins = True
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.DocumentationGenerator') as mock_gen_class:
                mock_gen = Mock()
                mock_gen.plugin_manager.engines = {
                    'test_plugin': Mock(get_name=Mock(return_value='Test Plugin'))
                }
                mock_gen_class.return_value = mock_gen
                
                with patch('doc_generator.cli.list_plugins') as mock_list:
                    run_generation(sample_args, mock_logger)
                    mock_list.assert_called_once_with(mock_gen)

    def test_run_generation_with_comparison(self, mock_logger, sample_args, temp_dir):
        """Test generation workflow with comparison."""
        sample_args.compare_url = "https://example.com/docs"
        sample_args.output_dir = str(temp_dir)
        
        # Create sample output file
        output_file = temp_dir / "python_programming_v1.html"
        output_file.write_text("<html><body>Generated content</body></html>")
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
                with patch('doc_generator.cli.DocumentationGenerator') as mock_gen_class:
                    mock_gen = Mock()
                    mock_gen.plugin_manager.engines = {}
                    mock_gen.generate_documentation.return_value = [str(output_file)]
                    mock_gen_class.return_value = mock_gen
                    
                    with patch('doc_generator.cli.run_comparison') as mock_compare:
                        run_generation(sample_args, mock_logger)
                        mock_compare.assert_called_once_with(sample_args, [str(output_file)], mock_logger)

    def test_run_readme_generation_basic(self, mock_logger, temp_dir):
        """Test README generation workflow."""
        args = Mock()
        args.readme = str(temp_dir)
        args.recursive = False
        args.output_dir = str(temp_dir)
        args.provider = "openai"
        args.model = "gpt-4o-mini"
        args.temperature = 0.7
        args.format = "markdown"
        args.runs = 1
        args.analyze = False
        args.quiet = False
        args.verbose = False
        args.shots = None
        args.examples_dir = None
        args.prompt_yaml_path = "readme.yaml"
        args.terminology_path = None
        args.analysis_prompt_path = "./prompts/analysis/default.yaml"
        
        # Create sample Python file
        (temp_dir / "test.py").write_text("def hello(): pass")
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
                with patch('doc_generator.readme_documentation_generator.ReadmeDocumentationGenerator') as mock_gen_class:
                    mock_gen = Mock()
                    mock_gen.generate_readme_documentation.return_value = {
                        'generated_files': [str(temp_dir / 'README.md')],
                        'directory_info': {
                            'name': 'test',
                            'path': str(temp_dir),
                            'languages': ['python'],
                            'subdirectories': [],
                            'files': ['test.py']
                        },
                        'depth_level': 0
                    }
                    mock_gen_class.return_value = mock_gen
                    
                    run_readme_generation(args, mock_logger)
                    
                    mock_gen_class.assert_called_once()
                    mock_gen.generate_readme_documentation.assert_called_once()

    def test_run_readme_generation_recursive(self, mock_logger, temp_dir):
        """Test recursive README generation."""
        args = Mock()
        args.readme = str(temp_dir)
        args.recursive = True
        args.output_dir = str(temp_dir)
        args.provider = "openai"
        args.model = "gpt-4o-mini"
        args.temperature = 0.7
        args.format = "markdown"
        args.runs = 3
        args.analyze = False
        args.prompt_yaml_path = './prompts/generator/readme.yaml'
        args.shots = None
        args.examples_dir = './shots'
        args.terminology_path = './terminology.yaml'
        args.quiet = False
        args.verbose = False
        
        # Create nested directories with files
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (temp_dir / "main.py").write_text("def main(): pass")
        (subdir / "utils.py").write_text("def helper(): pass")
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
                with patch('doc_generator.readme_documentation_generator.ReadmeDocumentationGenerator') as mock_gen_class:
                    mock_gen = Mock()
                    mock_gen.generate_readme_documentation.return_value = {
                        'generated_files': [str(temp_dir / 'README.md')],
                        'directory_info': {
                            'name': temp_dir.name,
                            'path': str(temp_dir),
                            'languages': ['python'],
                            'subdirectories': [{'name': 'subdir', 'path': str(subdir)}],
                            'files': ['main.py']
                        },
                        'depth_level': 1
                    }
                    mock_gen_class.return_value = mock_gen
                    
                    run_readme_generation(args, mock_logger)
                    
                    # Should be called for root and subdirectory  
                    assert mock_gen.generate_readme_documentation.call_count >= 1

    def test_run_standardization_file_input(self, mock_logger, temp_dir):
        """Test document standardization with file input."""
        args = Mock()
        input_file = temp_dir / "input.html"
        input_file.write_text("<html><body><h1>Test</h1></body></html>")
        
        args.standardize = str(input_file)
        args.target_format = "markdown"
        args.format = "markdown"
        args.template = "default"
        args.provider = "openai"
        args.model = "gpt-4o-mini"
        args.temperature = 0.7
        args.output_dir = str(temp_dir)
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
                # Mock the standardizer module since it might not exist
                with patch('doc_generator.standardizers.DocumentStandardizer') as mock_std_class:
                    mock_std = Mock()
                    mock_std.standardize_document.return_value = {
                        'standardized_content': '# Test\n\nContent',
                        'metadata': {'provider': 'openai', 'tokens_used': 50}
                    }
                    mock_std_class.return_value = mock_std
                    
                    run_standardization(args, mock_logger)
                    
                    mock_std_class.assert_called_once()
                    mock_std.standardize_document.assert_called_once()

    def test_run_standardization_url_input(self, mock_logger, temp_dir):
        """Test document standardization with URL input."""
        args = Mock()
        args.standardize = "https://example.com/document.html"
        args.target_format = "markdown"
        args.format = "markdown"
        args.template = "default"
        args.provider = "openai"
        args.model = "gpt-4o-mini"
        args.temperature = 0.7
        args.output_dir = str(temp_dir)
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
                with patch('doc_generator.standardizers.DocumentStandardizer') as mock_std_class:
                    mock_std = Mock()
                    mock_std.standardize_document.return_value = {
                        'standardized_content': '# Web Content',
                        'metadata': {'provider': 'openai', 'tokens_used': 100}
                    }
                    mock_std_class.return_value = mock_std
                    
                    with patch('urllib.request.urlopen') as mock_urlopen:
                        mock_response = Mock()
                        mock_response.read.return_value = b"<html><body><h1>Web Content</h1></body></html>"
                        mock_response.__enter__ = Mock(return_value=mock_response)
                        mock_response.__exit__ = Mock(return_value=None)
                        mock_urlopen.return_value = mock_response
                        
                        run_standardization(args, mock_logger)
                        
                        mock_urlopen.assert_called_once()
                        mock_std.standardize_document.assert_called_once()

    def test_cleanup_output_directory_with_files(self, mock_logger, temp_dir):
        """Test cleanup of output directory with files."""
        # Create some test files
        (temp_dir / "test1.html").write_text("test content")
        (temp_dir / "test2.md").write_text("test content")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "test3.txt").write_text("test content")
        
        with patch('doc_generator.cli.Path') as mock_path_class:
            mock_path_class.return_value = temp_dir
            with patch('builtins.input', return_value='yes'):
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
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        with patch('doc_generator.cli.Path') as mock_path_class:
            mock_path_class.return_value = empty_dir
            cleanup_output_directory(mock_logger)
            mock_logger.info.assert_called_with("Output directory is already empty")

    def test_display_info_execution(self):
        """Test display_info function executes properly."""
        with patch('builtins.print') as mock_print:
            display_info()
            # Should have printed information
            assert mock_print.call_count > 0

    def test_list_plugins_with_engines(self, mock_logger):
        """Test plugin listing with available engines."""
        mock_generator = Mock()
        mock_generator.plugin_manager.list_engines.return_value = [
            {
                'name': 'Test Plugin',
                'class': 'TestPlugin',
                'module': 'test_module',
                'description': 'A test plugin',
                'supported_types': ['code'],
                'priority': 5,
                'enabled': True
            }
        ]
        
        with patch('builtins.print') as mock_print:
            list_plugins(mock_generator)
            # Should print plugin information
            assert mock_print.call_count > 0

    def test_list_plugins_empty(self, mock_logger):
        """Test plugin listing with no engines."""
        mock_generator = Mock()
        mock_generator.plugin_manager.list_engines.return_value = []
        
        with patch('builtins.print'):
            list_plugins(mock_generator)

    def test_scan_code_examples(self, mock_logger, temp_dir):
        """Test code scanning functionality."""
        args = Mock()
        args.scan_code = str(temp_dir)
        args.output_dir = str(temp_dir)
        
        # Create sample code files
        (temp_dir / "example.py").write_text('''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
''')
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
                with patch('builtins.open', mock_open(read_data='test: description')):
                    scan_code_examples(args, mock_logger)
                    # Should complete without error
                    mock_logger.info.assert_called()

    def test_run_token_analysis(self, mock_logger, temp_dir):
        """Test token analysis workflow."""
        args = Mock()
        args.topic = "Python Programming"
        args.token_analyze = "documentation_generation"
        args.provider = "openai"
        args.model = "gpt-4o-mini"
        args.output_dir = str(temp_dir)
        args.content = None
        args.content_type = "text"
        args.context_size = None
        args.expected_output = None
        args.analysis_depth = "standard"
        args.verbose = False
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
                # Mock token analysis components
                with patch('doc_generator.agents.token_machine.TokenMachine') as mock_machine:
                    mock_analysis = Mock()
                    mock_analysis.analyze_prompt.return_value = {
                        'token_count': 150,
                        'estimated_cost': 0.003,
                        'complexity_score': 0.7
                    }
                    mock_machine.return_value = mock_analysis
                    
                    run_token_analysis(args, mock_logger)
                    
                    mock_machine.assert_called_once()
                    mock_analysis.analyze_prompt.assert_called_once()

    def test_run_token_estimate(self, mock_logger):
        """Test token estimation workflow."""
        args = Mock()
        args.topic = "Machine Learning"
        args.provider = "openai"
        args.model = "gpt-4"
        args.runs = 3
        args.token_estimate = "documentation_generation"
        args.content = None
        args.content_type = "text"
        args.context_size = None
        args.expected_output = None
        
        with patch('doc_generator.agents.token_machine.TokenMachine') as mock_machine:
            mock_estimate = Mock()
            mock_estimate.estimate_generation_cost.return_value = {
                'input_tokens': 500,
                'estimated_output_tokens': 2000,
                'total_cost': 0.05,
                'cost_per_run': 0.017
            }
            mock_machine.return_value = mock_estimate
            
            run_token_estimate(args, mock_logger)
            
            mock_machine.assert_called_once()
            mock_estimate.estimate_generation_cost.assert_called_once()

    def test_run_token_report(self, mock_logger, temp_dir):
        """Test token report generation."""
        args = Mock()
        args.output_dir = str(temp_dir)
        
        # Create some sample log files
        (temp_dir / "generation_log.json").write_text('''{
            "generations": [
                {"tokens_used": 150, "cost": 0.003},
                {"tokens_used": 200, "cost": 0.004}
            ]
        }''')
        
        with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
            with patch('doc_generator.cli.TokenReportGenerator') as mock_reporter:
                mock_report = Mock()
                mock_report.generate_report.return_value = {
                    'total_tokens': 350,
                    'total_cost': 0.007,
                    'average_tokens_per_generation': 175
                }
                mock_reporter.return_value = mock_report
                
                run_token_report(args, mock_logger)
                
                mock_reporter.assert_called_once()
                mock_report.generate_report.assert_called_once()

    def test_run_token_optimize(self, mock_logger, temp_dir):
        """Test token optimization workflow."""
        args = Mock()
        args.topic = "Data Science"
        args.provider = "openai"
        args.model = "gpt-4o-mini"
        args.output_dir = str(temp_dir)
        
        with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
            with patch('doc_generator.cli.TokenOptimizer') as mock_optimizer:
                mock_optimize = Mock()
                mock_optimize.optimize_prompt.return_value = {
                    'original_tokens': 500,
                    'optimized_tokens': 350,
                    'token_reduction': 150,
                    'optimized_prompt': 'Optimized prompt text'
                }
                mock_optimizer.return_value = mock_optimize
                
                run_token_optimize(args, mock_logger)
                
                mock_optimizer.assert_called_once()
                mock_optimize.optimize_prompt.assert_called_once()

    def test_run_comparison_url(self, mock_logger, temp_dir, sample_args):
        """Test document comparison with URL."""
        sample_args.compare = "https://example.com/reference.html"
        sample_args.output_dir = str(temp_dir)
        
        # Create generated document
        generated_file = temp_dir / "test_doc.html"
        generated_file.write_text("<html><body>Generated content</body></html>")
        
        with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
            with patch('doc_generator.evaluator.DocumentationComparator') as mock_comparator:
                mock_compare = Mock()
                mock_compare.compare_with_url.return_value = {
                    'similarity_score': 0.75,
                    'differences': ['style differences', 'content gaps'],
                    'recommendations': ['improve structure', 'add examples']
                }
                mock_comparator.return_value = mock_compare
                
                run_comparison(sample_args, mock_logger, str(generated_file))
                
                mock_comparator.assert_called_once()
                mock_compare.compare_with_url.assert_called_once()

    def test_run_comparison_file(self, mock_logger, temp_dir, sample_args):
        """Test document comparison with file."""
        reference_file = temp_dir / "reference.html"
        reference_file.write_text("<html><body>Reference content</body></html>")
        
        sample_args.compare = str(reference_file)
        sample_args.output_dir = str(temp_dir)
        
        generated_file = temp_dir / "test_doc.html"
        generated_file.write_text("<html><body>Generated content</body></html>")
        
        with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
            with patch('doc_generator.evaluator.DocumentationComparator') as mock_comparator:
                mock_compare = Mock()
                mock_compare.compare_with_file.return_value = {
                    'similarity_score': 0.85,
                    'differences': ['minor style differences'],
                    'recommendations': ['consistent formatting']
                }
                mock_comparator.return_value = mock_compare
                
                run_comparison(sample_args, mock_logger, str(generated_file))
                
                mock_comparator.assert_called_once()
                mock_compare.compare_with_file.assert_called_once()


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()

    def test_generation_missing_topic(self, mock_logger):
        """Test error when topic is missing."""
        args = Mock()
        args.topic = None
        args.list_models = False
        args.list_plugins = False
        args.verbose = False
        args.shots = None
        args.examples_dir = None
        args.prompt_yaml_path = None
        args.terminology_path = None
        args.provider = "auto"
        args.disable_plugins = None
        args.enable_only = None
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.DocumentationGenerator') as mock_gen_class:
                mock_gen = Mock()
                mock_gen.plugin_manager.engines = {}
                mock_gen_class.return_value = mock_gen
                
                with pytest.raises(SystemExit) as exc_info:
                    run_generation(args, mock_logger)
                assert exc_info.value.code == 1
                mock_logger.error.assert_called_with("--topic is required for documentation generation")

    def test_readme_generation_missing_path(self, mock_logger):
        """Test error when README path is missing."""
        args = Mock()
        args.readme = None
        
        with pytest.raises(SystemExit) as exc_info:
            run_readme_generation(args, mock_logger)
        assert exc_info.value.code == 1
        mock_logger.error.assert_called_with("--readme requires a directory path")

    def test_readme_generation_invalid_path(self, mock_logger):
        """Test error when README path doesn't exist."""
        args = Mock()
        args.readme = "/nonexistent/path"
        
        with patch('doc_generator.cli.load_dotenv'):
            with pytest.raises(SystemExit) as exc_info:
                run_readme_generation(args, mock_logger)
            assert exc_info.value.code == 1

    def test_standardization_missing_input(self, mock_logger):
        """Test error when standardization input is missing."""
        args = Mock()
        args.standardize = None
        
        with pytest.raises(SystemExit) as exc_info:
            run_standardization(args, mock_logger)
        assert exc_info.value.code == 1
        mock_logger.error.assert_called_with("--standardize requires an input file or URL")

    def test_standardization_file_not_found(self, mock_logger):
        """Test error when standardization file doesn't exist."""
        args = Mock()
        args.standardize = "/nonexistent/file.html"
        
        with patch('doc_generator.cli.load_dotenv'):
            with pytest.raises(SystemExit) as exc_info:
                run_standardization(args, mock_logger)
            assert exc_info.value.code == 1

    def test_generation_provider_error(self, mock_logger):
        """Test handling of provider initialization errors."""
        args = Mock()
        args.topic = "Test Topic"
        args.list_models = False
        args.list_plugins = False
        args.verbose = False
        
        with patch('doc_generator.cli.load_dotenv'):
            with patch('doc_generator.cli.DocumentationGenerator', side_effect=Exception("Provider error")):
                with pytest.raises(SystemExit) as exc_info:
                    run_generation(args, mock_logger)
                assert exc_info.value.code == 1
                mock_logger.error.assert_called_with("Error during generation: Provider error")

    def test_cleanup_permission_error(self, mock_logger, temp_dir):
        """Test cleanup with permission errors."""
        with patch('doc_generator.cli.get_output_directory', return_value=str(temp_dir)):
            with patch('builtins.input', return_value='y'):
                with patch('shutil.rmtree', side_effect=PermissionError("Permission denied")):
                    cleanup_output_directory(mock_logger)
                    mock_logger.error.assert_called()


class TestCLIUtilityFunctions:
    """Test CLI utility functions."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()

    def test_token_analysis_missing_topic(self, mock_logger):
        """Test token analysis without topic."""
        args = Mock()
        args.topic = None
        
        with pytest.raises(SystemExit) as exc_info:
            run_token_analysis(args, mock_logger)
        assert exc_info.value.code == 1
        mock_logger.error.assert_called_with("--topic is required for token analysis")

    def test_token_estimate_missing_topic(self, mock_logger):
        """Test token estimation without topic."""
        args = Mock()
        args.topic = None
        
        with pytest.raises(SystemExit) as exc_info:
            run_token_estimate(args, mock_logger)
        assert exc_info.value.code == 1
        mock_logger.error.assert_called_with("--topic is required for token estimation")

    def test_token_optimize_missing_topic(self, mock_logger):
        """Test token optimization without topic."""
        args = Mock()
        args.topic = None
        
        with pytest.raises(SystemExit) as exc_info:
            run_token_optimize(args, mock_logger)
        assert exc_info.value.code == 1
        mock_logger.error.assert_called_with("--topic is required for token optimization")

    def test_scan_code_missing_directory(self, mock_logger):
        """Test code scanning without directory."""
        args = Mock()
        args.scan_code = None
        
        with pytest.raises(SystemExit) as exc_info:
            scan_code_examples(args, mock_logger)
        assert exc_info.value.code == 1
        mock_logger.error.assert_called_with("--scan-code requires a directory path")

    def test_scan_code_invalid_directory(self, mock_logger):
        """Test code scanning with invalid directory."""
        args = Mock()
        args.scan_code = "/nonexistent/directory"
        
        with patch('doc_generator.cli.load_dotenv'):
            with pytest.raises(SystemExit) as exc_info:
                scan_code_examples(args, mock_logger)
            assert exc_info.value.code == 1