"""
Tests for the new CLI command pattern system.

Tests both the new command-based interface and backward compatibility
with the original CLI argument parsing.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from pathlib import Path
import tempfile
import shutil

from src.doc_generator.cli_commands.registry import CommandRegistry
from src.doc_generator.cli_commands.dispatcher import CommandDispatcher
from src.doc_generator.cli_commands.bootstrap import create_default_registry
from src.doc_generator.cli_commands.base import BaseCommand, CommandResult
from src.doc_generator.cli_commands.commands.generate_command import GenerateCommand
from src.doc_generator.cli_commands.commands.readme_command import ReadmeCommand
from src.doc_generator.cli_commands.commands.standardize_command import StandardizeCommand
from src.doc_generator.cli_commands.commands.utility_commands import (
    ListModelsCommand, CleanupCommand, InfoCommand, ListPluginsCommand
)


class TestCommandRegistry:
    """Test command registry functionality."""
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        registry = CommandRegistry()
        assert registry._commands == {}
        assert registry._aliases == {}
    
    def test_command_registration(self):
        """Test registering commands."""
        registry = CommandRegistry()
        
        # Test registering a command
        registry.register(GenerateCommand)
        
        # Check command is registered
        assert 'generate' in registry._commands
        assert registry._commands['generate'] == GenerateCommand
        
        # Check aliases are registered
        assert 'gen' in registry._aliases
        assert 'g' in registry._aliases
        assert registry._aliases['gen'] == 'generate'
        assert registry._aliases['g'] == 'generate'
    
    def test_command_lookup(self):
        """Test looking up commands by name and alias."""
        registry = CommandRegistry()
        registry.register(GenerateCommand)
        
        # Test lookup by name
        assert registry.get_command('generate') == GenerateCommand
        
        # Test lookup by alias
        assert registry.get_command('gen') == GenerateCommand
        assert registry.get_command('g') == GenerateCommand
        
        # Test non-existent command
        assert registry.get_command('nonexistent') is None
    
    def test_list_commands(self):
        """Test listing commands."""
        registry = CommandRegistry()
        registry.register(GenerateCommand)
        registry.register(ReadmeCommand)
        
        commands = registry.list_commands()
        assert 'generate' in commands
        assert 'readme' in commands
        assert len(commands) == 2
    
    def test_command_validation(self):
        """Test command name validation."""
        registry = CommandRegistry()
        
        # Valid names should work
        assert registry.validate_command_name('test-command')
        assert registry.validate_command_name('test_command')
        assert registry.validate_command_name('testcommand')
        
        # Invalid names should fail
        assert not registry.validate_command_name('')
        assert not registry.validate_command_name('test@command')
        assert not registry.validate_command_name('test command')  # spaces not allowed


class TestCommandDispatcher:
    """Test command dispatcher functionality."""
    
    @pytest.fixture
    def registry(self):
        """Create a test registry with commands."""
        registry = CommandRegistry()
        registry.register(GenerateCommand)
        registry.register(ListModelsCommand)
        return registry
    
    def test_dispatcher_initialization(self, registry):
        """Test dispatcher initialization."""
        dispatcher = CommandDispatcher(registry)
        assert dispatcher.registry == registry
        assert dispatcher.logger is not None
    
    def test_help_generation(self, registry):
        """Test help generation."""
        dispatcher = CommandDispatcher(registry)
        parser = dispatcher.create_parser()
        
        # Should have global arguments
        assert parser.prog == "doc-gen"
        
        # Should have subcommands
        assert hasattr(parser, '_subparsers')
    
    def test_command_dispatch_success(self, registry):
        """Test successful command dispatch."""
        dispatcher = CommandDispatcher(registry)
        
        # Mock the list-models command
        with patch('src.doc_generator.providers.ProviderManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_available_models.return_value = {
                'openai': ['gpt-4o-mini']
            }
            mock_manager.get_provider.return_value = Mock(is_available=lambda: True)
            mock_manager.get_default_provider.return_value = 'openai'
            mock_manager.get_default_model.return_value = 'gpt-4o-mini'
            mock_manager_class.return_value = mock_manager
            
            with patch('builtins.print'):  # Suppress output
                result = dispatcher.dispatch(['list-models'])
                assert result.success
                assert result.exit_code == 0
    
    def test_command_dispatch_failure(self, registry):
        """Test command dispatch with invalid command."""
        dispatcher = CommandDispatcher(registry)
        
        result = dispatcher.dispatch(['nonexistent-command'])
        assert not result.success
        assert result.exit_code == 2  # argparse returns 2 for invalid choice
        assert "Argument parsing failed" in result.message
    
    def test_empty_command(self, registry):
        """Test dispatch with no command."""
        dispatcher = CommandDispatcher(registry)
        
        with patch('builtins.print'):  # Suppress help output
            result = dispatcher.dispatch([])
            assert result.success  # Help display is considered success
            assert result.exit_code == 0


class TestCommands:
    """Test individual command implementations."""
    
    def test_generate_command_basic(self):
        """Test basic GenerateCommand functionality."""
        cmd = GenerateCommand()
        
        assert cmd.name == "generate"
        assert cmd.description is not None
        assert "gen" in cmd.aliases
        assert "g" in cmd.aliases
    
    def test_readme_command_basic(self):
        """Test basic ReadmeCommand functionality."""
        cmd = ReadmeCommand()
        
        assert cmd.name == "readme"
        assert cmd.description is not None
        assert "r" in cmd.aliases
    
    def test_standardize_command_basic(self):
        """Test basic StandardizeCommand functionality."""
        cmd = StandardizeCommand()
        
        assert cmd.name == "standardize"
        assert cmd.description is not None
        assert "std" in cmd.aliases
        assert "s" in cmd.aliases
    
    def test_list_models_command_basic(self):
        """Test basic ListModelsCommand functionality."""
        cmd = ListModelsCommand()
        
        assert cmd.name == "list-models"
        assert cmd.description is not None
        assert "lm" in cmd.aliases
        assert "models" in cmd.aliases
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_generate_command_validation(self, temp_dir):
        """Test GenerateCommand argument validation."""
        cmd = GenerateCommand()
        
        # Create a mock namespace with valid arguments
        from argparse import Namespace
        
        # Valid arguments should pass
        args = Namespace(
            topic="Test Topic",
            temperature=0.5,
            runs=3,
            list_models=False,
            list_plugins=False,
            prompt_yaml_path=None,
            quality_eval=False,
            analysis_prompt_path=None,
            compare_file=None
        )
        
        # Should not raise an exception
        try:
            cmd.validate_args(args)
            validation_passed = True
        except Exception:
            validation_passed = False
        assert validation_passed
        
        # Invalid temperature should fail
        args.temperature = 3.0
        with pytest.raises(Exception):
            cmd.validate_args(args)
    
    def test_readme_command_validation(self, temp_dir):
        """Test ReadmeCommand argument validation."""
        cmd = ReadmeCommand()
        
        # Create test directory
        test_dir = temp_dir / "test_project"
        test_dir.mkdir()
        
        from argparse import Namespace
        args = Namespace(
            directory=str(test_dir),
            temperature=0.3,
            runs=2,
            prompt_yaml_path=None,
            analyze=False,
            analysis_prompt_path=None
        )
        
        # Should pass with valid directory
        try:
            cmd.validate_args(args)
            validation_passed = True
        except Exception:
            validation_passed = False
        assert validation_passed
        
        # Should fail with non-existent directory
        args.directory = str(temp_dir / "nonexistent")
        with pytest.raises(Exception):
            cmd.validate_args(args)


class TestBootstrap:
    """Test the bootstrap system."""
    
    def test_create_default_registry(self):
        """Test creating default registry with all commands."""
        registry = create_default_registry()
        
        # Check all expected commands are registered
        commands = registry.list_commands(include_hidden=True)
        
        expected_commands = [
            'generate', 'readme', 'standardize', 
            'list-models', 'cleanup', 'info', 'list-plugins', 'test'
        ]
        
        for cmd_name in expected_commands:
            assert cmd_name in commands, f"Command {cmd_name} not found in registry"
    
    def test_command_aliases_registered(self):
        """Test that command aliases are properly registered."""
        registry = create_default_registry()
        
        # Test some key aliases
        assert registry.get_command('gen') == GenerateCommand
        assert registry.get_command('g') == GenerateCommand
        assert registry.get_command('r') == ReadmeCommand
        assert registry.get_command('std') == StandardizeCommand
        assert registry.get_command('lm') == ListModelsCommand


class TestBackwardCompatibility:
    """Test backward compatibility with old CLI."""
    
    def test_old_style_arguments_still_work(self):
        """Test that old-style --flag arguments still work."""
        from src.doc_generator.cli import parse_args
        
        # Test old-style topic argument
        with patch('sys.argv', ['doc-gen', '--topic', 'Test Topic']):
            args = parse_args()
            assert args.topic == 'Test Topic'
        
        # Test old-style readme argument
        with patch('sys.argv', ['doc-gen', '--readme', '/test/path']):
            args = parse_args()
            assert args.readme == '/test/path'
        
        # Test old-style standardize argument
        with patch('sys.argv', ['doc-gen', '--standardize', 'test.html']):
            args = parse_args()
            assert args.standardize == 'test.html'
    
    def test_new_command_detection(self):
        """Test that new command format is detected correctly."""
        from src.doc_generator.cli import main
        
        # Mock sys.argv for new command format
        with patch('sys.argv', ['doc-gen', 'generate', 'Test Topic']):
            with patch('src.doc_generator.cli_commands.main.main') as mock_new_main:
                mock_new_main.return_value = 0
                
                # Should detect new format and call new main
                try:
                    main()
                except SystemExit:
                    pass  # Expected from sys.exit()
                
                mock_new_main.assert_called_once()
    
    def test_legacy_format_handling(self):
        """Test that legacy --flag format goes to legacy handler."""
        from src.doc_generator.cli import main
        
        # Mock sys.argv for old command format
        with patch('sys.argv', ['doc-gen', '--list-models']):
            with patch('src.doc_generator.cli.parse_args') as mock_parse:
                with patch('src.doc_generator.cli.setup_logging') as mock_logging:
                    with patch('src.doc_generator.providers.ProviderManager'):
                        with patch('builtins.print'):  # Suppress output
                            mock_args = Mock()
                            mock_args.list_models = True
                            mock_args.verbose = False
                            mock_parse.return_value = mock_args
                            mock_logging.return_value = Mock()
                            
                            try:
                                main()
                            except SystemExit:
                                pass  # Expected
                            
                            # Should have called legacy parsing
                            mock_parse.assert_called_once()


class TestCommandIntegration:
    """Integration tests for command system."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temp output directory for cleanup tests."""
        output_dir = Path("./output_test")
        output_dir.mkdir(exist_ok=True)
        (output_dir / "test_file.txt").write_text("test content")
        (output_dir / "test_dir").mkdir(exist_ok=True)
        yield output_dir
        shutil.rmtree(output_dir, ignore_errors=True)
    
    def test_cleanup_command_with_force(self, temp_output_dir):
        """Test cleanup command with force flag."""
        from src.doc_generator.cli_commands.main import main
        
        # Change to force cleanup of test directory
        with patch('src.doc_generator.cli_commands.commands.utility_commands.Path') as mock_path:
            # Create mock files that can be sorted
            mock_file = Mock()
            mock_file.name = "test_file.txt"
            mock_file.is_file.return_value = True
            mock_file.is_dir.return_value = False
            mock_file.unlink = Mock()
            mock_file.__lt__ = Mock(return_value=True)  # For sorting
            
            mock_dir = Mock()
            mock_dir.name = "test_dir"
            mock_dir.is_file.return_value = False
            mock_dir.is_dir.return_value = True
            mock_dir.__lt__ = Mock(return_value=False)  # For sorting
            
            mock_output = Mock()
            mock_output.exists.return_value = True
            mock_output.iterdir.return_value = [mock_file, mock_dir]
            mock_path.return_value = mock_output
            
            with patch('shutil.rmtree'):
                with patch('builtins.print'):  # Suppress output
                    result = main(['cleanup', '--force'])
                    assert result == 0
    
    def test_info_command_execution(self):
        """Test that info command executes successfully."""
        from src.doc_generator.cli_commands.main import main
        
        with patch('builtins.print'):  # Suppress output
            result = main(['info'])
            assert result == 0
    
    def test_list_models_integration(self):
        """Test list-models command integration."""
        from src.doc_generator.cli_commands.main import main
        
        with patch('src.doc_generator.providers.ProviderManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_available_models.return_value = {'openai': ['gpt-4']}
            mock_manager.get_provider.return_value = Mock(is_available=lambda: True)
            mock_manager.get_default_provider.return_value = 'openai'
            mock_manager.get_default_model.return_value = 'gpt-4'
            mock_manager_class.return_value = mock_manager
            
            with patch('builtins.print'):  # Suppress output
                result = main(['list-models'])
                assert result == 0
    
    def test_command_help_system(self):
        """Test that command help system works."""
        from src.doc_generator.cli_commands.main import main
        
        # Test main help
        with patch('builtins.print'):
            result = main(['--help'])
            assert result == 0
        
        # Test command-specific help
        with patch('builtins.print'):
            result = main(['generate', '--help'])
            assert result == 0


class TestErrorHandling:
    """Test error handling in command system."""
    
    def test_invalid_command_error(self):
        """Test error handling for invalid commands."""
        from src.doc_generator.cli_commands.main import main
        
        result = main(['nonexistent-command'])
        assert result != 0  # Should fail
    
    def test_command_validation_errors(self):
        """Test command validation error handling."""
        from src.doc_generator.cli_commands.main import main
        
        # Test generate command with invalid temperature
        result = main(['generate', 'Test Topic', '--temperature', '5.0'])
        assert result != 0  # Should fail validation
    
    def test_missing_required_args(self):
        """Test missing required arguments."""
        from src.doc_generator.cli_commands.main import main
        
        # Test generate without topic
        result = main(['generate'])
        assert result != 0  # Should fail


if __name__ == '__main__':
    pytest.main([__file__])