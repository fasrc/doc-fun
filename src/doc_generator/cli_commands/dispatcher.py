"""
Command Dispatcher

Handles command routing and execution with consistent error handling
and help generation.
"""

import logging
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Optional

from .base import BaseCommand, CommandResult
from .registry import CommandRegistry


class CommandDispatcher:
    """
    Dispatches commands based on parsed arguments.
    
    Handles:
    - Command routing
    - Error handling
    - Help generation
    - Argument validation
    """
    
    def __init__(self, registry: CommandRegistry, logger: Optional[logging.Logger] = None):
        """
        Initialize command dispatcher.
        
        Args:
            registry: Command registry to use for command lookup
            logger: Optional logger instance
        """
        self.registry = registry
        self.logger = logger or logging.getLogger(__name__)
    
    def create_parser(self, prog: str = "doc-gen", description: str = None, 
                     include_hidden: bool = False) -> ArgumentParser:
        """
        Create main argument parser with subcommands.
        
        Args:
            prog: Program name
            description: Program description
            include_hidden: Whether to include hidden commands in help
            
        Returns:
            Configured ArgumentParser
        """
        if description is None:
            description = (
                "Generate technical documentation using AI with plugin-based recommendations. "
                "Commands implement specific functionality like generation, analysis, or standardization."
            )
        
        parser = ArgumentParser(
            prog=prog,
            description=description,
            epilog="Use '<command> --help' for help on specific commands."
        )
        
        # Add global arguments
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose logging')
        parser.add_argument('--quiet', '-q', action='store_true',
                          help='Suppress non-essential output')
        parser.add_argument('--version', action='version', 
                          version='%(prog)s 2.6.0-dev')
        
        # Create subcommands
        subparsers = parser.add_subparsers(
            dest='command',
            metavar='<command>',
            help='Available commands'
        )
        
        # Register all commands as subcommands
        commands = self.registry.list_commands(include_hidden=include_hidden)
        for cmd_name, cmd_class in commands.items():
            try:
                cmd_instance = cmd_class()
                
                # Create subparser for this command
                cmd_parser = subparsers.add_parser(
                    cmd_name,
                    help=cmd_instance.description,
                    description=cmd_instance.description,
                    aliases=cmd_instance.aliases
                )
                
                # Let command add its specific arguments
                cmd_instance.add_arguments(cmd_parser)
                
                self.logger.debug(f"Added subcommand: {cmd_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to add subcommand '{cmd_name}': {e}")
        
        return parser
    
    def dispatch(self, args: List[str]) -> CommandResult:
        """
        Parse arguments and dispatch to appropriate command.
        
        Args:
            args: Command-line arguments (typically sys.argv[1:])
            
        Returns:
            CommandResult indicating success/failure
        """
        parser = self.create_parser(include_hidden=True)
        
        try:
            parsed_args = parser.parse_args(args)
            
            # If no command specified, show help
            if not hasattr(parsed_args, 'command') or parsed_args.command is None:
                parser.print_help()
                return CommandResult(
                    success=True,
                    message="No command specified",
                    exit_code=0
                )
            
            # Get command class
            cmd_class = self.registry.get_command(parsed_args.command)
            if cmd_class is None:
                return CommandResult(
                    success=False,
                    message=f"Unknown command: {parsed_args.command}",
                    exit_code=1
                )
            
            # Create command instance
            command = cmd_class(logger=self.logger)
            
            # Validate arguments
            try:
                command.validate_args(parsed_args)
            except Exception as e:
                return CommandResult(
                    success=False,
                    message=f"Argument validation failed: {e}",
                    exit_code=1
                )
            
            # Execute command
            self.logger.debug(f"Executing command: {parsed_args.command}")
            return command.run(parsed_args)
            
        except SystemExit as e:
            # argparse calls sys.exit() for help/version/errors
            # Convert to CommandResult
            if e.code == 0:
                return CommandResult(success=True, message="Help or version displayed", exit_code=0)
            else:
                return CommandResult(success=False, message="Argument parsing failed", exit_code=e.code)
                
        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user")
            return CommandResult(
                success=False,
                message="Operation cancelled by user",
                exit_code=1
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error during dispatch: {e}")
            if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
                import traceback
                self.logger.debug(traceback.format_exc())
            return CommandResult(
                success=False,
                message=f"Unexpected error: {e}",
                exit_code=1
            )
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run dispatcher and return exit code.
        
        Args:
            args: Command-line arguments. If None, uses sys.argv[1:]
            
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        if args is None:
            args = sys.argv[1:]
            
        result = self.dispatch(args)
        
        if result.success:
            if result.message and not getattr(sys.modules.get('__main__'), '_quiet', False):
                print(result.message)
        else:
            print(f"Error: {result.message}", file=sys.stderr)
            
        return result.exit_code
    
    def list_available_commands(self) -> List[str]:
        """
        Get list of available command names.
        
        Returns:
            List of command names
        """
        return self.registry.get_command_names(include_hidden=False)
    
    def get_command_help(self, command_name: str) -> Optional[str]:
        """
        Get help text for a specific command.
        
        Args:
            command_name: Name of the command
            
        Returns:
            Help text if command exists, None otherwise
        """
        cmd_class = self.registry.get_command(command_name)
        if cmd_class is None:
            return None
            
        try:
            command = cmd_class()
            return command.description
        except Exception:
            return f"Command '{command_name}' (unable to get description)"