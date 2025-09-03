"""
Base Command Abstract Class

Defines the interface and common patterns for all CLI commands.
Inspired by Command Pattern and modern CLI frameworks.
"""

import logging
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ..config import get_settings
from ..exceptions import DocGeneratorError


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    message: str
    exit_code: int = 0
    data: Optional[Dict[str, Any]] = None


class BaseCommand(ABC):
    """
    Abstract base class for all CLI commands.
    
    Implements common patterns like:
    - Consistent error handling
    - Logging integration  
    - Settings access
    - Argument validation
    - User confirmation utilities
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize command.
        
        Args:
            logger: Optional logger instance. If not provided, creates one.
        """
        self.logger = logger or logging.getLogger(f"doc_generator.cli_commands.{self.get_name()}")
        self.settings = get_settings()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Command name (e.g., 'generate', 'analyze')."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what this command does."""
        pass
    
    @property
    def aliases(self) -> List[str]:
        """Alternative names for this command."""
        return []
    
    @property
    def hidden(self) -> bool:
        """Whether this command should be hidden from help."""
        return False
    
    @abstractmethod
    def add_arguments(self, parser: ArgumentParser) -> None:
        """
        Add command-specific arguments to the parser.
        
        Args:
            parser: ArgumentParser to add arguments to
        """
        pass
    
    @abstractmethod
    def run(self, args: Namespace) -> CommandResult:
        """
        Execute the command.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            CommandResult indicating success/failure and any data
        """
        pass
    
    def validate_args(self, args: Namespace) -> None:
        """
        Validate command-line arguments.
        
        Args:
            args: Parsed arguments to validate
            
        Raises:
            DocGeneratorError: If arguments are invalid
        """
        # Base implementation does nothing - override in subclasses
        pass
    
    def get_name(self) -> str:
        """Get command name (convenience method)."""
        return self.name
    
    def requires_confirmation(self, message: str, default: bool = False) -> bool:
        """
        Ask user for confirmation.
        
        Args:
            message: Confirmation message to display
            default: Default response if user just presses enter
            
        Returns:
            True if user confirmed, False otherwise
        """
        if getattr(self.settings, 'quiet', False):
            return default
            
        suffix = " [Y/n]" if default else " [y/N]"
        prompt = f"{message}{suffix}: "
        
        try:
            response = input(prompt).strip().lower()
            if not response:
                return default
            return response.startswith('y')
        except (KeyboardInterrupt, EOFError):
            print()  # New line after ^C
            return False
    
    def validate_path(self, path: Union[str, Path], must_exist: bool = True, 
                     must_be_file: bool = False, must_be_dir: bool = False) -> Path:
        """
        Validate and return a Path object.
        
        Args:
            path: Path to validate
            must_exist: Whether path must exist
            must_be_file: Whether path must be a file
            must_be_dir: Whether path must be a directory
            
        Returns:
            Validated Path object
            
        Raises:
            DocGeneratorError: If validation fails
        """
        path_obj = Path(path)
        
        if must_exist and not path_obj.exists():
            raise DocGeneratorError(f"Path does not exist: {path_obj}")
        
        if must_be_file and path_obj.exists() and not path_obj.is_file():
            raise DocGeneratorError(f"Path is not a file: {path_obj}")
            
        if must_be_dir and path_obj.exists() and not path_obj.is_dir():
            raise DocGeneratorError(f"Path is not a directory: {path_obj}")
            
        return path_obj
    
    def handle_error(self, error: Exception, context: str = "") -> CommandResult:
        """
        Handle command execution errors consistently.
        
        Args:
            error: Exception that occurred
            context: Additional context about where error occurred
            
        Returns:
            CommandResult indicating failure
        """
        if isinstance(error, DocGeneratorError):
            message = str(error)
            if error.context:
                self.logger.debug(f"Error context: {error.context}")
        else:
            message = f"Unexpected error: {error}"
            
        if context:
            message = f"{context}: {message}"
            
        self.logger.error(message)
        
        # Show stack trace in verbose mode
        if hasattr(self.settings, 'verbose') and self.settings.verbose:
            import traceback
            self.logger.debug(traceback.format_exc())
            
        return CommandResult(
            success=False,
            message=message,
            exit_code=1
        )
    
    def success(self, message: str, data: Optional[Dict[str, Any]] = None) -> CommandResult:
        """
        Create a success result.
        
        Args:
            message: Success message
            data: Optional data to include
            
        Returns:
            CommandResult indicating success
        """
        self.logger.info(message)
        return CommandResult(
            success=True,
            message=message,
            exit_code=0,
            data=data
        )