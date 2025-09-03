"""
CLI Commands Bootstrap

Registers all built-in commands with the command registry.
"""

import logging
from typing import Optional

from .registry import CommandRegistry
from .commands.generate_command import GenerateCommand
from .commands.readme_command import ReadmeCommand
from .commands.standardize_command import StandardizeCommand
from .commands.test_command import TestCommand
from .commands.utility_commands import (
    ListModelsCommand,
    CleanupCommand,
    InfoCommand,
    ListPluginsCommand
)


def create_default_registry(logger: Optional[logging.Logger] = None) -> CommandRegistry:
    """
    Create a command registry with all default commands registered.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        Configured CommandRegistry instance
    """
    registry = CommandRegistry(logger=logger)
    
    # Register all built-in commands
    registry.register(GenerateCommand)
    registry.register(ReadmeCommand)
    registry.register(StandardizeCommand)
    registry.register(TestCommand)
    
    # Register utility commands
    registry.register(ListModelsCommand)
    registry.register(CleanupCommand)
    registry.register(InfoCommand)
    registry.register(ListPluginsCommand)
    
    # Try to load additional commands from entry points
    registry.register_from_entry_points("doc_generator.commands")
    
    return registry


def get_available_commands(logger: Optional[logging.Logger] = None, include_hidden: bool = False) -> dict:
    """
    Get all available commands as a dictionary.
    
    Args:
        logger: Optional logger instance
        include_hidden: Whether to include hidden commands
        
    Returns:
        Dictionary of command_name -> command_class
    """
    registry = create_default_registry(logger)
    return registry.list_commands(include_hidden=include_hidden)