"""
Command Registry

Manages registration and discovery of CLI commands.
Supports both explicit registration and entry point discovery.
"""

import logging
from typing import Dict, List, Optional, Type, Set
from importlib.metadata import entry_points

from .base import BaseCommand


class CommandRegistry:
    """
    Registry for CLI commands.
    
    Supports:
    - Manual command registration
    - Entry point discovery
    - Command lookup by name/alias
    - Command validation
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize command registry.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._commands: Dict[str, Type[BaseCommand]] = {}
        self._aliases: Dict[str, str] = {}  # alias -> command_name
        
    def register(self, command_class: Type[BaseCommand], name: Optional[str] = None) -> None:
        """
        Register a command class.
        
        Args:
            command_class: Command class to register
            name: Optional override for command name
            
        Raises:
            ValueError: If command name conflicts or is invalid
        """
        if not issubclass(command_class, BaseCommand):
            raise ValueError(f"Command must inherit from BaseCommand: {command_class}")
        
        # Create instance to get metadata
        try:
            instance = command_class()
            cmd_name = name or instance.name
            
            if not cmd_name or not cmd_name.replace('-', '').replace('_', '').isalnum():
                raise ValueError(f"Invalid command name: '{cmd_name}' (must be alphanumeric with dashes/underscores)")
                
            if cmd_name in self._commands:
                existing_class = self._commands[cmd_name]
                raise ValueError(
                    f"Command '{cmd_name}' already registered by {existing_class.__module__}.{existing_class.__name__}"
                )
            
            # Register main command
            self._commands[cmd_name] = command_class
            self.logger.debug(f"Registered command: {cmd_name} -> {command_class.__name__}")
            
            # Register aliases
            for alias in instance.aliases:
                if not alias or not alias.replace('-', '').replace('_', '').isalnum():
                    self.logger.warning(f"Skipping invalid alias '{alias}' for command '{cmd_name}'")
                    continue
                    
                if alias in self._aliases:
                    existing_cmd = self._aliases[alias]
                    self.logger.warning(
                        f"Alias '{alias}' already registered for command '{existing_cmd}', "
                        f"skipping for '{cmd_name}'"
                    )
                    continue
                    
                self._aliases[alias] = cmd_name
                self.logger.debug(f"Registered alias: {alias} -> {cmd_name}")
                
        except Exception as e:
            raise ValueError(f"Failed to register command {command_class}: {e}")
    
    def register_from_entry_points(self, entry_point_group: str = "doc_generator.commands") -> None:
        """
        Register commands from entry points.
        
        Args:
            entry_point_group: Entry point group name
        """
        self.logger.debug(f"Loading commands from entry point group: {entry_point_group}")
        
        try:
            eps = entry_points(group=entry_point_group)
            for ep in eps:
                try:
                    command_class = ep.load()
                    self.register(command_class, ep.name)
                    self.logger.info(f"Loaded command from entry point: {ep.name} -> {command_class.__name__}")
                except Exception as e:
                    self.logger.error(f"Failed to load command from entry point '{ep.name}': {e}")
                    
        except Exception as e:
            self.logger.debug(f"No entry points found for group '{entry_point_group}': {e}")
    
    def get_command(self, name: str) -> Optional[Type[BaseCommand]]:
        """
        Get command class by name or alias.
        
        Args:
            name: Command name or alias
            
        Returns:
            Command class if found, None otherwise
        """
        # Check direct command name
        if name in self._commands:
            return self._commands[name]
            
        # Check aliases
        if name in self._aliases:
            cmd_name = self._aliases[name]
            return self._commands.get(cmd_name)
            
        return None
    
    def list_commands(self, include_hidden: bool = False) -> Dict[str, Type[BaseCommand]]:
        """
        List all registered commands.
        
        Args:
            include_hidden: Whether to include hidden commands
            
        Returns:
            Dictionary of command_name -> command_class
        """
        if include_hidden:
            return self._commands.copy()
            
        visible_commands = {}
        for name, cmd_class in self._commands.items():
            try:
                instance = cmd_class()
                if not instance.hidden:
                    visible_commands[name] = cmd_class
            except Exception as e:
                self.logger.warning(f"Error checking if command '{name}' is hidden: {e}")
                # Include by default if we can't determine
                visible_commands[name] = cmd_class
                
        return visible_commands
    
    def get_command_names(self, include_hidden: bool = False) -> List[str]:
        """
        Get list of command names.
        
        Args:
            include_hidden: Whether to include hidden commands
            
        Returns:
            List of command names
        """
        return list(self.list_commands(include_hidden).keys())
    
    def get_aliases(self, command_name: str) -> List[str]:
        """
        Get aliases for a command.
        
        Args:
            command_name: Name of the command
            
        Returns:
            List of aliases for the command
        """
        return [alias for alias, cmd_name in self._aliases.items() if cmd_name == command_name]
    
    def get_all_names(self, include_hidden: bool = False) -> Set[str]:
        """
        Get all names (commands + aliases).
        
        Args:
            include_hidden: Whether to include hidden commands
            
        Returns:
            Set of all command names and aliases
        """
        names = set(self.get_command_names(include_hidden))
        
        # Add aliases for visible commands
        for alias, cmd_name in self._aliases.items():
            if include_hidden or cmd_name in names:
                names.add(alias)
                
        return names
    
    def validate_command_name(self, name: str) -> bool:
        """
        Check if a command name is valid and available.
        
        Args:
            name: Command name to validate
            
        Returns:
            True if name is valid and available
        """
        if not name or not name.replace('-', '').replace('_', '').isalnum():
            return False
            
        return name not in self._commands and name not in self._aliases
    
    def clear(self) -> None:
        """Clear all registered commands and aliases."""
        self._commands.clear()
        self._aliases.clear()
        self.logger.debug("Cleared all registered commands")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a command by name.
        
        Args:
            name: Command name to unregister
            
        Returns:
            True if command was unregistered, False if not found
        """
        if name not in self._commands:
            return False
            
        # Remove command
        command_class = self._commands.pop(name)
        
        # Remove associated aliases
        aliases_to_remove = [alias for alias, cmd_name in self._aliases.items() if cmd_name == name]
        for alias in aliases_to_remove:
            self._aliases.pop(alias)
            
        self.logger.debug(f"Unregistered command: {name} (class: {command_class.__name__})")
        return True