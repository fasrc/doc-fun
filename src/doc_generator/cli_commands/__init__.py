"""
CLI Commands Package

Command Pattern infrastructure for CLI refactoring.
This package provides the foundation for migrating from the monolithic
cli.py to a more modular command-based architecture.
"""

from .base import BaseCommand, CommandResult
from .registry import CommandRegistry
from .dispatcher import CommandDispatcher

__all__ = ['BaseCommand', 'CommandResult', 'CommandRegistry', 'CommandDispatcher']