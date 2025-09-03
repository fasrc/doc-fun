"""
Test Command

Simple test command to verify CLI infrastructure works correctly.
"""

from argparse import ArgumentParser, Namespace
from typing import Optional

from ..base import BaseCommand, CommandResult


class TestCommand(BaseCommand):
    """Test command for verifying CLI infrastructure."""
    
    @property
    def name(self) -> str:
        return "test"
    
    @property
    def description(self) -> str:
        return "Test CLI infrastructure functionality"
    
    @property
    def aliases(self):
        return ["t"]
    
    @property
    def hidden(self) -> bool:
        return True  # Hide from normal help
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add test command arguments."""
        parser.add_argument('--message', '-m', 
                          default="CLI infrastructure test successful!",
                          help='Message to display')
        parser.add_argument('--fail', action='store_true',
                          help='Force command to fail for testing error handling')
    
    def run(self, args: Namespace) -> CommandResult:
        """Execute test command."""
        if args.fail:
            return self.handle_error(
                Exception("Simulated failure for testing"),
                "Test command failure simulation"
            )
        
        return self.success(args.message)