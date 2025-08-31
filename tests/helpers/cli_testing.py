"""
Helper utilities for CLI testing.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from unittest.mock import patch


class CLITestRunner:
    """Helper class for running CLI commands in tests."""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
    
    def run_cli_command(
        self, 
        args: List[str], 
        env_vars: Optional[Dict[str, str]] = None,
        expect_success: bool = True
    ) -> Dict[str, Any]:
        """
        Run a CLI command with the given arguments.
        
        Args:
            args: Command line arguments (excluding 'doc-gen')
            env_vars: Environment variables to set
            expect_success: Whether to expect the command to succeed
        
        Returns:
            Dictionary with 'returncode', 'stdout', 'stderr'
        """
        cmd = [sys.executable, '-m', 'doc_generator.cli'] + args
        
        # Set up environment
        env = {}
        if env_vars:
            env.update(env_vars)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=30  # Prevent hanging tests
            )
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'success': False
            }
    
    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a test file in the temp directory."""
        file_path = self.temp_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def create_test_directory(self, dirname: str) -> Path:
        """Create a test directory."""
        dir_path = self.temp_dir / dirname
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path


def mock_successful_generation():
    """Create a mock for successful documentation generation."""
    return {
        'content': '# Test Documentation\n\nThis is generated content.',
        'metadata': {
            'provider': 'openai',
            'model': 'gpt-4',
            'tokens_used': 100
        },
        'sections': ['title', 'content']
    }


def mock_successful_standardization():
    """Create a mock for successful document standardization."""
    return {
        'standardized_content': '# Standardized Document\n\nStandardized content.',
        'original_format': 'html',
        'target_format': 'markdown',
        'sections_processed': ['title', 'content'],
        'metadata': {
            'provider': 'openai',
            'model': 'gpt-4',
            'tokens_used': 50
        }
    }


class CLITestCase:
    """Base test case for CLI testing with common setup."""
    
    def __init__(self):
        self.runner = None
        self.temp_dir = None
    
    def setup_method(self):
        """Set up test case."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.runner = CLITestRunner(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test case."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)