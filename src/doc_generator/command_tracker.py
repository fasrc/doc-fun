"""
Command tracking utilities for saving generation commands alongside output files.
"""
import sys
from pathlib import Path
from typing import List, Optional


class CommandTracker:
    """Tracks and saves the CLI command used to generate documentation files."""
    
    @staticmethod
    def get_current_command() -> str:
        """Get the current command line invocation with proper quoting."""
        import shlex
        # Use shlex.quote to properly escape arguments that need quoting
        quoted_args = []
        for arg in sys.argv:
            # Check if the argument contains spaces and doesn't already have quotes
            if ' ' in arg and not (arg.startswith('"') and arg.endswith('"')):
                quoted_args.append(shlex.quote(arg))
            else:
                quoted_args.append(arg)
        return ' '.join(quoted_args)
    
    @staticmethod
    def generate_command_filename(base_filename: str) -> str:
        """Generate command file name based on the output file name."""
        # Remove extension and add -command.sh
        base_path = Path(base_filename)
        name_without_ext = base_path.stem
        return f"{name_without_ext}-command.sh"
    
    @staticmethod
    def save_command_file(output_filepath: str, command: Optional[str] = None) -> str:
        """
        Save the command used to generate a file alongside the output file.
        
        Args:
            output_filepath: Path to the generated file
            command: Command to save (defaults to current command)
            
        Returns:
            Path to the created command file
        """
        if command is None:
            command = CommandTracker.get_current_command()
        
        output_path = Path(output_filepath)
        command_filename = CommandTracker.generate_command_filename(output_path.name)
        command_filepath = output_path.parent / command_filename
        
        try:
            with open(command_filepath, 'w', encoding='utf-8') as f:
                f.write("#!/bin/bash\n")
                f.write(f"# Command used to generate {output_path.name}\n")
                f.write(f"# Generated on: $(date)\n\n")
                f.write(f"{command}\n")
            
            # Make the file executable
            command_filepath.chmod(0o755)
            
            return str(command_filepath)
        except Exception as e:
            # Log error but don't fail the main generation
            print(f"Warning: Could not save command file {command_filepath}: {e}")
            return ""
    
    @staticmethod
    def save_command_files_for_batch(output_filepaths: List[str], command: Optional[str] = None) -> List[str]:
        """
        Save command files for a batch of generated files.
        
        Args:
            output_filepaths: List of paths to generated files
            command: Command to save (defaults to current command)
            
        Returns:
            List of paths to created command files
        """
        command_files = []
        for filepath in output_filepaths:
            command_file = CommandTracker.save_command_file(filepath, command)
            if command_file:
                command_files.append(command_file)
        
        return command_files