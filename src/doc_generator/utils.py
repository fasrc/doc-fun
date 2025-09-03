"""
Utility functions for doc-generator.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from .config import get_settings


def get_output_directory(output_dir: str, logger: logging.Logger) -> str:
    """
    Get the output directory path, creating a timestamped subdirectory if using default.
    
    Args:
        output_dir: The output directory from command line args
        logger: Logger instance
        
    Returns:
        The final output directory path to use
    """
    settings = get_settings()
    
    if output_dir == 'output':  # Default value, create timestamped directory
        timestamp = int(time.time())
        base_output = settings.paths.output_dir
        final_output_dir = base_output / str(timestamp)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created timestamped output directory: {final_output_dir}")
        return str(final_output_dir)
    else:
        final_output_dir = output_dir
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)
    
    return final_output_dir


def safe_file_read(file_path: Path, max_chars: int = 2000, encoding: str = 'utf-8') -> Optional[str]:
    """
    Safely read file contents with error handling and size limits.
    
    Args:
        file_path: Path to the file to read
        max_chars: Maximum number of characters to read
        encoding: File encoding to use
        
    Returns:
        File contents as string, or None if reading fails
    """
    try:
        if not file_path.exists() or not file_path.is_file():
            return None
            
        # Check file size to avoid reading huge files
        if file_path.stat().st_size > max_chars * 4:  # Rough estimate for UTF-8
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                return f.read(max_chars)
        else:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
                return content[:max_chars] if len(content) > max_chars else content
                
    except Exception:
        return None