"""
Utility functions for doc-generator.
"""

import logging
import time
from pathlib import Path

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