"""
New CLI Main Entry Point

Uses the command pattern dispatcher for better organization and extensibility.
"""

import logging
import sys
from typing import List, Optional

from .bootstrap import create_default_registry
from .dispatcher import CommandDispatcher


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point using command pattern dispatcher.
    
    Args:
        args: Command-line arguments. If None, uses sys.argv[1:]
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Set up basic logging for registry initialization
    logger = setup_logging()
    
    try:
        # Create command registry with all default commands
        registry = create_default_registry(logger)
        
        # Create dispatcher
        dispatcher = CommandDispatcher(registry, logger)
        
        # Run dispatcher
        return dispatcher.run(args)
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        return 1


def cli_main():
    """Entry point for console scripts."""
    sys.exit(main())


if __name__ == "__main__":
    cli_main()