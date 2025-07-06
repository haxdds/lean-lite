#!/usr/bin/env python3
"""
Main entry point for Lean-Lite runtime.

This module serves as the primary entry point for the Lean-Lite application,
providing a lightweight alternative to the full QuantConnect LEAN engine.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lean_lite.engine import LeanEngine
from lean_lite.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    try:
        logger.info("Starting Lean-Lite runtime...")
        
        # Load configuration
        config = Config()
        
        # Initialize the LEAN engine
        engine = LeanEngine(config)
        
        # Start the engine
        engine.start()
        
        logger.info("Lean-Lite runtime started successfully")
        
        # Keep the engine running
        engine.run()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting Lean-Lite: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 