"""
Algorithm framework for Lean-Lite.

This module provides the framework for managing and executing
trading strategies with alpaca-py integration.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

from .base_algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)


class AlgorithmFramework:
    """Framework for managing and executing trading algorithms."""
    
    def __init__(self, broker, data_manager, config):
        """Initialize the algorithm framework."""
        self.broker = broker
        self.data_manager = data_manager
        self.config = config
        self.algorithms: List[BaseAlgorithm] = []
        self.running = False
        
        logger.info("AlgorithmFramework initialized")
    
    def initialize(self):
        """Initialize the algorithm framework."""
        logger.info("Initializing AlgorithmFramework")
        
        # Load strategies from the strategies directory
        self._load_strategies()
        
        # Initialize all algorithms
        for algorithm in self.algorithms:
            try:
                # Set broker and data manager references
                algorithm.broker = self.broker
                algorithm.data_manager = self.data_manager
                
                # Initialize the algorithm
                algorithm.initialize()
                
                # Subscribe to data for the algorithm's symbols
                if hasattr(algorithm, 'symbol') and algorithm.symbol:
                    self.data_manager.subscribe_symbol(algorithm.symbol)
                
                logger.info(f"Algorithm {algorithm.__class__.__name__} initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize algorithm {algorithm.__class__.__name__}: {e}")
    
    def _load_strategies(self):
        """Load strategies from the strategies directory."""
        strategy_path = Path(self.config.strategy_path)
        
        if not strategy_path.exists():
            logger.warning(f"Strategy path {strategy_path} does not exist")
            return
        
        # Import and instantiate strategies
        try:
            # Import example strategy
            from strategies.example_strategy import ExampleStrategy
            self.algorithms.append(ExampleStrategy())
            logger.info("Loaded ExampleStrategy")
            
        except ImportError as e:
            logger.warning(f"Could not import example strategy: {e}")
    
    def execute_algorithms(self):
        """Execute all running algorithms."""
        if not self.running:
            return
        
        try:
            # Get latest market data
            market_data = self.data_manager.process_data()
            
            # Execute each algorithm
            for algorithm in self.algorithms:
                try:
                    algorithm.on_data(market_data)
                except Exception as e:
                    logger.error(f"Error executing algorithm {algorithm.__class__.__name__}: {e}")
                    algorithm.on_error(e)
                    
        except Exception as e:
            logger.error(f"Error in algorithm execution: {e}")
    
    def start(self):
        """Start the algorithm framework."""
        self.running = True
        logger.info("AlgorithmFramework started")
    
    def stop(self):
        """Stop the algorithm framework."""
        self.running = False
        logger.info("AlgorithmFramework stopped")
    
    def add_algorithm(self, algorithm: BaseAlgorithm):
        """Add an algorithm to the framework."""
        self.algorithms.append(algorithm)
        logger.info(f"Added algorithm: {algorithm.__class__.__name__}")
    
    def remove_algorithm(self, algorithm: BaseAlgorithm):
        """Remove an algorithm from the framework."""
        if algorithm in self.algorithms:
            self.algorithms.remove(algorithm)
            logger.info(f"Removed algorithm: {algorithm.__class__.__name__}")
    
    def get_algorithms(self) -> List[BaseAlgorithm]:
        """Get all algorithms in the framework."""
        return self.algorithms.copy() 