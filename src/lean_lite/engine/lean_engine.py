"""
Lean Engine implementation for Lean-Lite.

This module contains the main engine that orchestrates the algorithmic trading system.
"""

import logging
import time
from typing import Dict, List, Optional
from pathlib import Path

from ..config import Config
from ..brokers import BaseBroker
from ..data import DataManager
from ..algorithm import AlgorithmFramework

logger = logging.getLogger(__name__)


class LeanEngine:
    """Main engine for Lean-Lite runtime."""
    
    def __init__(self, config: Config):
        """Initialize the Lean engine."""
        self.config = config
        self.broker: Optional[BaseBroker] = None
        self.data_manager: Optional[DataManager] = None
        self.algorithm_framework: Optional[AlgorithmFramework] = None
        self.running = False
        
        logger.info("LeanEngine initialized")
    
    def start(self):
        """Start the Lean engine."""
        if not self.config.validate():
            raise ValueError("Invalid configuration")
        
        # Initialize components
        self._initialize_broker()
        self._initialize_data_manager()
        self._initialize_algorithm_framework()
        
        self.running = True
        logger.info("LeanEngine started successfully")
    
    def _initialize_broker(self):
        """Initialize the broker connection."""
        from ..brokers import AlpacaBroker
        
        if self.config.broker_type == "alpaca":
            self.broker = AlpacaBroker(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
                paper=True  # Default to paper trading
            )
            self.broker.connect()
        else:
            raise ValueError(f"Unsupported broker type: {self.config.broker_type}")
    
    def _initialize_data_manager(self):
        """Initialize the data manager."""
        from ..data import DataManager
        
        self.data_manager = DataManager(
            broker=self.broker,
            config=self.config
        )
        self.data_manager.initialize()
    
    def _initialize_algorithm_framework(self):
        """Initialize the algorithm framework."""
        from ..algorithm import AlgorithmFramework
        
        self.algorithm_framework = AlgorithmFramework(
            broker=self.broker,
            data_manager=self.data_manager,
            config=self.config
        )
        self.algorithm_framework.initialize()
    
    def run(self):
        """Run the main engine loop."""
        logger.info("Starting main engine loop")
        
        try:
            while self.running:
                # Process market data
                self._process_market_data()
                
                # Execute algorithms
                self._execute_algorithms()
                
                # Small delay to prevent CPU spinning
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in main engine loop: {e}")
            raise
    
    def _process_market_data(self):
        """Process incoming market data."""
        if self.data_manager:
            self.data_manager.process_data()
    
    def _execute_algorithms(self):
        """Execute all running algorithms."""
        if self.algorithm_framework:
            self.algorithm_framework.execute_algorithms()
    
    def stop(self):
        """Stop the Lean engine."""
        logger.info("Stopping LeanEngine")
        self.running = False
        
        if self.broker:
            self.broker.disconnect()
        
        logger.info("LeanEngine stopped") 