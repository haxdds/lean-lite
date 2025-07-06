"""
Example strategy for Lean-Lite.

This is a simple example strategy that demonstrates the basic structure
for implementing trading algorithms in Lean-Lite.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from lean_lite.algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)


class ExampleStrategy(BaseAlgorithm):
    """Example trading strategy implementation."""
    
    def __init__(self):
        """Initialize the example strategy."""
        super().__init__()
        self.symbol = "SPY"
        self.position_size = 100
        
    def initialize(self):
        """Initialize the strategy."""
        logger.info("Initializing ExampleStrategy")
        
        # Set the symbol to trade
        self.set_symbol(self.symbol)
        
        # Set initial parameters
        self.set_position_size(self.position_size)
        
        logger.info(f"Strategy initialized for {self.symbol}")
    
    def on_data(self, data: Dict[str, Any]):
        """Handle incoming market data."""
        if not data or self.symbol not in data:
            return
        
        current_price = data[self.symbol]["close"]
        logger.info(f"Current price for {self.symbol}: ${current_price}")
        
        # Simple example logic
        if current_price > 400:  # Example threshold
            if not self.has_position(self.symbol):
                self.buy(self.symbol, self.position_size)
                logger.info(f"Bought {self.position_size} shares of {self.symbol}")
        elif current_price < 380:  # Example threshold
            if self.has_position(self.symbol):
                self.sell(self.symbol, self.position_size)
                logger.info(f"Sold {self.position_size} shares of {self.symbol}")
    
    def on_order_filled(self, order):
        """Handle order fill events."""
        logger.info(f"Order filled: {order}")
    
    def on_error(self, error):
        """Handle errors."""
        logger.error(f"Strategy error: {error}") 