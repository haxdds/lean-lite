"""
Base broker interface for Lean-Lite.

This module provides the base interface that all broker implementations
must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseBroker(ABC):
    """Base interface for all broker implementations."""
    
    @abstractmethod
    def connect(self):
        """Connect to the broker."""
        pass
        
    @abstractmethod
    def disconnect(self):
        """Disconnect from the broker."""
        pass
        
    @abstractmethod
    def buy(self, symbol: str, quantity: int):
        """Place a buy order."""
        pass
        
    @abstractmethod
    def sell(self, symbol: str, quantity: int):
        """Place a sell order."""
        pass
        
    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        pass 