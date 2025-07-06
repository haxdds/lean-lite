"""
Base indicator class for Lean-Lite.

This module provides the base class for all technical indicators.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseIndicator(ABC):
    """Base class for all technical indicators."""
    
    def __init__(self, period: int = 14):
        """Initialize the base indicator.
        
        Args:
            period (int): The period for the indicator
        """
        self.period = period
        self.values: List[float] = []
    
    @abstractmethod
    def update(self, value: float) -> Optional[float]:
        """Update the indicator with a new value.
        
        Args:
            value (float): New value to add
            
        Returns:
            Optional[float]: Current indicator value if available
        """
        pass
    
    def reset(self):
        """Reset the indicator."""
        self.values.clear() 