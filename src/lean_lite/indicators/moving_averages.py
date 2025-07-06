"""
Moving average indicators for Lean-Lite.

This module provides simple and exponential moving average indicators.
"""

from typing import List, Optional
from .base_indicator import BaseIndicator


class SimpleMovingAverage(BaseIndicator):
    """Simple Moving Average (SMA) indicator."""
    
    def update(self, value: float) -> Optional[float]:
        """Update the SMA with a new value.
        
        Args:
            value (float): New value to add
            
        Returns:
            Optional[float]: Current SMA value if enough data
        """
        self.values.append(value)
        
        if len(self.values) >= self.period:
            # Keep only the last 'period' values
            self.values = self.values[-self.period:]
            return sum(self.values) / len(self.values)
        
        return None


class ExponentialMovingAverage(BaseIndicator):
    """Exponential Moving Average (EMA) indicator."""
    
    def __init__(self, period: int = 14):
        """Initialize the EMA.
        
        Args:
            period (int): The period for the EMA
        """
        super().__init__(period)
        self.multiplier = 2.0 / (period + 1)
        self.ema_value: Optional[float] = None
    
    def update(self, value: float) -> Optional[float]:
        """Update the EMA with a new value.
        
        Args:
            value (float): New value to add
            
        Returns:
            Optional[float]: Current EMA value
        """
        if self.ema_value is None:
            self.ema_value = value
        else:
            self.ema_value = (value * self.multiplier) + (self.ema_value * (1 - self.multiplier))
        
        return self.ema_value 