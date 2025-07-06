"""
Oscillator indicators for Lean-Lite.

This module provides RSI and MACD oscillator indicators.
"""

from typing import List, Optional
from .base_indicator import BaseIndicator


class RSI(BaseIndicator):
    """Relative Strength Index (RSI) indicator."""
    
    def __init__(self, period: int = 14):
        """Initialize the RSI.
        
        Args:
            period (int): The period for the RSI
        """
        super().__init__(period)
        self.gains: List[float] = []
        self.losses: List[float] = []
        self.previous_price: Optional[float] = None
    
    def update(self, value: float) -> Optional[float]:
        """Update the RSI with a new value.
        
        Args:
            value (float): New price value
            
        Returns:
            Optional[float]: Current RSI value if enough data
        """
        if self.previous_price is not None:
            change = value - self.previous_price
            gain = max(change, 0)
            loss = max(-change, 0)
            
            self.gains.append(gain)
            self.losses.append(loss)
            
            # Keep only the last 'period' values
            if len(self.gains) > self.period:
                self.gains = self.gains[-self.period:]
                self.losses = self.losses[-self.period:]
        
        self.previous_price = value
        
        if len(self.gains) >= self.period:
            avg_gain = sum(self.gains) / len(self.gains)
            avg_loss = sum(self.losses) / len(self.losses)
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        return None


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence (MACD) indicator."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Initialize the MACD.
        
        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
        """
        super().__init__(slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fast_ema = 0.0
        self.slow_ema = 0.0
        self.signal_ema = 0.0
        self.macd_values: List[float] = []
    
    def update(self, value: float) -> Optional[float]:
        """Update the MACD with a new value.
        
        Args:
            value (float): New price value
            
        Returns:
            Optional[float]: Current MACD value if enough data
        """
        # Update EMAs
        if self.fast_ema == 0:
            self.fast_ema = value
            self.slow_ema = value
        else:
            fast_multiplier = 2.0 / (self.fast_period + 1)
            slow_multiplier = 2.0 / (self.slow_period + 1)
            
            self.fast_ema = (value * fast_multiplier) + (self.fast_ema * (1 - fast_multiplier))
            self.slow_ema = (value * slow_multiplier) + (self.slow_ema * (1 - slow_multiplier))
        
        # Calculate MACD line
        macd_line = self.fast_ema - self.slow_ema
        self.macd_values.append(macd_line)
        
        # Calculate signal line
        if len(self.macd_values) >= self.signal_period:
            if self.signal_ema == 0:
                self.signal_ema = macd_line
            else:
                signal_multiplier = 2.0 / (self.signal_period + 1)
                self.signal_ema = (macd_line * signal_multiplier) + (self.signal_ema * (1 - signal_multiplier))
            
            return macd_line
        
        return None 