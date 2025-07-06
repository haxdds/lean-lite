"""
Example QuantConnect-compatible strategy for Lean-Lite.

This strategy demonstrates how to use the QCAlgorithm base class
to write strategies that are compatible with QuantConnect's API.
"""

from lean_lite.algorithm import QCAlgorithm, Symbol


class QCExampleStrategy(QCAlgorithm):
    """Example QuantConnect-compatible trading strategy."""
    
    def Initialize(self):
        """Initialize the algorithm."""
        # Set start and end dates
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 12, 31)
        
        # Set starting cash
        self.SetCash(100000)
        
        # Add securities
        self.spy = self.AddEquity("SPY")
        self.aapl = self.AddEquity("AAPL")
        
        # Set up variables
        self.previous_price = 0
        
        self.Log("QCExampleStrategy initialized")
    
    def OnData(self, data):
        """Handle incoming market data."""
        # Get current price for SPY
        if "SPY" in data:
            current_price = data["SPY"]["close"]
            
            # Simple moving average crossover strategy
            if self.previous_price > 0:
                if current_price > self.previous_price * 1.01:  # 1% increase
                    if not self.has_position("SPY"):
                        self.Buy(self.spy, 100)
                        self.Log(f"Bought 100 shares of SPY at ${current_price}")
                
                elif current_price < self.previous_price * 0.99:  # 1% decrease
                    if self.has_position("SPY"):
                        self.Sell(self.spy, 100)
                        self.Log(f"Sold 100 shares of SPY at ${current_price}")
            
            self.previous_price = current_price
    
    def OnOrderEvent(self, order_event):
        """Handle order events."""
        self.Log(f"Order filled: {order_event.symbol} {order_event.filled_quantity} "
                f"shares at ${order_event.filled_price}")
    
    def OnEndOfDay(self, symbol):
        """Handle end of day events."""
        self.Log(f"End of day for {symbol}") 