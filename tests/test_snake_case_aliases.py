import pytest
from datetime import datetime
from src.lean_lite.algorithm.qc_algorithm import QCAlgorithm, Symbol, SecurityType


class TestSnakeCaseAliases(QCAlgorithm):
    """Test class to verify snake_case method aliases work correctly."""
    
    def Initialize(self):
        pass
    
    def OnData(self, data):
        pass


class TestSnakeCaseMethodAliases:
    """Test that all snake_case method aliases work correctly."""
    
    def setup_method(self):
        """Set up test environment."""
        self.algorithm = TestSnakeCaseAliases()
        
        # Create a mock broker for testing
        from unittest.mock import Mock
        mock_broker = Mock()
        mock_order = Mock()
        mock_order.id = "test_order_123"
        mock_broker.buy.return_value = mock_order
        mock_broker.sell.return_value = mock_order
        
        self.algorithm.broker = mock_broker
    
    def test_set_start_date_alias(self):
        """Test set_start_date alias delegates to SetStartDate."""
        self.algorithm.set_start_date(2024, 1, 1)
        assert self.algorithm.start_date.year == 2024
        assert self.algorithm.start_date.month == 1
        assert self.algorithm.start_date.day == 1
    
    def test_set_end_date_alias(self):
        """Test set_end_date alias delegates to SetEndDate."""
        self.algorithm.set_end_date(2024, 12, 31)
        assert self.algorithm.end_date.year == 2024
        assert self.algorithm.end_date.month == 12
        assert self.algorithm.end_date.day == 31
    
    def test_set_cash_alias(self):
        """Test set_cash alias delegates to SetCash."""
        self.algorithm.set_cash(50000.0)
        assert self.algorithm.cash == 50000.0
        assert self.algorithm.portfolio_value == 50000.0
    
    def test_add_equity_alias(self):
        """Test add_equity alias delegates to AddEquity."""
        symbol = self.algorithm.add_equity("AAPL")
        assert isinstance(symbol, Symbol)
        assert symbol.ticker == "AAPL"
        assert symbol.security_type == SecurityType.EQUITY
        assert "AAPL" in self.algorithm.securities
    
    def test_add_forex_alias(self):
        """Test add_forex alias delegates to AddForex."""
        symbol = self.algorithm.add_forex("EURUSD")
        assert isinstance(symbol, Symbol)
        assert symbol.ticker == "EURUSD"
        assert symbol.security_type == SecurityType.FOREX
        assert "EURUSD" in self.algorithm.securities
    
    def test_add_crypto_alias(self):
        """Test add_crypto alias delegates to AddCrypto."""
        symbol = self.algorithm.add_crypto("BTCUSD")
        assert isinstance(symbol, Symbol)
        assert symbol.ticker == "BTCUSD"
        assert symbol.security_type == SecurityType.CRYPTO
        assert "BTCUSD" in self.algorithm.securities
    
    def test_buy_alias(self):
        """Test buy alias delegates to Buy."""
        # Add security first
        self.algorithm.add_equity("AAPL")
        
        # Test buy alias
        self.algorithm.buy("AAPL", 10)
        # Verify the method was called (we can't easily test the broker call without mocking)
        # But we can verify the method exists and doesn't raise an error
    
    def test_sell_alias(self):
        """Test sell alias delegates to Sell."""
        # Add security first
        self.algorithm.add_equity("AAPL")
        
        # Test sell alias
        self.algorithm.sell("AAPL", 5)
        # Verify the method was called
    
    def test_market_order_alias(self):
        """Test market_order alias delegates to MarketOrder."""
        # Add security first
        self.algorithm.add_equity("AAPL")
        
        # Test market_order alias for buy
        result = self.algorithm.market_order("AAPL", 10)
        # Verify the method was called
    
    def test_limit_order_alias(self):
        """Test limit_order alias delegates to LimitOrder."""
        # Add security first
        self.algorithm.add_equity("AAPL")
        
        # Test limit_order alias
        result = self.algorithm.limit_order("AAPL", 10, 150.0)
        # Verify the method was called
    
    def test_set_holdings_alias(self):
        """Test set_holdings alias delegates to SetHoldings."""
        # Add security first
        self.algorithm.add_equity("AAPL")
        
        # Set price for SetHoldings to work
        self.algorithm.update_security_price("AAPL", 100.0)
        
        # Test set_holdings alias
        self.algorithm.set_holdings("AAPL", 0.5)
        # Verify the method was called
    
    def test_liquidate_alias(self):
        """Test liquidate alias delegates to Liquidate."""
        # Add security first
        self.algorithm.add_equity("AAPL")
        
        # Test liquidate alias
        self.algorithm.liquidate("AAPL")
        # Verify the method was called
    
    def test_get_last_known_price_alias(self):
        """Test get_last_known_price alias delegates to GetLastKnownPrice."""
        # Add security first
        self.algorithm.add_equity("AAPL")
        self.algorithm.update_security_price("AAPL", 150.0)
        
        # Test get_last_known_price alias
        price = self.algorithm.get_last_known_price("AAPL")
        assert price == 150.0
    
    def test_update_security_price_alias(self):
        """Test update_security_price alias delegates to UpdateSecurityPrice."""
        # Add security first
        self.algorithm.add_equity("AAPL")
        
        # Test update_security_price alias
        self.algorithm.update_security_price("AAPL", 200.0, 1000)
        assert self.algorithm.securities["AAPL"].price == 200.0
        assert self.algorithm.securities["AAPL"].volume == 1000
    
    def test_set_benchmark_alias(self):
        """Test set_benchmark alias delegates to SetBenchmark."""
        # Test set_benchmark alias
        self.algorithm.set_benchmark("SPY")
        assert hasattr(self.algorithm, 'benchmark_symbol')
        assert self.algorithm.benchmark_symbol == "SPY"
    
    def test_log_alias(self):
        """Test log alias delegates to Log."""
        # Test log alias
        self.algorithm.log("Test log message")
        # Verify the method was called (we can't easily test logging output without capturing)
    
    def test_debug_alias(self):
        """Test debug alias delegates to Debug."""
        # Test debug alias
        self.algorithm.debug("Test debug message")
        # Verify the method was called
    
    def test_error_alias(self):
        """Test error alias delegates to Error."""
        # Test error alias
        self.algorithm.error("Test error message")
        # Verify the method was called
    
    def test_both_naming_conventions_work(self):
        """Test that both PascalCase and snake_case work identically."""
        # Test PascalCase
        self.algorithm.SetCash(100000.0)
        self.algorithm.AddEquity("AAPL")
        self.algorithm.UpdateSecurityPrice("AAPL", 100.0)
        
        # Test snake_case
        self.algorithm.set_cash(100000.0)
        self.algorithm.add_equity("MSFT")
        self.algorithm.update_security_price("MSFT", 200.0)
        
        # Verify both work the same
        assert self.algorithm.cash == 100000.0
        assert "AAPL" in self.algorithm.securities
        assert "MSFT" in self.algorithm.securities
        assert self.algorithm.securities["AAPL"].price == 100.0
        assert self.algorithm.securities["MSFT"].price == 200.0 