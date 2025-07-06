"""
Tests for Lean-Lite data models.

This module tests all data structures for QuantConnect compatibility,
including serialization, validation, and edge cases.
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
import time

from lean_lite.algorithm.data_models import (
    Symbol, Security, TradeBar, QuoteBar, Portfolio, SecurityHolding,
    Resolution, SecurityType, Market, create_symbol, create_trade_bar,
    create_quote_bar, create_portfolio
)


class TestSymbol:
    """Test cases for Symbol class."""
    
    def test_symbol_creation(self):
        """Test basic symbol creation."""
        symbol = Symbol("AAPL")
        assert symbol.Value == "AAPL"
        assert symbol.SecurityType == SecurityType.EQUITY
        assert symbol.Market == Market.US_EQUITY
    
    def test_symbol_case_normalization(self):
        """Test that symbols are normalized to uppercase."""
        symbol = Symbol("aapl")
        assert symbol.Value == "AAPL"
    
    def test_symbol_with_custom_type_and_market(self):
        """Test symbol with custom security type and market."""
        symbol = Symbol("EURUSD", SecurityType.FOREX, Market.FOREX)
        assert symbol.Value == "EURUSD"
        assert symbol.SecurityType == SecurityType.FOREX
        assert symbol.Market == Market.FOREX
    
    def test_symbol_validation(self):
        """Test symbol validation."""
        with pytest.raises(ValueError):
            Symbol("")
        
        with pytest.raises(ValueError):
            Symbol(None)
    
    def test_symbol_string_representation(self):
        """Test string representation."""
        symbol = Symbol("AAPL", SecurityType.EQUITY)
        assert str(symbol) == "AAPL (equity)"
    
    def test_symbol_repr(self):
        """Test detailed representation."""
        symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
        repr_str = repr(symbol)
        assert "Symbol" in repr_str
        assert "AAPL" in repr_str
        assert "SecurityType.EQUITY" in repr_str
    
    def test_symbol_equality(self):
        """Test symbol equality."""
        symbol1 = Symbol("AAPL", SecurityType.EQUITY)
        symbol2 = Symbol("AAPL", SecurityType.EQUITY)
        symbol3 = Symbol("AAPL", SecurityType.FOREX)
        
        assert symbol1 == symbol2
        assert symbol1 != symbol3
        assert symbol1 != "AAPL"  # Different type
    
    def test_symbol_hash(self):
        """Test symbol hashing."""
        symbol1 = Symbol("AAPL", SecurityType.EQUITY)
        symbol2 = Symbol("AAPL", SecurityType.EQUITY)
        
        assert hash(symbol1) == hash(symbol2)
        
        # Can be used as dictionary key
        d = {symbol1: "value"}
        assert d[symbol2] == "value"
    
    def test_symbol_json_serialization(self):
        """Test JSON serialization."""
        symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
        
        # Test to_dict
        data = symbol.to_dict()
        assert data['Value'] == "AAPL"
        assert data['SecurityType'] == "equity"
        assert data['Market'] == "us_equity"
        
        # Test from_dict
        symbol2 = Symbol.from_dict(data)
        assert symbol2 == symbol
        
        # Test to_json
        json_str = symbol.to_json()
        assert "AAPL" in json_str
        assert "equity" in json_str
        
        # Test from_json
        symbol3 = Symbol.from_json(json_str)
        assert symbol3 == symbol


class TestSecurityHolding:
    """Test cases for SecurityHolding class."""
    
    def test_holding_creation(self):
        """Test basic holding creation."""
        symbol = Symbol("AAPL")
        holding = SecurityHolding(symbol, 100, 150.0)
        
        assert holding.Symbol == symbol
        assert holding.Quantity == 100
        assert holding.AveragePrice == 150.0
        assert holding.IsLong
    
    def test_holding_properties(self):
        """Test holding properties."""
        symbol = Symbol("AAPL")
        
        # Long position
        long_holding = SecurityHolding(symbol, 100, 150.0)
        assert long_holding.IsLong
        assert not long_holding.IsShort
        assert not long_holding.IsFlat
        assert long_holding.AbsoluteQuantity == 100
        
        # Short position
        short_holding = SecurityHolding(symbol, -50, 150.0)
        assert short_holding.IsShort
        assert not short_holding.IsLong
        assert not short_holding.IsFlat
        assert short_holding.AbsoluteQuantity == 50
        
        # Flat position
        flat_holding = SecurityHolding(symbol, 0, 150.0)
        assert flat_holding.IsFlat
        assert not flat_holding.IsLong
        assert not flat_holding.IsShort
        assert flat_holding.AbsoluteQuantity == 0
    
    def test_unrealized_profit_calculation(self):
        """Test unrealized profit calculation."""
        symbol = Symbol("AAPL")
        holding = SecurityHolding(symbol, 100, 150.0, 500.0)
        
        # 100 shares bought at $150, current profit $500
        # Unrealized profit percent = 500 / (150 * 100) * 100 = 3.33%
        assert abs(holding.UnrealizedProfitPercent - 3.33) < 0.1
    
    def test_holding_string_representation(self):
        """Test string representation."""
        symbol = Symbol("AAPL")
        holding = SecurityHolding(symbol, 100, 150.0)
        
        assert "AAPL" in str(holding)
        assert "100" in str(holding)
        assert "LONG" in str(holding)
        assert "150.00" in str(holding)
    
    def test_holding_json_serialization(self):
        """Test JSON serialization."""
        symbol = Symbol("AAPL")
        holding = SecurityHolding(symbol, 100, 150.0, 500.0)
        
        data = holding.to_dict()
        assert data['Quantity'] == 100
        assert data['AveragePrice'] == 150.0
        assert data['UnrealizedProfit'] == 500.0
        
        holding2 = SecurityHolding.from_dict(data)
        assert holding2.Quantity == holding.Quantity
        assert holding2.AveragePrice == holding.AveragePrice
        assert holding2.Symbol == holding.Symbol


class TestSecurity:
    """Test cases for Security class."""
    
    def test_security_creation(self):
        """Test basic security creation."""
        symbol = Symbol("AAPL")
        security = Security(symbol, 150.0, 1000)
        
        assert security.Symbol == symbol
        assert security.Price == 150.0
        assert security.Volume == 1000
        assert isinstance(security.Holdings, SecurityHolding)
        assert security.Holdings.Symbol == symbol
    
    def test_security_price_update(self):
        """Test price update functionality."""
        symbol = Symbol("AAPL")
        security = Security(symbol, 150.0)
        
        # Update price
        security.UpdatePrice(160.0, 2000)
        assert security.Price == 160.0
        assert security.Volume == 2000
        assert security.LastUpdate is not None
    
    def test_security_holdings_update_on_price_change(self):
        """Test that holdings are updated when price changes."""
        symbol = Symbol("AAPL")
        security = Security(symbol, 150.0)
        
        # Add some holdings
        security.Holdings.Quantity = 100
        security.Holdings.AveragePrice = 140.0
        
        # Update price (profit should increase)
        security.UpdatePrice(160.0)
        
        # Unrealized profit should be (160 - 140) * 100 = 2000
        assert security.Holdings.UnrealizedProfit == 2000.0
        assert security.Holdings.LastTradePrice == 160.0
    
    def test_security_string_representation(self):
        """Test string representation."""
        symbol = Symbol("AAPL")
        security = Security(symbol, 150.0, 1000)
        
        assert "AAPL" in str(security)
        assert "150.00" in str(security)
        assert "1000" in str(security)
    
    def test_security_json_serialization(self):
        """Test JSON serialization."""
        symbol = Symbol("AAPL")
        security = Security(symbol, 150.0, 1000)
        
        data = security.to_dict()
        assert data['Price'] == 150.0
        assert data['Volume'] == 1000
        assert 'Symbol' in data
        assert 'Holdings' in data
        
        security2 = Security.from_dict(data)
        assert security2.Price == security.Price
        assert security2.Volume == security.Volume
        assert security2.Symbol == security.Symbol


class TestTradeBar:
    """Test cases for TradeBar class."""
    
    def test_trade_bar_creation(self):
        """Test basic trade bar creation."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        bar = TradeBar(symbol, time, 150.0, 155.0, 148.0, 152.0, 1000)
        
        assert bar.Symbol == symbol
        assert bar.Time == time
        assert bar.Open == 150.0
        assert bar.High == 155.0
        assert bar.Low == 148.0
        assert bar.Close == 152.0
        assert bar.Volume == 1000
        assert bar.Price == 152.0  # Price should equal Close
    
    def test_trade_bar_validation(self):
        """Test trade bar data validation."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        
        # High < max(Open, Close)
        with pytest.raises(ValueError):
            TradeBar(symbol, time, 150.0, 149.0, 148.0, 152.0, 1000)
        
        # Low > min(Open, Close)
        with pytest.raises(ValueError):
            TradeBar(symbol, time, 150.0, 155.0, 151.0, 152.0, 1000)
        
        # Negative volume
        with pytest.raises(ValueError):
            TradeBar(symbol, time, 150.0, 155.0, 148.0, 152.0, -100)
    
    def test_trade_bar_properties(self):
        """Test trade bar computed properties."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        
        # Up bar
        up_bar = TradeBar(symbol, time, 150.0, 155.0, 148.0, 152.0, 1000)
        assert up_bar.IsUp
        assert not up_bar.IsDown
        assert not up_bar.IsDoji
        assert up_bar.Range == 7.0  # 155 - 148
        assert up_bar.Body == 2.0   # 152 - 150
        
        # Down bar
        down_bar = TradeBar(symbol, time, 152.0, 155.0, 148.0, 150.0, 1000)
        assert down_bar.IsDown
        assert not down_bar.IsUp
        assert not down_bar.IsDoji
        assert down_bar.Body == -2.0  # 150 - 152
        
        # Doji bar
        doji_bar = TradeBar(symbol, time, 150.0, 155.0, 148.0, 150.01, 1000)
        assert doji_bar.IsDoji
        assert not doji_bar.IsUp
        assert not doji_bar.IsDown
    
    def test_trade_bar_string_representation(self):
        """Test string representation."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        bar = TradeBar(symbol, time, 150.0, 155.0, 148.0, 152.0, 1000)
        
        str_repr = str(bar)
        assert "AAPL" in str_repr
        assert "▲" in str_repr  # Up arrow
        assert "150.00" in str_repr
        assert "155.00" in str_repr
        assert "148.00" in str_repr
        assert "152.00" in str_repr
        assert "1000" in str_repr
    
    def test_trade_bar_json_serialization(self):
        """Test JSON serialization."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        bar = TradeBar(symbol, time, 150.0, 155.0, 148.0, 152.0, 1000)
        
        data = bar.to_dict()
        assert data['Open'] == 150.0
        assert data['High'] == 155.0
        assert data['Low'] == 148.0
        assert data['Close'] == 152.0
        assert data['Volume'] == 1000
        assert 'Time' in data
        assert 'Symbol' in data
        
        bar2 = TradeBar.from_dict(data)
        assert bar2.Open == bar.Open
        assert bar2.High == bar.High
        assert bar2.Low == bar.Low
        assert bar2.Close == bar.Close
        assert bar2.Volume == bar.Volume


class TestQuoteBar:
    """Test cases for QuoteBar class."""
    
    def test_quote_bar_creation(self):
        """Test basic quote bar creation."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        
        # Create bid and ask bars
        bid_bar = TradeBar(symbol, time, 149.0, 149.5, 148.5, 149.2, 500)
        ask_bar = TradeBar(symbol, time, 150.0, 150.5, 149.8, 150.2, 500)
        
        quote_bar = QuoteBar(symbol, time, bid_bar, ask_bar)
        
        assert quote_bar.Symbol == symbol
        assert quote_bar.Time == time
        assert quote_bar.Bid == bid_bar
        assert quote_bar.Ask == ask_bar
    
    def test_quote_bar_price_calculation(self):
        """Test price calculation from bid/ask."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        
        # Both bid and ask
        bid_bar = TradeBar(symbol, time, 149.0, 149.5, 148.5, 149.2, 500)
        ask_bar = TradeBar(symbol, time, 150.0, 150.5, 149.8, 150.2, 500)
        quote_bar = QuoteBar(symbol, time, bid_bar, ask_bar)
        
        # Mid price should be (149.2 + 150.2) / 2 = 149.7
        assert abs(quote_bar.Price - 149.7) < 0.01
        assert abs(quote_bar.Spread - 1.0) < 0.01  # 150.2 - 149.2
        assert abs(quote_bar.SpreadPercent - 0.67) < 0.1  # (1.0 / 149.7) * 100
        
        # Only bid
        quote_bar_bid_only = QuoteBar(symbol, time, bid_bar, None)
        assert quote_bar_bid_only.Price == 149.2
        
        # Only ask
        quote_bar_ask_only = QuoteBar(symbol, time, None, ask_bar)
        assert quote_bar_ask_only.Price == 150.2
        
        # No quotes
        quote_bar_empty = QuoteBar(symbol, time, None, None)
        assert quote_bar_empty.Price == 0.0
    
    def test_quote_bar_validation(self):
        """Test quote bar validation."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        
        # Bid > Ask (should log warning but not raise exception)
        bid_bar = TradeBar(symbol, time, 151.0, 151.5, 150.5, 151.2, 500)
        ask_bar = TradeBar(symbol, time, 150.0, 150.5, 149.8, 150.2, 500)
        
        # This should not raise an exception, just log a warning
        quote_bar = QuoteBar(symbol, time, bid_bar, ask_bar)
        assert quote_bar.Bid == bid_bar
        assert quote_bar.Ask == ask_bar
    
    def test_quote_bar_string_representation(self):
        """Test string representation."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        bid_bar = TradeBar(symbol, time, 149.0, 149.5, 148.5, 149.2, 500)
        ask_bar = TradeBar(symbol, time, 150.0, 150.5, 149.8, 150.2, 500)
        quote_bar = QuoteBar(symbol, time, bid_bar, ask_bar)
        
        str_repr = str(quote_bar)
        assert "AAPL" in str_repr
        assert "149.20" in str_repr  # Bid
        assert "150.20" in str_repr  # Ask
        assert "1.00" in str_repr    # Spread
    
    def test_quote_bar_json_serialization(self):
        """Test JSON serialization."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        bid_bar = TradeBar(symbol, time, 149.0, 149.5, 148.5, 149.2, 500)
        ask_bar = TradeBar(symbol, time, 150.0, 150.5, 149.8, 150.2, 500)
        quote_bar = QuoteBar(symbol, time, bid_bar, ask_bar)
        
        data = quote_bar.to_dict()
        assert 'Bid' in data
        assert 'Ask' in data
        assert data['Bid'] is not None
        assert data['Ask'] is not None
        
        quote_bar2 = QuoteBar.from_dict(data)
        assert quote_bar2.Bid is not None
        assert quote_bar2.Ask is not None
        assert abs(quote_bar2.Price - quote_bar.Price) < 0.01


class TestPortfolio:
    """Test cases for Portfolio class."""
    
    def test_portfolio_creation(self):
        """Test basic portfolio creation."""
        portfolio = Portfolio(100000.0)
        
        assert portfolio.Cash == 100000.0
        assert portfolio.TotalFees == 0.0
        assert portfolio.TotalProfit == 0.0
        assert len(portfolio.Holdings) == 0
        assert portfolio.TotalPortfolioValue == 100000.0
    
    def test_portfolio_properties(self):
        """Test portfolio computed properties."""
        portfolio = Portfolio(100000.0)
        
        # Add some holdings
        symbol = Symbol("AAPL")
        holding = SecurityHolding(symbol, 100, 150.0)
        holding.LastTradePrice = 160.0  # Current price
        portfolio.Holdings["AAPL"] = holding
        
        # Total holdings value = 100 * 160 = 16000
        assert portfolio.TotalHoldingsValue == 16000.0
        # Total portfolio value = 100000 + 16000 = 116000
        assert portfolio.TotalPortfolioValue == 116000.0
        # Cash percent = (100000 / 116000) * 100 = 86.21%
        assert abs(portfolio.CashPercent - 86.21) < 0.1
    
    def test_portfolio_operations(self):
        """Test portfolio operations."""
        portfolio = Portfolio(100000.0)
        symbol = Symbol("AAPL")
        
        # Add security
        holding = portfolio.AddSecurity(symbol)
        assert holding.Symbol == symbol
        assert "AAPL" in portfolio.Holdings
        
        # Get holding
        retrieved_holding = portfolio.GetHolding("AAPL")
        assert retrieved_holding == holding
        
        retrieved_holding2 = portfolio.GetHolding(symbol)
        assert retrieved_holding2 == holding
        
        # Get non-existent holding
        assert portfolio.GetHolding("UNKNOWN") is None
    
    def test_portfolio_cash_operations(self):
        """Test cash operations."""
        portfolio = Portfolio(100000.0)
        
        # Update cash
        portfolio.UpdateCash(5000.0)
        assert portfolio.Cash == 105000.0
        
        portfolio.UpdateCash(-2000.0)
        assert portfolio.Cash == 103000.0
        
        # Add fees
        portfolio.AddFees(100.0)
        assert portfolio.TotalFees == 100.0
        assert portfolio.Cash == 102900.0
    
    def test_portfolio_unrealized_profit(self):
        """Test unrealized profit calculation."""
        portfolio = Portfolio(100000.0)
        
        # Add holdings with unrealized profit
        symbol1 = Symbol("AAPL")
        holding1 = SecurityHolding(symbol1, 100, 150.0, 1000.0)  # $1000 profit
        portfolio.Holdings["AAPL"] = holding1
        
        symbol2 = Symbol("MSFT")
        holding2 = SecurityHolding(symbol2, 50, 200.0, -500.0)   # $500 loss
        portfolio.Holdings["MSFT"] = holding2
        
        # Total unrealized profit = 1000 - 500 = 500
        assert portfolio.TotalUnrealizedProfit == 500.0
    
    def test_portfolio_string_representation(self):
        """Test string representation."""
        portfolio = Portfolio(100000.0)
        
        str_repr = str(portfolio)
        assert "Portfolio" in str_repr
        assert "100,000.00" in str_repr
        assert "0" in str_repr  # 0 holdings
    
    def test_portfolio_json_serialization(self):
        """Test JSON serialization."""
        portfolio = Portfolio(100000.0)
        symbol = Symbol("AAPL")
        holding = SecurityHolding(symbol, 100, 150.0)
        portfolio.Holdings["AAPL"] = holding
        
        data = portfolio.to_dict()
        assert data['Cash'] == 100000.0
        assert 'Holdings' in data
        assert 'AAPL' in data['Holdings']
        
        portfolio2 = Portfolio.from_dict(data)
        assert portfolio2.Cash == portfolio.Cash
        assert len(portfolio2.Holdings) == len(portfolio.Holdings)
        assert "AAPL" in portfolio2.Holdings


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_symbol(self):
        """Test create_symbol function."""
        symbol = create_symbol("AAPL")
        assert symbol.Value == "AAPL"
        assert symbol.SecurityType == SecurityType.EQUITY
        assert symbol.Market == Market.US_EQUITY
        
        symbol2 = create_symbol("EURUSD", SecurityType.FOREX, Market.FOREX)
        assert symbol2.Value == "EURUSD"
        assert symbol2.SecurityType == SecurityType.FOREX
        assert symbol2.Market == Market.FOREX
    
    def test_create_trade_bar(self):
        """Test create_trade_bar function."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        bar = create_trade_bar(symbol, time, 150.0, 155.0, 148.0, 152.0, 1000)
        
        assert bar.Symbol == symbol
        assert bar.Time == time
        assert bar.Open == 150.0
        assert bar.High == 155.0
        assert bar.Low == 148.0
        assert bar.Close == 152.0
        assert bar.Volume == 1000
        assert bar.Period == Resolution.MINUTE
    
    def test_create_quote_bar(self):
        """Test create_quote_bar function."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        bid_bar = TradeBar(symbol, time, 149.0, 149.5, 148.5, 149.2, 500)
        ask_bar = TradeBar(symbol, time, 150.0, 150.5, 149.8, 150.2, 500)
        
        quote_bar = create_quote_bar(symbol, time, bid_bar, ask_bar)
        
        assert quote_bar.Symbol == symbol
        assert quote_bar.Time == time
        assert quote_bar.Bid == bid_bar
        assert quote_bar.Ask == ask_bar
        assert quote_bar.Period == Resolution.MINUTE
    
    def test_create_portfolio(self):
        """Test create_portfolio function."""
        portfolio = create_portfolio(50000.0)
        assert portfolio.Cash == 50000.0
        assert len(portfolio.Holdings) == 0
        
        portfolio2 = create_portfolio()  # Default cash
        assert portfolio2.Cash == 100000.0


class TestJSONSerialization:
    """Test JSON serialization across all classes."""
    
    def test_complete_serialization_cycle(self):
        """Test complete serialization cycle for all classes."""
        # Create a complex portfolio with all data types
        portfolio = create_portfolio(100000.0)
        
        # Add securities with holdings
        symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
        holding = SecurityHolding(symbol, 100, 150.0, 1000.0)
        portfolio.Holdings["AAPL"] = holding
        
        # Create trade bars
        time = datetime.now()
        trade_bar = create_trade_bar(symbol, time, 150.0, 155.0, 148.0, 152.0, 1000)
        
        # Create quote bar
        bid_bar = TradeBar(symbol, time, 149.0, 149.5, 148.5, 149.2, 500)
        ask_bar = TradeBar(symbol, time, 150.0, 150.5, 149.8, 150.2, 500)
        quote_bar = create_quote_bar(symbol, time, bid_bar, ask_bar)
        
        # Test serialization for each class
        classes_to_test = [symbol, holding, portfolio, trade_bar, quote_bar]
        
        for obj in classes_to_test:
            # Test to_dict and from_dict
            data = obj.to_dict()
            obj2 = obj.__class__.from_dict(data)
            
            # Test to_json and from_json
            json_str = obj.to_json()
            obj3 = obj.__class__.from_json(json_str)
            
            # Verify objects are equivalent
            assert obj2.to_dict() == obj.to_dict()
            assert obj3.to_dict() == obj.to_dict()
    
    def test_json_contains_expected_fields(self):
        """Test that JSON contains expected fields."""
        symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
        json_str = symbol.to_json()
        
        # Parse JSON and check fields
        data = json.loads(json_str)
        assert 'Value' in data
        assert 'SecurityType' in data
        assert 'Market' in data
        assert data['Value'] == "AAPL"
        assert data['SecurityType'] == "equity"
        assert data['Market'] == "us_equity"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_quantity_holdings(self):
        """Test holdings with zero quantity."""
        symbol = Symbol("AAPL")
        holding = SecurityHolding(symbol, 0, 150.0)
        
        assert holding.IsFlat
        assert not holding.IsLong
        assert not holding.IsShort
        assert holding.AbsoluteQuantity == 0
        assert holding.UnrealizedProfitPercent == 0.0
    
    def test_negative_average_price(self):
        """Test holdings with negative average price."""
        symbol = Symbol("AAPL")
        holding = SecurityHolding(symbol, 100, -150.0)
        
        # Should handle gracefully
        assert holding.AveragePrice == -150.0
        assert holding.UnrealizedProfitPercent == 0.0  # Division by zero protection
    
    def test_empty_portfolio(self):
        """Test empty portfolio operations."""
        portfolio = Portfolio(100000.0)
        
        assert portfolio.TotalHoldingsValue == 0.0
        assert portfolio.TotalUnrealizedProfit == 0.0
        assert portfolio.CashPercent == 100.0
        assert portfolio.GetHolding("UNKNOWN") is None
    
    def test_portfolio_with_zero_cash(self):
        """Test portfolio with zero cash."""
        portfolio = Portfolio(0.0)
        
        assert portfolio.Cash == 0.0
        assert portfolio.TotalPortfolioValue == 0.0
        assert portfolio.CashPercent == 0.0
    
    def test_trade_bar_with_same_open_close(self):
        """Test trade bar where open equals close."""
        symbol = Symbol("AAPL")
        time = datetime.now()
        bar = TradeBar(symbol, time, 150.0, 155.0, 148.0, 150.0, 1000)
        
        assert bar.IsDoji
        assert not bar.IsUp
        assert not bar.IsDown
        assert bar.Body == 0.0


class TestResolutionEnum:
    def test_resolution_enum_values(self):
        assert Resolution.SECOND.value == "second"
        assert Resolution.MINUTE.value == "minute"
        assert Resolution.HOUR.value == "hour"
        assert Resolution.DAILY.value == "daily"
        # Conversion from string
        assert Resolution("minute") == Resolution.MINUTE
        assert str(Resolution.HOUR) == "Resolution.HOUR"


def generate_mock_data(num=1000):
    """Generate a list of mock TradeBar objects for performance testing."""
    symbol = Symbol("MOCK")
    now = datetime.now()
    bars = [
        TradeBar(symbol, now, 100.0, 105.0, 95.0, 102.0, 1000)
        for _ in range(num)
    ]
    return bars


class TestPerformance:
    def test_bulk_serialization_performance(self):
        bars = generate_mock_data(1000)
        start = time.perf_counter()
        json_strs = [bar.to_json() for bar in bars]
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0  # Should serialize 1000 bars in under 1 second
        # Deserialize
        start = time.perf_counter()
        bars2 = [TradeBar.from_json(js) for js in json_strs]
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0
        assert all(isinstance(b, TradeBar) for b in bars2)


class TestMockDataGenerator:
    def test_generate_mock_data(self):
        bars = generate_mock_data(10)
        assert len(bars) == 10
        for bar in bars:
            assert isinstance(bar, TradeBar)
            assert bar.Open == 100.0
            assert bar.High == 105.0
            assert bar.Low == 95.0
            assert bar.Close == 102.0
            assert bar.Volume == 1000 