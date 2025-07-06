"""
Tests for QCAlgorithm and related classes.
"""

import pytest
import logging
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date

from lean_lite.algorithm import (
    QCAlgorithm, Symbol, Security, OrderEvent, SecurityType
)


class TestSymbol:
    """Test cases for Symbol class."""
    
    def test_symbol_creation(self):
        """Test symbol creation."""
        symbol = Symbol("AAPL", SecurityType.EQUITY)
        
        assert symbol.ticker == "AAPL"
        assert symbol.security_type == SecurityType.EQUITY
        assert str(symbol) == "AAPL"
        assert repr(symbol) == "Symbol('AAPL', equity)"
    
    def test_symbol_case_normalization(self):
        """Test that ticker is converted to uppercase."""
        symbol = Symbol("aapl", SecurityType.EQUITY)
        assert symbol.ticker == "AAPL"
    
    def test_default_security_type(self):
        """Test default security type."""
        symbol = Symbol("AAPL")
        assert symbol.security_type == SecurityType.EQUITY


class TestSecurity:
    """Test cases for Security class."""
    
    def test_security_creation(self):
        """Test security creation."""
        symbol = Symbol("AAPL")
        security = Security(symbol, price=150.0, volume=1000)
        
        assert security.symbol == symbol
        assert security.price == 150.0
        assert security.volume == 1000
        assert security.last_update is not None
    
    def test_security_default_values(self):
        """Test security default values."""
        symbol = Symbol("AAPL")
        security = Security(symbol)
        
        assert security.price == 0.0
        assert security.volume == 0
        assert security.last_update is not None


class TestOrderEvent:
    """Test cases for OrderEvent class."""
    
    def test_order_event_creation(self):
        """Test order event creation."""
        order_event = OrderEvent(
            order_id="123",
            symbol="AAPL",
            quantity=100,
            side="buy",
            status="filled",
            filled_quantity=100,
            filled_price=150.0
        )
        
        assert order_event.order_id == "123"
        assert order_event.symbol == "AAPL"
        assert order_event.quantity == 100
        assert order_event.side == "buy"
        assert order_event.status == "filled"
        assert order_event.filled_quantity == 100
        assert order_event.filled_price == 150.0
        assert order_event.time is not None


class TestQCAlgorithm:
    """Test cases for QCAlgorithm class."""
    
    def _create_test_algorithm(self):
        """Helper method to create a concrete QCAlgorithm instance."""
        class TestAlgo(QCAlgorithm):
            def Initialize(self):
                pass
            def OnData(self, data):
                pass
        return TestAlgo()
    
    def test_qc_algorithm_initialization(self):
        """Test QCAlgorithm initialization."""
        algorithm = self._create_test_algorithm()
        
        assert algorithm.cash == 100000.0
        assert algorithm.portfolio_value == 100000.0
        assert algorithm.start_date is None
        assert algorithm.end_date is None
        assert len(algorithm.securities) == 0
        assert len(algorithm.symbols) == 0
        assert algorithm.logger is not None
    
    def test_set_start_date(self):
        """Test SetStartDate method."""
        algorithm = self._create_test_algorithm()
        algorithm.SetStartDate(2023, 1, 1)
        
        assert algorithm.start_date == date(2023, 1, 1)
    
    def test_set_end_date(self):
        """Test SetEndDate method."""
        algorithm = self._create_test_algorithm()
        algorithm.SetEndDate(2023, 12, 31)
        
        assert algorithm.end_date == date(2023, 12, 31)
    
    def test_set_cash(self):
        """Test SetCash method."""
        algorithm = self._create_test_algorithm()
        algorithm.SetCash(50000)
        
        assert algorithm.cash == 50000
        assert algorithm.portfolio_value == 50000
    
    def test_add_equity(self):
        """Test AddEquity method."""
        algorithm = self._create_test_algorithm()
        symbol = algorithm.AddEquity("AAPL")
        
        assert isinstance(symbol, Symbol)
        assert symbol.ticker == "AAPL"
        assert symbol.security_type == SecurityType.EQUITY
        assert "AAPL" in algorithm.symbols
        assert "AAPL" in algorithm.securities
    
    def test_add_forex(self):
        """Test AddForex method."""
        algorithm = self._create_test_algorithm()
        symbol = algorithm.AddForex("EURUSD")
        
        assert isinstance(symbol, Symbol)
        assert symbol.ticker == "EURUSD"
        assert symbol.security_type == SecurityType.FOREX
        assert "EURUSD" in algorithm.symbols
        assert "EURUSD" in algorithm.securities
    
    def test_add_crypto(self):
        """Test AddCrypto method."""
        algorithm = self._create_test_algorithm()
        symbol = algorithm.AddCrypto("BTCUSD")
        
        assert isinstance(symbol, Symbol)
        assert symbol.ticker == "BTCUSD"
        assert symbol.security_type == SecurityType.CRYPTO
        assert "BTCUSD" in algorithm.symbols
        assert "BTCUSD" in algorithm.securities
    
    def test_buy_with_string_symbol(self):
        """Test Buy method with string symbol."""
        algorithm = self._create_test_algorithm()
        algorithm.broker = Mock()  # Set mock broker
        with patch.object(algorithm.broker, 'buy') as mock_buy:
            algorithm.Buy("AAPL", 100)
            mock_buy.assert_called_once_with("AAPL", 100)
    
    def test_buy_with_symbol_object(self):
        """Test Buy method with Symbol object."""
        algorithm = self._create_test_algorithm()
        algorithm.broker = Mock()  # Set mock broker
        symbol = Symbol("AAPL")
        with patch.object(algorithm.broker, 'buy') as mock_buy:
            algorithm.Buy(symbol, 100)
            mock_buy.assert_called_once_with("AAPL", 100)
    
    def test_sell_with_string_symbol(self):
        """Test Sell method with string symbol."""
        algorithm = self._create_test_algorithm()
        algorithm.broker = Mock()  # Set mock broker
        with patch.object(algorithm.broker, 'sell') as mock_sell:
            algorithm.Sell("AAPL", 100)
            mock_sell.assert_called_once_with("AAPL", 100)
    
    def test_sell_with_symbol_object(self):
        """Test Sell method with Symbol object."""
        algorithm = self._create_test_algorithm()
        algorithm.broker = Mock()  # Set mock broker
        symbol = Symbol("AAPL")
        with patch.object(algorithm.broker, 'sell') as mock_sell:
            algorithm.Sell(symbol, 100)
            mock_sell.assert_called_once_with("AAPL", 100)
    
    def test_set_holdings(self):
        """Test SetHoldings method."""
        algorithm = self._create_test_algorithm()
        algorithm.AddEquity("AAPL")
        algorithm.UpdateSecurityPrice("AAPL", 150.0)
        
        with patch.object(algorithm, 'has_position') as mock_has_position:
            mock_has_position.return_value = False
            
            with patch.object(algorithm, 'Buy') as mock_buy:
                algorithm.SetHoldings("AAPL", 0.5)  # 50% of portfolio
                
                # Should buy approximately 333 shares (50000 / 150)
                mock_buy.assert_called_once()
                args = mock_buy.call_args
                assert args[0][0] == "AAPL"
                assert args[0][1] > 0
    
    def test_liquidate(self):
        """Test Liquidate method."""
        algorithm = self._create_test_algorithm()
        
        with patch.object(algorithm, 'has_position') as mock_has_position:
            mock_has_position.return_value = True
            
            with patch.object(algorithm, 'Sell') as mock_sell:
                algorithm.Liquidate("AAPL")
                mock_sell.assert_called_once_with("AAPL", 100)  # Default quantity
    
    def test_get_last_known_price(self):
        """Test GetLastKnownPrice method."""
        algorithm = self._create_test_algorithm()
        algorithm.AddEquity("AAPL")
        algorithm.UpdateSecurityPrice("AAPL", 150.0)
        
        price = algorithm.GetLastKnownPrice("AAPL")
        assert price == 150.0
    
    def test_get_last_known_price_not_found(self):
        """Test GetLastKnownPrice for non-existent symbol."""
        algorithm = self._create_test_algorithm()
        
        price = algorithm.GetLastKnownPrice("UNKNOWN")
        assert price == 0.0
    
    def test_update_security_price(self):
        """Test UpdateSecurityPrice method."""
        algorithm = self._create_test_algorithm()
        algorithm.AddEquity("AAPL")
        
        algorithm.UpdateSecurityPrice("AAPL", 150.0, 1000)
        
        security = algorithm.securities["AAPL"]
        assert security.price == 150.0
        assert security.volume == 1000
        assert security.last_update is not None
    
    def test_on_data_updates_prices(self):
        """Test that OnData updates security prices."""
        algorithm = self._create_test_algorithm()
        algorithm.AddEquity("AAPL")
        
        data = {"AAPL": {"close": 150.0, "timestamp": "2023-01-01T10:00:00"}}
        
        with patch.object(algorithm, 'OnData') as mock_on_data:
            algorithm.on_data(data)
            
            # Check that price was updated
            assert algorithm.securities["AAPL"].price == 150.0
            
            # Check that OnData was called
            mock_on_data.assert_called_once_with(data)
    
    def test_on_order_filled_creates_order_event(self):
        """Test that on_order_filled creates OrderEvent."""
        algorithm = self._create_test_algorithm()
        
        # Mock order object
        mock_order = Mock()
        mock_order.id = "123"
        mock_order.symbol = "AAPL"
        mock_order.qty = 100
        mock_order.side.value = "BUY"
        mock_order.status.value = "FILLED"
        mock_order.filled_qty = 100
        mock_order.filled_avg_price = 150.0
        
        with patch.object(algorithm, 'OnOrderEvent') as mock_on_order_event:
            algorithm.on_order_filled(mock_order)
            
            # Check that OnOrderEvent was called with OrderEvent object
            mock_on_order_event.assert_called_once()
            order_event = mock_on_order_event.call_args[0][0]
            assert isinstance(order_event, OrderEvent)
            assert order_event.order_id == "123"
            assert order_event.symbol == "AAPL"
            assert order_event.quantity == 100
            assert order_event.side == "buy"
            assert order_event.status == "filled"
            assert order_event.filled_quantity == 100
            assert order_event.filled_price == 150.0
    
    def test_logging_methods(self):
        """Test logging methods."""
        algorithm = self._create_test_algorithm()
        
        with patch.object(algorithm.logger, 'info') as mock_info:
            algorithm.Log("Test message")
            mock_info.assert_called_once_with("Test message")
        
        with patch.object(algorithm.logger, 'debug') as mock_debug:
            algorithm.Debug("Debug message")
            mock_debug.assert_called_once_with("Debug message")
        
        with patch.object(algorithm.logger, 'error') as mock_error:
            algorithm.Error("Error message")
            mock_error.assert_called_once_with("Error message")


class TestQCAlgorithmAbstractMethods:
    """Test that QCAlgorithm requires abstract methods to be implemented."""
    
    def test_qc_algorithm_is_abstract(self):
        """Test that QCAlgorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            QCAlgorithm()
    
    def test_concrete_subclass_works(self):
        """Test that a concrete subclass can be instantiated."""
        class ConcreteStrategy(QCAlgorithm):
            def Initialize(self):
                pass
            
            def OnData(self, data):
                pass
        
        strategy = ConcreteStrategy()
        assert isinstance(strategy, QCAlgorithm)


# --- Concrete Test Strategy ---
class TestStrategy(QCAlgorithm):
    def __init__(self):
        super().__init__()
        self.init_called = False
        self.on_data_called = False
        self.on_order_event_called = False
        self.on_end_of_day_called = False
        self.last_data = None
        self.last_order_event = None
        self.last_eod_symbol = None

    def Initialize(self):
        self.init_called = True
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(50000)
        self.spy = self.AddEquity("SPY")
        self.eurusd = self.AddForex("EURUSD")
        self.btcusd = self.AddCrypto("BTCUSD")

    def OnData(self, data):
        self.on_data_called = True
        self.last_data = data

    def OnOrderEvent(self, order_event):
        self.on_order_event_called = True
        self.last_order_event = order_event

    def OnEndOfDay(self, symbol):
        self.on_end_of_day_called = True
        self.last_eod_symbol = symbol


# --- Tests ---
def test_qcalgorithm_instantiation():
    strategy = TestStrategy()
    assert isinstance(strategy, QCAlgorithm)
    assert hasattr(strategy, 'Initialize')
    assert hasattr(strategy, 'OnData')
    assert hasattr(strategy, 'OnOrderEvent')
    assert hasattr(strategy, 'OnEndOfDay')

def test_set_start_end_cash():
    strategy = TestStrategy()
    strategy.SetStartDate(2021, 5, 10)
    strategy.SetEndDate(2021, 12, 31)
    strategy.SetCash(12345.67)
    assert strategy.start_date == date(2021, 5, 10)
    assert strategy.end_date == date(2021, 12, 31)
    assert strategy.cash == 12345.67
    assert strategy.portfolio_value == 12345.67

def test_add_equity_forex_crypto():
    strategy = TestStrategy()
    eq = strategy.AddEquity("AAPL")
    fx = strategy.AddForex("EURUSD")
    cr = strategy.AddCrypto("BTCUSD")
    assert isinstance(eq, Symbol)
    assert eq.ticker == "AAPL"
    assert eq.security_type == SecurityType.EQUITY
    assert isinstance(fx, Symbol)
    assert fx.ticker == "EURUSD"
    assert fx.security_type == SecurityType.FOREX
    assert isinstance(cr, Symbol)
    assert cr.ticker == "BTCUSD"
    assert cr.security_type == SecurityType.CRYPTO
    assert "AAPL" in strategy.securities
    assert "EURUSD" in strategy.securities
    assert "BTCUSD" in strategy.securities

def test_algorithm_state_and_lifecycle():
    strategy = TestStrategy()
    strategy.Initialize()
    assert strategy.init_called
    assert strategy.start_date == date(2022, 1, 1)
    assert strategy.end_date == date(2022, 12, 31)
    assert strategy.cash == 50000
    assert "SPY" in strategy.securities
    assert "EURUSD" in strategy.securities
    assert "BTCUSD" in strategy.securities
    # Simulate data event
    data = {"SPY": {"close": 400.0, "timestamp": "2022-01-03T10:00:00"}}
    strategy.on_data(data)
    assert strategy.on_data_called
    assert strategy.last_data == data
    # Simulate order event
    mock_order_event = MagicMock()
    strategy.OnOrderEvent(mock_order_event)
    assert strategy.on_order_event_called
    assert strategy.last_order_event == mock_order_event
    # Simulate end of day
    strategy.OnEndOfDay("SPY")
    assert strategy.on_end_of_day_called
    assert strategy.last_eod_symbol == "SPY"

def test_logging_functionality(caplog):
    strategy = TestStrategy()
    with caplog.at_level(logging.DEBUG):
        strategy.Log("Test log message")
        strategy.Debug("Test debug message")
        strategy.Error("Test error message")
    assert any("Test log message" in m for m in caplog.messages)
    assert any("Test debug message" in m for m in caplog.messages)
    assert any("Test error message" in m for m in caplog.messages)

def test_error_handling_invalid_inputs():
    strategy = TestStrategy()
    # Invalid date
    with pytest.raises(ValueError):
        strategy.SetStartDate(2022, 2, 30)
    # Invalid cash
    with pytest.raises(TypeError):
        strategy.SetCash("not_a_number")

def test_mock_methods():
    strategy = TestStrategy()
    strategy.broker = Mock()  # Set mock broker
    # Mock buy/sell to avoid real trading logic
    mock_order = Mock()
    mock_order.id = "ORDER123"
    with patch.object(strategy.broker, 'buy', return_value=mock_order) as mock_buy:
        strategy.Buy("AAPL", 10)
        mock_buy.assert_called_once_with("AAPL", 10)
    with patch.object(strategy.broker, 'sell', return_value=mock_order) as mock_sell:
        strategy.Sell("AAPL", 5)
        mock_sell.assert_called_once_with("AAPL", 5)

def test_performance_timing():
    strategy = TestStrategy()
    start = time.perf_counter()
    for _ in range(1000):
        strategy.SetCash(100000)
        strategy.SetStartDate(2022, 1, 1)
        strategy.SetEndDate(2022, 12, 31)
        strategy.AddEquity("SPY")
    elapsed = time.perf_counter() - start
    # Should be fast (arbitrary threshold, e.g., < 0.5s for 1000 calls)
    assert elapsed < 0.5

def test_algorithm_lifecycle():
    strategy = TestStrategy()
    # Full lifecycle
    strategy.Initialize()
    data = {"SPY": {"close": 400.0, "timestamp": "2022-01-03T10:00:00"}}
    strategy.on_data(data)
    mock_order_event = MagicMock()
    strategy.on_order_filled(mock_order_event)
    strategy.OnEndOfDay("SPY")
    # All hooks should have been called
    assert strategy.init_called
    assert strategy.on_data_called
    assert strategy.on_order_event_called
    assert strategy.on_end_of_day_called 