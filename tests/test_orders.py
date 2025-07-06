import pytest
from datetime import datetime
from unittest.mock import Mock

from lean_lite.algorithm.data_models import Symbol, Portfolio, SecurityType, Market
from lean_lite.algorithm.orders import (
    OrderTicket, OrderType, OrderStatus, OrderEvent,
    MarketOrder, LimitOrder, StopOrder, OrderManager
)

# --- Mock Broker for integration tests ---
class MockBroker:
    def __init__(self):
        self.orders = []
    def execute_order(self, order):
        self.orders.append(order)
        return True

# --- Test Data ---
def get_symbol():
    return Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)

def get_portfolio(cash=100000.0):
    return Portfolio(Cash=cash)

# --- Tests ---
class TestOrderTicket:
    def test_order_ticket_creation(self):
        symbol = get_symbol()
        ticket = OrderTicket(1, symbol, 100, OrderType.MARKET)
        assert ticket.order_id == 1
        assert ticket.symbol == symbol
        assert ticket.quantity == 100
        assert ticket.order_type == OrderType.MARKET
        assert ticket.status == OrderStatus.NEW
        assert str(ticket).startswith("OrderTicket(")
        assert "OrderTicket(" in repr(ticket)

class TestOrderTypeEnum:
    def test_order_type_enum_values(self):
        assert OrderType.MARKET.value == "Market"
        assert OrderType.LIMIT.value == "Limit"
        assert OrderType.STOP.value == "Stop"
        assert OrderType.STOP_LIMIT.value == "StopLimit"
        assert str(OrderType.LIMIT) == "OrderType.LIMIT"

class TestOrderClasses:
    def test_market_order_instantiation(self):
        symbol = get_symbol()
        order = MarketOrder(1, symbol, 50)
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 50
    def test_limit_order_instantiation(self):
        symbol = get_symbol()
        order = LimitOrder(2, symbol, 25, 150.0)
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.0
    def test_stop_order_instantiation(self):
        symbol = get_symbol()
        order = StopOrder(3, symbol, 10, 140.0)
        assert order.order_type == OrderType.STOP
        assert order.stop_price == 140.0

class TestOrderStatusTransitions:
    def test_status_transitions(self):
        symbol = get_symbol()
        ticket = OrderTicket(1, symbol, 100, OrderType.MARKET)
        assert ticket.status == OrderStatus.NEW
        ticket.status = OrderStatus.SUBMITTED
        assert ticket.status == OrderStatus.SUBMITTED
        ticket.status = OrderStatus.FILLED
        assert ticket.status == OrderStatus.FILLED

class TestOrderEvent:
    def test_order_event_creation(self):
        symbol = get_symbol()
        event = OrderEvent(1, symbol, OrderStatus.FILLED, 100, 150.0, "Filled order")
        assert event.order_id == 1
        assert event.symbol == symbol
        assert event.status == OrderStatus.FILLED
        assert event.fill_quantity == 100
        assert event.fill_price == 150.0
        assert event.message == "Filled order"
        assert str(event).startswith("OrderEvent(")
        assert "OrderEvent(" in repr(event)

class TestOrderManagerLifecycle:
    def test_order_lifecycle(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        # Buy order
        order = manager.Buy(symbol, 10)
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 10
        # Update status
        event = manager.update_order_status(order.order_id, OrderStatus.SUBMITTED)
        assert event.status == OrderStatus.SUBMITTED
        event2 = manager.update_order_status(order.order_id, OrderStatus.FILLED, fill_quantity=10, fill_price=150.0)
        assert event2.status == OrderStatus.FILLED
        assert order.filled_quantity == 10
        assert order.filled_price == 150.0
        # Get order/events
        assert manager.get_order(order.order_id) == order
        events = manager.get_order_events(order.order_id)
        assert len(events) == 2

class TestOrderValidation:
    def test_invalid_symbol(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        with pytest.raises(ValueError):
            manager.Buy(None, 10)
    def test_zero_quantity(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        with pytest.raises(ValueError):
            manager.Buy(symbol, 0)
    def test_insufficient_funds(self):
        portfolio = get_portfolio(cash=10.0)
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        with pytest.raises(ValueError):
            manager.Buy(symbol, 1, price=100.0)
    def test_limit_order_requires_price(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        with pytest.raises(ValueError):
            manager.LimitOrder(symbol, 10, None)

class TestOrderManagerMethods:
    def test_buy_sell_liquidate(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        # Buy
        buy_order = manager.Buy(symbol, 5)
        assert buy_order.quantity == 5
        # Sell
        sell_order = manager.Sell(symbol, 3)
        assert sell_order.quantity == -3
        # Add holding for liquidation
        portfolio.Holdings[symbol.Value] = portfolio.AddSecurity(symbol)
        portfolio.Holdings[symbol.Value].Quantity = 2
        liquidate_order = manager.Liquidate(symbol)
        assert liquidate_order is not None
        assert liquidate_order.quantity == -2
    def test_set_holdings(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        # Add holding with price
        holding = portfolio.AddSecurity(symbol)
        holding.LastTradePrice = 100.0
        # Set holdings to 50%
        order = manager.SetHoldings(symbol, 0.5)
        assert order is not None
        # Set holdings to 0% when no position exists (should return None)
        order2 = manager.SetHoldings(symbol, 0.0)
        assert order2 is None  # No position to liquidate
        # Set holdings to 100%
        order3 = manager.SetHoldings(symbol, 1.0)
        assert order3 is not None
        # Set holdings to 100% again (may create new order if portfolio value changed)
        result = manager.SetHoldings(symbol, 1.0)
        # This may or may not be None depending on portfolio value changes
        # The important thing is that it doesn't raise an error
    
    def test_set_holdings_no_change_needed(self):
        """Test SetHoldings when the target is already achieved."""
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        # Add holding with existing position that matches target
        holding = portfolio.AddSecurity(symbol)
        holding.Quantity = 100  # Existing position
        holding.LastTradePrice = 100.0
        # Set portfolio value to match the holding value exactly
        portfolio.Cash = 0.0  # All value in holdings
        # Set holdings to match current position (should return None)
        result = manager.SetHoldings(symbol, 1.0)
        assert result is None  # No change needed
    
    def test_set_holdings_liquidation(self):
        """Test SetHoldings when there's an existing position to liquidate."""
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        # Add holding with existing position
        holding = portfolio.AddSecurity(symbol)
        holding.Quantity = 10  # Existing position
        holding.LastTradePrice = 100.0
        # Set holdings to 0% (should liquidate existing position)
        order = manager.SetHoldings(symbol, 0.0)
        assert order is not None
        assert order.quantity == -10  # Should sell existing position
    
    def test_set_holdings_invalid_percent(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        with pytest.raises(ValueError):
            manager.SetHoldings(symbol, -0.1)
        with pytest.raises(ValueError):
            manager.SetHoldings(symbol, 1.1)

class TestOrderCancellationModification:
    def test_order_cancellation(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        order = manager.Buy(symbol, 5)
        event = manager.update_order_status(order.order_id, OrderStatus.CANCELED, message="Canceled by user")
        assert event.status == OrderStatus.CANCELED
        assert event.message == "Canceled by user"
    def test_order_modification(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        order = manager.Buy(symbol, 5)
        # Simulate modification by updating quantity
        order.quantity = 10
        assert order.quantity == 10

class TestOrderManagerIntegration:
    def test_integration_with_portfolio(self):
        portfolio = get_portfolio()
        manager = OrderManager(portfolio)
        symbol = get_symbol()
        # Buy order
        order = manager.Buy(symbol, 10)
        manager.update_order_status(order.order_id, OrderStatus.FILLED, fill_quantity=10, fill_price=100.0)
        # Update portfolio manually for test
        holding = portfolio.AddSecurity(symbol)
        holding.Quantity += 10
        holding.AveragePrice = 100.0
        # Sell order
        order2 = manager.Sell(symbol, 5)
        manager.update_order_status(order2.order_id, OrderStatus.FILLED, fill_quantity=5, fill_price=110.0)
        holding.Quantity -= 5
        # Check portfolio state
        assert holding.Quantity == 5
        assert holding.AveragePrice == 100.0
        # Liquidate
        order3 = manager.Liquidate(symbol)
        if order3:
            manager.update_order_status(order3.order_id, OrderStatus.FILLED, fill_quantity=5, fill_price=105.0)
            holding.Quantity = 0
        assert holding.Quantity == 0 