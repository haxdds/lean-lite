import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List, Union

from .data_models import Symbol, Portfolio

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"

class OrderStatus(Enum):
    NEW = "New"
    SUBMITTED = "Submitted"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELED = "Canceled"
    REJECTED = "Rejected"

@dataclass
class OrderTicket:
    order_id: int
    symbol: Symbol
    quantity: int
    order_type: OrderType
    status: OrderStatus = OrderStatus.NEW
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    filled_quantity: int = 0
    filled_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    tag: Optional[str] = None
    error: Optional[str] = None

    def __str__(self):
        return f"OrderTicket({self.order_id}, {self.symbol}, {self.quantity}, {self.order_type}, {self.status})"
    def __repr__(self):
        return (f"OrderTicket(order_id={self.order_id}, symbol={self.symbol}, quantity={self.quantity}, "
                f"order_type={self.order_type}, status={self.status}, price={self.price}, "
                f"stop_price={self.stop_price}, limit_price={self.limit_price}, filled_quantity={self.filled_quantity}, "
                f"filled_price={self.filled_price}, timestamp={self.timestamp}, tag={self.tag}, error={self.error})")

@dataclass
class OrderEvent:
    order_id: int
    symbol: Symbol
    status: OrderStatus
    fill_quantity: int = 0
    fill_price: float = 0.0
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self):
        return f"OrderEvent({self.order_id}, {self.symbol}, {self.status}, {self.fill_quantity}, {self.fill_price})"
    def __repr__(self):
        return (f"OrderEvent(order_id={self.order_id}, symbol={self.symbol}, status={self.status}, "
                f"fill_quantity={self.fill_quantity}, fill_price={self.fill_price}, message={self.message}, "
                f"timestamp={self.timestamp})")

@dataclass
class MarketOrder(OrderTicket):
    def __init__(self, order_id: int, symbol: Symbol, quantity: int, tag: Optional[str] = None):
        super().__init__(order_id, symbol, quantity, OrderType.MARKET, tag=tag)

@dataclass
class LimitOrder(OrderTicket):
    def __init__(self, order_id: int, symbol: Symbol, quantity: int, limit_price: float, tag: Optional[str] = None):
        super().__init__(order_id, symbol, quantity, OrderType.LIMIT, limit_price=limit_price, tag=tag)

@dataclass
class StopOrder(OrderTicket):
    def __init__(self, order_id: int, symbol: Symbol, quantity: int, stop_price: float, tag: Optional[str] = None):
        super().__init__(order_id, symbol, quantity, OrderType.STOP, stop_price=stop_price, tag=tag)

class OrderManager:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.orders: Dict[int, OrderTicket] = {}
        self.order_events: List[OrderEvent] = []
        self.next_order_id = 1

    def _validate_order(self, symbol: Symbol, quantity: int, order_type: OrderType, price: Optional[float] = None) -> Optional[str]:
        if not symbol or not isinstance(symbol, Symbol):
            return "Invalid symbol."
        if quantity == 0:
            return "Order quantity cannot be zero."
        if order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and price is None:
            return "Limit/StopLimit orders require a price."
        # Sufficient funds check for buy orders
        if quantity > 0:
            holding = self.portfolio.GetHolding(symbol)
            cash_needed = price * quantity if price else 0
            if self.portfolio.Cash < cash_needed:
                return f"Insufficient funds: Need ${cash_needed:.2f}, have ${self.portfolio.Cash:.2f}."
        return None

    def _create_order(self, order_cls, symbol: Symbol, quantity: int, **kwargs) -> OrderTicket:
        order_id = self.next_order_id
        self.next_order_id += 1
        order = order_cls(order_id, symbol, quantity, **kwargs)
        self.orders[order_id] = order
        logger.info(f"Created order: {order}")
        return order

    def Buy(self, symbol: Symbol, quantity: int, price: Optional[float] = None) -> OrderTicket:
        error = self._validate_order(symbol, quantity, OrderType.MARKET, price)
        if error:
            logger.error(f"Buy order validation failed: {error}")
            raise ValueError(error)
        return self._create_order(MarketOrder, symbol, abs(quantity))

    def Sell(self, symbol: Symbol, quantity: int, price: Optional[float] = None) -> OrderTicket:
        error = self._validate_order(symbol, -quantity, OrderType.MARKET, price)
        if error:
            logger.error(f"Sell order validation failed: {error}")
            raise ValueError(error)
        return self._create_order(MarketOrder, symbol, -abs(quantity))

    def LimitOrder(self, symbol: Symbol, quantity: int, limit_price: float) -> OrderTicket:
        error = self._validate_order(symbol, quantity, OrderType.LIMIT, limit_price)
        if error:
            logger.error(f"Limit order validation failed: {error}")
            raise ValueError(error)
        return self._create_order(LimitOrder, symbol, quantity, limit_price=limit_price)

    def StopOrder(self, symbol: Symbol, quantity: int, stop_price: float) -> OrderTicket:
        error = self._validate_order(symbol, quantity, OrderType.STOP, stop_price)
        if error:
            logger.error(f"Stop order validation failed: {error}")
            raise ValueError(error)
        return self._create_order(StopOrder, symbol, quantity, stop_price=stop_price)

    def Liquidate(self, symbol: Symbol) -> Optional[OrderTicket]:
        holding = self.portfolio.GetHolding(symbol)
        if not holding or holding.Quantity == 0:
            logger.info(f"No position to liquidate for {symbol}")
            return None
        if holding.Quantity > 0:
            return self.Sell(symbol, holding.Quantity)
        else:
            return self.Buy(symbol, abs(holding.Quantity))

    def SetHoldings(self, symbol: Symbol, target_percent: float) -> Optional[OrderTicket]:
        if not (0.0 <= target_percent <= 1.0):
            logger.error("Target percent must be between 0.0 and 1.0.")
            raise ValueError("Target percent must be between 0.0 and 1.0.")
        current_holding = self.portfolio.GetHolding(symbol)
        current_qty = current_holding.Quantity if current_holding else 0
        price = current_holding.LastTradePrice if current_holding else 0.0
        target_value = self.portfolio.TotalPortfolioValue * target_percent
        target_qty = int(target_value / price) if price > 0 else 0
        delta_qty = target_qty - current_qty
        if delta_qty > 0:
            return self.Buy(symbol, delta_qty, price)
        elif delta_qty < 0:
            return self.Sell(symbol, abs(delta_qty), price)
        else:
            logger.info(f"Holdings for {symbol} already at target.")
            return None

    def update_order_status(self, order_id: int, status: OrderStatus, fill_quantity: int = 0, fill_price: float = 0.0, message: Optional[str] = None):
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"Order ID {order_id} not found for update.")
            return
        order.status = status
        if fill_quantity:
            order.filled_quantity += fill_quantity
            order.filled_price = fill_price
        event = OrderEvent(order_id, order.symbol, status, fill_quantity, fill_price, message)
        self.order_events.append(event)
        logger.info(f"Order updated: {event}")
        return event

    def get_order(self, order_id: int) -> Optional[OrderTicket]:
        return self.orders.get(order_id)

    def get_orders(self, symbol: Optional[Symbol] = None) -> List[OrderTicket]:
        if symbol:
            return [o for o in self.orders.values() if o.symbol == symbol]
        return list(self.orders.values())

    def get_order_events(self, order_id: Optional[int] = None) -> List[OrderEvent]:
        if order_id:
            return [e for e in self.order_events if e.order_id == order_id]
        return list(self.order_events) 