"""
Data models for Lean-Lite's QuantConnect compatibility.

This module provides comprehensive data structures that mirror QuantConnect's
data models for seamless compatibility and easy debugging.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class JSONSerializable(ABC):
    """Abstract base class for JSON serializable objects."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary for JSON serialization."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create object from dictionary."""
        pass
    
    def to_json(self) -> str:
        """Convert object to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create object from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class Resolution(Enum):
    """Data resolution for market data."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAILY = "daily"


class SecurityType(Enum):
    """Security types supported by Lean-Lite."""
    EQUITY = "equity"
    FOREX = "forex"
    CRYPTO = "crypto"
    OPTION = "option"
    FUTURE = "future"
    INDEX = "index"


class Market(Enum):
    """Market exchanges."""
    US_EQUITY = "us_equity"
    FOREX = "forex"
    CRYPTO = "crypto"
    US_OPTIONS = "us_options"
    US_FUTURES = "us_futures"


@dataclass
class Symbol(JSONSerializable):
    """
    Represents a trading symbol with its properties.
    
    This class mirrors QuantConnect's Symbol class for compatibility.
    """
    Value: str
    SecurityType: SecurityType = SecurityType.EQUITY
    Market: Market = Market.US_EQUITY
    
    def __post_init__(self):
        """Validate symbol after initialization."""
        if not self.Value or not isinstance(self.Value, str):
            raise ValueError("Symbol value must be a non-empty string")
        
        # Normalize symbol to uppercase
        self.Value = self.Value.upper()
    
    def __str__(self) -> str:
        """String representation of the symbol."""
        return f"{self.Value} ({self.SecurityType.value})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"Symbol(Value='{self.Value}', SecurityType={self.SecurityType}, Market={self.Market})"
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, Symbol):
            return False
        return (self.Value == other.Value and 
                self.SecurityType == other.SecurityType and 
                self.Market == other.Market)
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        return hash((self.Value, self.SecurityType, self.Market))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'Value': self.Value,
            'SecurityType': self.SecurityType.value,
            'Market': self.Market.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Symbol':
        """Create Symbol from dictionary."""
        return cls(
            Value=data['Value'],
            SecurityType=SecurityType(data['SecurityType']),
            Market=Market(data['Market'])
        )


@dataclass
class SecurityHolding(JSONSerializable):
    """
    Represents holdings for a specific security.
    
    Tracks position size, average price, and unrealized profit/loss.
    """
    Symbol: Symbol
    Quantity: int = 0
    AveragePrice: float = 0.0
    UnrealizedProfit: float = 0.0
    TotalFees: float = 0.0
    LastTradePrice: float = 0.0
    LastTradeTime: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize after creation."""
        if self.LastTradeTime is None:
            self.LastTradeTime = datetime.now()
    
    @property
    def AbsoluteQuantity(self) -> int:
        """Absolute quantity of holdings."""
        return abs(self.Quantity)
    
    @property
    def IsLong(self) -> bool:
        """Whether the position is long."""
        return self.Quantity > 0
    
    @property
    def IsShort(self) -> bool:
        """Whether the position is short."""
        return self.Quantity < 0
    
    @property
    def IsFlat(self) -> bool:
        """Whether the position is flat (no holdings)."""
        return self.Quantity == 0
    
    @property
    def UnrealizedProfitPercent(self) -> float:
        """Unrealized profit as percentage of average price."""
        if self.AveragePrice == 0 or self.Quantity == 0:
            return 0.0
        return (self.UnrealizedProfit / (self.AveragePrice * abs(self.Quantity))) * 100
    
    def __str__(self) -> str:
        """String representation."""
        direction = "LONG" if self.IsLong else "SHORT" if self.IsShort else "FLAT"
        return f"{self.Symbol.Value}: {self.Quantity} shares ({direction}) @ ${self.AveragePrice:.2f}"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"SecurityHolding(Symbol={self.Symbol}, Quantity={self.Quantity}, "
                f"AveragePrice={self.AveragePrice:.2f}, UnrealizedProfit={self.UnrealizedProfit:.2f})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'Symbol': self.Symbol.to_dict(),
            'Quantity': self.Quantity,
            'AveragePrice': self.AveragePrice,
            'UnrealizedProfit': self.UnrealizedProfit,
            'TotalFees': self.TotalFees,
            'LastTradePrice': self.LastTradePrice,
            'LastTradeTime': self.LastTradeTime.isoformat() if self.LastTradeTime else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityHolding':
        """Create SecurityHolding from dictionary."""
        return cls(
            Symbol=Symbol.from_dict(data['Symbol']),
            Quantity=data['Quantity'],
            AveragePrice=data['AveragePrice'],
            UnrealizedProfit=data['UnrealizedProfit'],
            TotalFees=data['TotalFees'],
            LastTradePrice=data['LastTradePrice'],
            LastTradeTime=datetime.fromisoformat(data['LastTradeTime']) if data['LastTradeTime'] else None
        )


@dataclass
class Security(JSONSerializable):
    """
    Represents a security with its configuration and current data.
    
    This class tracks the security's symbol, current price, volume, and holdings.
    """
    Symbol: Symbol
    Price: float = 0.0
    Volume: int = 0
    Holdings: SecurityHolding = field(init=False)
    LastUpdate: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize holdings after creation."""
        self.Holdings = SecurityHolding(self.Symbol)
        if self.LastUpdate is None:
            self.LastUpdate = datetime.now()
    
    def UpdatePrice(self, price: float, volume: int = 0):
        """Update the security's price and volume."""
        self.Price = price
        self.Volume = volume
        self.LastUpdate = datetime.now()
        
        # Update unrealized profit in holdings
        if not self.Holdings.IsFlat:
            self.Holdings.LastTradePrice = price
            self.Holdings.UnrealizedProfit = (
                (price - self.Holdings.AveragePrice) * self.Holdings.Quantity
            )
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.Symbol.Value}: ${self.Price:.2f} (Vol: {self.Volume})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"Security(Symbol={self.Symbol}, Price={self.Price:.2f}, "
                f"Volume={self.Volume}, Holdings={self.Holdings})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'Symbol': self.Symbol.to_dict(),
            'Price': self.Price,
            'Volume': self.Volume,
            'Holdings': self.Holdings.to_dict(),
            'LastUpdate': self.LastUpdate.isoformat() if self.LastUpdate else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Security':
        """Create Security from dictionary."""
        security = cls(
            Symbol=Symbol.from_dict(data['Symbol']),
            Price=data['Price'],
            Volume=data['Volume'],
            LastUpdate=datetime.fromisoformat(data['LastUpdate']) if data['LastUpdate'] else None
        )
        security.Holdings = SecurityHolding.from_dict(data['Holdings'])
        return security


@dataclass
class TradeBar(JSONSerializable):
    """
    Represents OHLCV (Open, High, Low, Close, Volume) data for a security.
    
    This class mirrors QuantConnect's TradeBar for compatibility.
    """
    Symbol: Symbol
    Time: datetime
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int
    Period: Resolution = Resolution.MINUTE
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.High < max(self.Open, self.Close):
            raise ValueError("High must be >= max(Open, Close)")
        if self.Low > min(self.Open, self.Close):
            raise ValueError("Low must be <= min(Open, Close)")
        if self.Volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def Price(self) -> float:
        """Current price (same as Close)."""
        return self.Close
    
    @property
    def Range(self) -> float:
        """Price range (High - Low)."""
        return self.High - self.Low
    
    @property
    def Body(self) -> float:
        """Body size (Close - Open)."""
        return self.Close - self.Open
    
    @property
    def IsUp(self) -> bool:
        """Whether the bar closed higher than it opened."""
        if self.IsDoji:
            return False
        return self.Close > self.Open
    
    @property
    def IsDown(self) -> bool:
        """Whether the bar closed lower than it opened."""
        if self.IsDoji:
            return False
        return self.Close < self.Open
    
    @property
    def IsDoji(self) -> bool:
        """Whether the bar is a doji (open ≈ close)."""
        return abs(self.Close - self.Open) < 0.01
    
    def __str__(self) -> str:
        """String representation."""
        direction = "▲" if self.IsUp else "▼" if self.IsDown else "─"
        return f"{self.Symbol.Value} {direction} O:{self.Open:.2f} H:{self.High:.2f} L:{self.Low:.2f} C:{self.Close:.2f} V:{self.Volume}"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"TradeBar(Symbol={self.Symbol}, Time={self.Time}, "
                f"OHLC=({self.Open:.2f}, {self.High:.2f}, {self.Low:.2f}, {self.Close:.2f}), "
                f"Volume={self.Volume})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'Symbol': self.Symbol.to_dict(),
            'Time': self.Time.isoformat(),
            'Open': self.Open,
            'High': self.High,
            'Low': self.Low,
            'Close': self.Close,
            'Volume': self.Volume,
            'Period': self.Period.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeBar':
        """Create TradeBar from dictionary."""
        return cls(
            Symbol=Symbol.from_dict(data['Symbol']),
            Time=datetime.fromisoformat(data['Time']),
            Open=data['Open'],
            High=data['High'],
            Low=data['Low'],
            Close=data['Close'],
            Volume=data['Volume'],
            Period=Resolution(data['Period'])
        )


@dataclass
class QuoteBar(JSONSerializable):
    """
    Represents bid/ask quote data for a security.
    
    This class mirrors QuantConnect's QuoteBar for compatibility.
    """
    Symbol: Symbol
    Time: datetime
    Bid: Optional[TradeBar] = None
    Ask: Optional[TradeBar] = None
    Period: Resolution = Resolution.MINUTE
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.Bid and self.Ask:
            if self.Bid.Close > self.Ask.Close:
                logger.warning(f"Bid price ({self.Bid.Close}) > Ask price ({self.Ask.Close}) for {self.Symbol.Value}")
    
    @property
    def Price(self) -> float:
        """Mid price (average of bid and ask)."""
        if self.Bid and self.Ask:
            return (self.Bid.Close + self.Ask.Close) / 2
        elif self.Bid:
            return self.Bid.Close
        elif self.Ask:
            return self.Ask.Close
        return 0.0
    
    @property
    def Spread(self) -> float:
        """Bid-ask spread."""
        if self.Bid and self.Ask:
            return self.Ask.Close - self.Bid.Close
        return 0.0
    
    @property
    def SpreadPercent(self) -> float:
        """Bid-ask spread as percentage of mid price."""
        if self.Price > 0:
            return (self.Spread / self.Price) * 100
        return 0.0
    
    def __str__(self) -> str:
        """String representation."""
        if self.Bid and self.Ask:
            return f"{self.Symbol.Value} Bid:{self.Bid.Close:.2f} Ask:{self.Ask.Close:.2f} Spread:{self.Spread:.2f}"
        elif self.Bid:
            return f"{self.Symbol.Value} Bid:{self.Bid.Close:.2f}"
        elif self.Ask:
            return f"{self.Symbol.Value} Ask:{self.Ask.Close:.2f}"
        return f"{self.Symbol.Value}: No quote data"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"QuoteBar(Symbol={self.Symbol}, Time={self.Time}, "
                f"Bid={self.Bid}, Ask={self.Ask})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'Symbol': self.Symbol.to_dict(),
            'Time': self.Time.isoformat(),
            'Bid': self.Bid.to_dict() if self.Bid else None,
            'Ask': self.Ask.to_dict() if self.Ask else None,
            'Period': self.Period.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuoteBar':
        """Create QuoteBar from dictionary."""
        return cls(
            Symbol=Symbol.from_dict(data['Symbol']),
            Time=datetime.fromisoformat(data['Time']),
            Bid=TradeBar.from_dict(data['Bid']) if data['Bid'] else None,
            Ask=TradeBar.from_dict(data['Ask']) if data['Ask'] else None,
            Period=Resolution(data['Period'])
        )


@dataclass
class Portfolio(JSONSerializable):
    """
    Represents a portfolio with cash and security holdings.
    
    This class tracks total portfolio value, cash, and individual security positions.
    """
    Cash: float = 100000.0
    TotalFees: float = 0.0
    TotalProfit: float = 0.0
    Holdings: Dict[str, SecurityHolding] = field(default_factory=dict)
    LastUpdate: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize after creation."""
        if self.LastUpdate is None:
            self.LastUpdate = datetime.now()
    
    @property
    def TotalPortfolioValue(self) -> float:
        """Total portfolio value (cash + holdings)."""
        holdings_value = sum(
            holding.Quantity * holding.LastTradePrice 
            for holding in self.Holdings.values()
        )
        return self.Cash + holdings_value
    
    @property
    def TotalUnrealizedProfit(self) -> float:
        """Total unrealized profit/loss."""
        return sum(holding.UnrealizedProfit for holding in self.Holdings.values())
    
    @property
    def TotalHoldingsValue(self) -> float:
        """Total value of all holdings."""
        return sum(
            holding.Quantity * holding.LastTradePrice 
            for holding in self.Holdings.values()
        )
    
    @property
    def CashPercent(self) -> float:
        """Cash as percentage of total portfolio value."""
        if self.TotalPortfolioValue > 0:
            return (self.Cash / self.TotalPortfolioValue) * 100
        return 0.0
    
    def AddSecurity(self, symbol: Symbol) -> SecurityHolding:
        """Add a new security to the portfolio."""
        if symbol.Value not in self.Holdings:
            self.Holdings[symbol.Value] = SecurityHolding(symbol)
        return self.Holdings[symbol.Value]
    
    def GetHolding(self, symbol: Union[str, Symbol]) -> Optional[SecurityHolding]:
        """Get holdings for a specific symbol."""
        symbol_value = symbol.Value if isinstance(symbol, Symbol) else symbol
        return self.Holdings.get(symbol_value)
    
    def UpdateCash(self, amount: float):
        """Update cash balance."""
        self.Cash += amount
        self.LastUpdate = datetime.now()
        logger.info(f"Cash updated: ${amount:+.2f}, New balance: ${self.Cash:.2f}")
    
    def AddFees(self, amount: float):
        """Add trading fees."""
        self.TotalFees += amount
        self.Cash -= amount
        self.LastUpdate = datetime.now()
        logger.info(f"Fees added: ${amount:.2f}, Total fees: ${self.TotalFees:.2f}")
    
    def __str__(self) -> str:
        """String representation."""
        return f"Portfolio: Cash=${self.Cash:,.2f}, Value=${self.TotalPortfolioValue:,.2f}, Holdings={len(self.Holdings)}"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"Portfolio(Cash=${self.Cash:.2f}, TotalValue=${self.TotalPortfolioValue:.2f}, "
                f"Holdings={len(self.Holdings)}, TotalFees=${self.TotalFees:.2f})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'Cash': self.Cash,
            'TotalFees': self.TotalFees,
            'TotalProfit': self.TotalProfit,
            'Holdings': {k: v.to_dict() for k, v in self.Holdings.items()},
            'LastUpdate': self.LastUpdate.isoformat() if self.LastUpdate else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Create Portfolio from dictionary."""
        portfolio = cls(
            Cash=data['Cash'],
            TotalFees=data['TotalFees'],
            TotalProfit=data['TotalProfit'],
            LastUpdate=datetime.fromisoformat(data['LastUpdate']) if data['LastUpdate'] else None
        )
        portfolio.Holdings = {
            k: SecurityHolding.from_dict(v) for k, v in data['Holdings'].items()
        }
        return portfolio



# Convenience functions
def create_symbol(ticker: str, security_type: SecurityType = SecurityType.EQUITY, 
                 market: Market = Market.US_EQUITY) -> Symbol:
    """Create a Symbol object with default values."""
    return Symbol(Value=ticker, SecurityType=security_type, Market=market)


def create_trade_bar(symbol: Symbol, time: datetime, open_price: float, high: float, 
                    low: float, close: float, volume: int, 
                    period: Resolution = Resolution.MINUTE) -> TradeBar:
    """Create a TradeBar object."""
    return TradeBar(
        Symbol=symbol, Time=time, Open=open_price, High=high, 
        Low=low, Close=close, Volume=volume, Period=period
    )


def create_quote_bar(symbol: Symbol, time: datetime, bid: Optional[TradeBar] = None,
                    ask: Optional[TradeBar] = None, 
                    period: Resolution = Resolution.MINUTE) -> QuoteBar:
    """Create a QuoteBar object."""
    return QuoteBar(Symbol=symbol, Time=time, Bid=bid, Ask=ask, Period=period)


def create_portfolio(initial_cash: float = 100000.0) -> Portfolio:
    """Create a Portfolio object with initial cash."""
    return Portfolio(Cash=initial_cash) 