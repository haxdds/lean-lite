# Lean-Lite

A lightweight, containerized QuantConnect LEAN runtime focused on live trading with Alpaca integration.

## Overview

Lean-Lite is a streamlined implementation of QuantConnect's algorithmic trading platform, designed for production deployment of trading strategies. It provides a familiar QuantConnect API while being optimized for containerized environments and live trading scenarios.

### Key Features

- **QuantConnect API Compatibility**: Write strategies using familiar QuantConnect Python syntax
- **Live Trading Focus**: Optimized for real-time trading with Alpaca broker integration
- **Containerized Deployment**: Each strategy runs in isolated Docker containers
- **Lightweight Runtime**: Minimal overhead for production environments
- **Portfolio Management**: Real-time position tracking and risk management
- **Order Management**: Comprehensive order lifecycle management with multiple order types

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   QCAlgorithm   │    │  Order Manager  │    │   Portfolio     │
│   (Strategy)    │◄──►│                 │◄──►│   Manager       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Lean-Lite Runtime                          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Docker (for containerized deployment)
- Alpaca account with API keys

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/lean-lite.git
cd lean-lite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Use paper trading
```

### Your First Strategy

Create a simple buy-and-hold strategy:

```python
# strategies/my_first_strategy.py
from src.algorithm.qc_algorithm import QCAlgorithm

class MyFirstStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)
        
        # Add assets
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        
    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1.0)
            self.Debug("Purchased SPY")
```

### Running Your Strategy

Run locally for testing:
```bash
python src/main.py --strategy strategies/my_first_strategy.py
```

## API Reference

### QCAlgorithm Base Class

The core class that all strategies inherit from.

#### Initialization Methods

- `SetStartDate(year, month, day)`: Set strategy start date
- `SetEndDate(year, month, day)`: Set strategy end date (optional)
- `SetCash(amount)`: Set initial capital
- `AddEquity(symbol, resolution)`: Add equity for trading
- `AddForex(symbol, resolution)`: Add forex pair for trading
- `AddCrypto(symbol, resolution)`: Add cryptocurrency for trading

#### Trading Methods

- `Buy(symbol, quantity)`: Place market buy order
- `Sell(symbol, quantity)`: Place market sell order
- `MarketOrder(symbol, quantity)`: Place market order
- `LimitOrder(symbol, quantity, price)`: Place limit order
- `StopOrder(symbol, quantity, price)`: Place stop order
- `Liquidate(symbol)`: Close all positions for symbol
- `SetHoldings(symbol, percentage)`: Set position to percentage of portfolio

#### Event Handlers

- `Initialize()`: Called once at strategy start
- `OnData(data)`: Called when new market data arrives
- `OnOrderEvent(orderEvent)`: Called when order status changes
- `OnEndOfDay()`: Called at market close

#### Properties

- `Portfolio`: Access to portfolio holdings and cash
- `Securities`: Dictionary of added securities
- `Time`: Current algorithm time

### Data Models

#### Symbol
Represents a financial instrument:
```python
symbol = Symbol("AAPL", SecurityType.Equity, Market.USA)
```

#### Security
Represents a tradeable asset:
```python
security = self.Securities["AAPL"]
price = security.Price
volume = security.Volume
```

#### TradeBar
OHLCV market data:
```python
def OnData(self, data):
    bar = data["AAPL"]
    open_price = bar.Open
    close_price = bar.Close
    volume = bar.Volume
```

#### Portfolio
Portfolio management:
```python
cash = self.Portfolio.Cash
total_value = self.Portfolio.TotalPortfolioValue
invested = self.Portfolio.Invested
```

### Order Management

#### Order Types

- `MarketOrder`: Execute immediately at market price
- `LimitOrder`: Execute only at specified price or better
- `StopOrder`: Trigger market order when price reached

#### Order Status

- `New`: Order created but not submitted
- `Submitted`: Order sent to broker
- `PartiallyFilled`: Order partially executed
- `Filled`: Order completely executed
- `Canceled`: Order canceled

Example order handling:
```python
def OnOrderEvent(self, orderEvent):
    if orderEvent.Status == OrderStatus.Filled:
        self.Debug(f"Order filled: {orderEvent.Symbol} {orderEvent.FillQuantity}")
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ALPACA_API_KEY` | Alpaca API key | Yes |
| `ALPACA_SECRET_KEY` | Alpaca secret key | Yes |
| `ALPACA_BASE_URL` | Alpaca API base URL | Yes |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No |

### Strategy Configuration

Strategies can be configured via initialization parameters:

```python
def Initialize(self):
    # Set trading parameters
    self.SetCash(100000)
    self.SetStartDate(2024, 1, 1)
    
    # Risk management
    self.max_position_size = 0.1  # 10% max per position
    self.stop_loss_percent = 0.05  # 5% stop loss
```

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_qc_algorithm.py
python -m pytest tests/test_orders.py
python -m pytest tests/test_portfolio.py

# Run with coverage
python -m pytest tests/ --cov=src/
```

## Development

### Project Structure

```
lean-lite/
├── src/
│   ├── algorithm/          # Core algorithm classes
│   │   ├── qc_algorithm.py
│   │   ├── data_models.py
│   │   ├── orders.py
│   │   └── portfolio.py
│   ├── data/              # Data pipeline (future)
│   ├── brokers/           # Broker integrations (future)
│   ├── engine/            # Runtime engine (future)
│   ├── indicators/        # Technical indicators (future)
│   └── main.py
├── strategies/            # User strategies
├── tests/                 # Test suite
├── docker/               # Docker configurations (future)
├── requirements.txt
└── README.md
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Roadmap

- [ ] Live data integration with Alpaca
- [ ] Technical indicators library
- [ ] Advanced order types
- [ ] Risk management framework
- [ ] Performance analytics
- [ ] Docker containerization
- [ ] Multi-broker support

## Support

- **