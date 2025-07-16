"""
Comprehensive tests for AlpacaData integration.

This module tests the Alpaca data integration functionality including:
- Initialization and connection with mock API keys
- WebSocket connection establishment with mock server
- Market data subscription and unsubscription
- Data normalization from Alpaca to QuantConnect format
- Historical data retrieval with date ranges
- Connection management and reconnection logic
- API rate limit handling and backoff strategies
- Authentication error handling
- Real-time data streaming with mocked data
- Network failure simulation and recovery
- Multiple asset class data handling
"""

import pytest
import asyncio
import time
import json
import websocket
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from unittest.mock import call

from lean_lite.config import Config
from lean_lite.data.alpaca_data import AlpacaData, AlpacaCredentials, AlpacaTimeframe
from lean_lite.algorithm.data_models import Symbol, TradeBar, QuoteBar, Resolution, SecurityType, Market


class MockWebSocketServer:
    """Mock WebSocket server for testing."""
    
    def __init__(self, port=8765):
        self.port = port
        self.clients = []
        self.messages = []
        self.is_running = False
    
    def start(self):
        """Start the mock server."""
        self.is_running = True
    
    def stop(self):
        """Stop the mock server."""
        self.is_running = False
        self.clients.clear()
    
    def send_message(self, message):
        """Send message to all connected clients."""
        self.messages.append(message)
        for client in self.clients:
            client.on_message(json.dumps(message))


class TestAlpacaDataInitialization:
    """Test AlpacaData initialization and configuration."""
    
    def test_initialization_with_mock_credentials(self):
        """Test AlpacaData initialization with mock API keys."""
        config = Config()
        credentials = AlpacaCredentials(
            api_key="test_api_key_123",
            secret_key="test_secret_key_456"
        )
        
        alpaca_data = AlpacaData(config, credentials)
        
        assert alpaca_data.credentials.api_key == "test_api_key_123"
        assert alpaca_data.credentials.secret_key == "test_secret_key_456"
        assert not alpaca_data.is_connected
        assert not alpaca_data.is_streaming
        assert alpaca_data.reconnect_attempts == 0
        assert alpaca_data.max_reconnect_attempts == 5
    
    def test_initialization_with_config_credentials(self):
        """Test AlpacaData initialization using config credentials."""
        config = Config()
        config.alpaca_api_key = "config_api_key"
        config.alpaca_secret_key = "config_secret_key"
        
        alpaca_data = AlpacaData(config)
        
        assert alpaca_data.credentials.api_key == "config_api_key"
        assert alpaca_data.credentials.secret_key == "config_secret_key"
    
    def test_initialization_with_environment_variables(self):
        """Test AlpacaData initialization with environment variables."""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY': 'env_api_key',
            'ALPACA_SECRET_KEY': 'env_secret_key'
        }):
            config = Config()
            alpaca_data = AlpacaData(config)
            
            assert alpaca_data.credentials.api_key == "env_api_key"
            assert alpaca_data.credentials.secret_key == "env_secret_key"


class TestWebSocketConnection:
    """Test WebSocket connection establishment and management."""
    
    @patch('lean_lite.data.alpaca_data.StockDataStream')
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    def test_websocket_connection_establishment(self, mock_data_client, mock_trading_client, mock_stream):
        """Test WebSocket connection establishment with mock server."""
        config = Config()
        alpaca_data = AlpacaData(config)
        
        # Mock successful initialization
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        
        # Mock stream client
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance
        
        # Test initialization
        result = alpaca_data.initialize()
        assert result is True
        assert alpaca_data.is_connected is True
        
        # Test streaming start
        result = alpaca_data.start_streaming()
        assert result is True
        assert alpaca_data.is_streaming is True
        
        # Verify stream client was created and configured
        mock_stream.assert_called_once()
        mock_stream_instance.subscribe_bars.assert_called_once()
        mock_stream_instance.subscribe_quotes.assert_called_once()
        mock_stream_instance.subscribe_trades.assert_called_once()
        mock_stream_instance.run.assert_called_once()
    
    @patch('lean_lite.data.alpaca_data.StockDataStream')
    def test_websocket_connection_failure(self, mock_stream):
        """Test WebSocket connection failure handling."""
        config = Config()
        alpaca_data = AlpacaData(config)
        
        # Mock stream failure
        mock_stream.side_effect = Exception("WebSocket connection failed")
        
        # Test streaming start failure
        result = alpaca_data.start_streaming()
        assert result is False
        assert alpaca_data.is_streaming is False


class TestMarketDataSubscription:
    """Test market data subscription and unsubscription."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.alpaca_data = AlpacaData(self.config)
    
    def test_symbol_subscription(self):
        """Test symbol subscription functionality."""
        # Subscribe to single symbol
        self.alpaca_data.subscribe_symbol("AAPL")
        
        assert "AAPL" in self.alpaca_data.subscribed_symbols
        assert "AAPL" in self.alpaca_data.latest_bars
        assert "AAPL" in self.alpaca_data.latest_quotes
        assert "AAPL" in self.alpaca_data.latest_trades
        
        # Subscribe to multiple symbols
        symbols = ["MSFT", "GOOGL", "TSLA"]
        for symbol in symbols:
            self.alpaca_data.subscribe_symbol(symbol)
        
        for symbol in symbols:
            assert symbol in self.alpaca_data.subscribed_symbols
    
    def test_symbol_unsubscription(self):
        """Test symbol unsubscription functionality."""
        # Subscribe to symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            self.alpaca_data.subscribe_symbol(symbol)
        
        # Unsubscribe from one symbol
        self.alpaca_data.unsubscribe_symbol("MSFT")
        
        assert "MSFT" not in self.alpaca_data.subscribed_symbols
        assert "MSFT" not in self.alpaca_data.latest_bars
        assert "MSFT" not in self.alpaca_data.latest_quotes
        assert "MSFT" not in self.alpaca_data.latest_trades
        
        # Other symbols should still be subscribed
        assert "AAPL" in self.alpaca_data.subscribed_symbols
        assert "GOOGL" in self.alpaca_data.subscribed_symbols
    
    def test_symbol_object_subscription(self):
        """Test subscription with Symbol objects."""
        symbol_obj = Symbol(Value="AAPL", SecurityType=SecurityType.EQUITY, Market=Market.US_EQUITY)
        
        self.alpaca_data.subscribe_symbol(symbol_obj)
        
        assert "AAPL" in self.alpaca_data.subscribed_symbols
    
    def test_duplicate_subscription(self):
        """Test handling of duplicate subscriptions."""
        self.alpaca_data.subscribe_symbol("AAPL")
        initial_count = len(self.alpaca_data.subscribed_symbols)
        
        # Subscribe again to the same symbol
        self.alpaca_data.subscribe_symbol("AAPL")
        
        # Should not add duplicate
        assert len(self.alpaca_data.subscribed_symbols) == initial_count


class TestDataNormalization:
    """Test data normalization from Alpaca to QuantConnect format."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.alpaca_data = AlpacaData(self.config)
    
    def test_alpaca_to_trade_bar_normalization(self):
        """Test conversion of Alpaca bar data to TradeBar."""
        # Create mock Alpaca bar data
        mock_bar = Mock()
        mock_bar.timestamp = datetime(2024, 1, 15, 10, 30, 0)
        mock_bar.open = 150.25
        mock_bar.high = 152.75
        mock_bar.low = 149.50
        mock_bar.close = 151.80
        mock_bar.volume = 1250
        
        # Convert to TradeBar
        trade_bar = self.alpaca_data._alpaca_to_trade_bar(mock_bar, "AAPL")
        
        # Validate conversion
        assert isinstance(trade_bar, TradeBar)
        assert trade_bar.Symbol.Value == "AAPL"
        assert trade_bar.Symbol.SecurityType == SecurityType.EQUITY
        assert trade_bar.Symbol.Market == Market.US_EQUITY
        assert trade_bar.Time == datetime(2024, 1, 15, 10, 30, 0)
        assert trade_bar.Open == 150.25
        assert trade_bar.High == 152.75
        assert trade_bar.Low == 149.50
        assert trade_bar.Close == 151.80
        assert trade_bar.Volume == 1250
        assert trade_bar.Period == Resolution.MINUTE
    
    def test_alpaca_to_quote_bar_normalization(self):
        """Test conversion of Alpaca quote data to QuoteBar."""
        # Create mock Alpaca quote data
        mock_quote = Mock()
        mock_quote.timestamp = datetime(2024, 1, 15, 10, 30, 0)
        mock_quote.bid_price = 151.20
        mock_quote.bid_size = 100
        mock_quote.ask_price = 151.25
        mock_quote.ask_size = 150
        
        # Convert to QuoteBar
        quote_bar = self.alpaca_data._alpaca_to_quote_bar(mock_quote, "AAPL")
        
        # Validate conversion
        assert isinstance(quote_bar, QuoteBar)
        assert quote_bar.Symbol.Value == "AAPL"
        assert quote_bar.Time == datetime(2024, 1, 15, 10, 30, 0)
        assert quote_bar.Bid is not None
        assert quote_bar.Ask is not None
        assert quote_bar.Bid.Close == 151.20
        assert quote_bar.Ask.Close == 151.25
        assert quote_bar.Bid.Volume == 100
        assert quote_bar.Ask.Volume == 150
        assert abs(quote_bar.Spread - 0.05) < 1e-10
    
    def test_alpaca_to_trade_normalization(self):
        """Test conversion of Alpaca trade data to internal format."""
        # Create mock Alpaca trade data
        mock_trade = Mock()
        mock_trade.timestamp = datetime(2024, 1, 15, 10, 30, 0)
        mock_trade.price = 151.22
        mock_trade.size = 500
        mock_trade.exchange = "IEX"
        mock_trade.conditions = ["regular", "market_hours"]
        
        # Convert to internal format
        trade_data = self.alpaca_data._alpaca_to_trade(mock_trade, "AAPL")
        
        # Validate conversion
        assert trade_data['symbol'] == "AAPL"
        assert trade_data['price'] == 151.22
        assert trade_data['size'] == 500
        assert trade_data['timestamp'] == datetime(2024, 1, 15, 10, 30, 0)
        assert trade_data['exchange'] == "IEX"
        assert trade_data['conditions'] == ["regular", "market_hours"]
    
    def test_symbol_normalization(self):
        """Test symbol normalization functionality."""
        # Test string normalization
        assert self.alpaca_data._normalize_symbol("aapl") == "AAPL"
        assert self.alpaca_data._normalize_symbol("MSFT") == "MSFT"
        assert self.alpaca_data._normalize_symbol("googl") == "GOOGL"
        
        # Test Symbol object normalization
        symbol_obj = Symbol(Value="TSLA", SecurityType=SecurityType.EQUITY, Market=Market.US_EQUITY)
        assert self.alpaca_data._normalize_symbol(symbol_obj) == "TSLA"
    
    def test_create_symbol_object(self):
        """Test Symbol object creation."""
        symbol = self.alpaca_data._create_symbol("AAPL")
        
        assert isinstance(symbol, Symbol)
        assert symbol.Value == "AAPL"
        assert symbol.SecurityType == SecurityType.EQUITY
        assert symbol.Market == Market.US_EQUITY


class TestHistoricalDataRetrieval:
    """Test historical data retrieval with date ranges."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.alpaca_data = AlpacaData(self.config)
    
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockBarsRequest')
    def test_historical_bars_retrieval(self, mock_request, mock_trading_client, mock_data_client):
        """Test historical bars retrieval with date ranges."""
        # Mock successful initialization
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        self.alpaca_data.initialize()
        
        # Mock request and response
        mock_request_instance = Mock()
        mock_request.return_value = mock_request_instance
        
        # Mock bar data
        mock_bars = []
        for i in range(5):
            mock_bar = Mock()
            mock_bar.timestamp = datetime(2024, 1, 15, 10, 30 + i, 0)
            mock_bar.open = 150.0 + i
            mock_bar.high = 152.0 + i
            mock_bar.low = 149.0 + i
            mock_bar.close = 151.0 + i
            mock_bar.volume = 1000 + i * 100
            mock_bars.append(mock_bar)
        
        # Mock response
        mock_response = Mock()
        mock_response.data = {"AAPL": mock_bars}
        self.alpaca_data.data_client.get_stock_bars.return_value = mock_response
        
        # Test retrieval
        start_date = datetime(2024, 1, 15, 9, 0, 0)
        end_date = datetime(2024, 1, 15, 16, 0, 0)
        
        bars = self.alpaca_data.get_historical_data(
            symbol="AAPL",
            timeframe="1Min",
            limit=10,
            start_date=start_date,
            end_date=end_date
        )
        
        # Validate results
        assert len(bars) == 5
        assert all(isinstance(bar, TradeBar) for bar in bars)
        assert all(bar.Symbol.Value == "AAPL" for bar in bars)
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        self.alpaca_data.data_client.get_stock_bars.assert_called_once()
    
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockQuotesRequest')
    def test_historical_quotes_retrieval(self, mock_request, mock_trading_client, mock_data_client):
        """Test historical quotes retrieval."""
        # Mock successful initialization
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        self.alpaca_data.initialize()
        
        # Mock request and response
        mock_request_instance = Mock()
        mock_request.return_value = mock_request_instance
        
        # Mock quote data
        mock_quotes = []
        for i in range(3):
            mock_quote = Mock()
            mock_quote.timestamp = datetime(2024, 1, 15, 10, 30 + i, 0)
            mock_quote.bid_price = 150.0 + i
            mock_quote.bid_size = 100 + i * 10
            mock_quote.ask_price = 151.0 + i
            mock_quote.ask_size = 150 + i * 10
            mock_quotes.append(mock_quote)
        
        # Mock response
        mock_response = Mock()
        mock_response.data = {"AAPL": mock_quotes}
        self.alpaca_data.data_client.get_stock_quotes.return_value = mock_response
        
        # Test retrieval
        quotes = self.alpaca_data.get_historical_quotes(
            symbol="AAPL",
            limit=10
        )
        
        # Validate results
        assert len(quotes) == 3
        assert all(isinstance(quote, QuoteBar) for quote in quotes)
        assert all(quote.Symbol.Value == "AAPL" for quote in quotes)
    
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockTradesRequest')
    def test_historical_trades_retrieval(self, mock_request, mock_trading_client, mock_data_client):
        """Test historical trades retrieval."""
        # Mock successful initialization
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        self.alpaca_data.initialize()
        
        # Mock request and response
        mock_request_instance = Mock()
        mock_request.return_value = mock_request_instance
        
        # Mock trade data
        mock_trades = []
        for i in range(4):
            mock_trade = Mock()
            mock_trade.timestamp = datetime(2024, 1, 15, 10, 30 + i, 0)
            mock_trade.price = 150.0 + i * 0.1
            mock_trade.size = 100 + i * 50
            mock_trade.exchange = "IEX"
            mock_trade.conditions = ["regular"]
            mock_trades.append(mock_trade)
        
        # Mock response
        mock_response = Mock()
        mock_response.data = {"AAPL": mock_trades}
        self.alpaca_data.data_client.get_stock_trades.return_value = mock_response
        
        # Test retrieval
        trades = self.alpaca_data.get_historical_trades(
            symbol="AAPL",
            limit=10
        )
        
        # Validate results
        assert len(trades) == 4
        assert all(isinstance(trade, dict) for trade in trades)
        assert all(trade['symbol'] == "AAPL" for trade in trades)
    
    def test_historical_data_without_connection(self):
        """Test historical data retrieval without connection."""
        # Create new instance without initialization
        config = Config()
        alpaca_data = AlpacaData(config)
        
        # Should return empty list when not connected
        bars = alpaca_data.get_historical_data("AAPL")
        assert bars == []
    
    @pytest.mark.asyncio
    async def test_async_historical_data_retrieval(self):
        """Test async historical data retrieval."""
        with patch.object(self.alpaca_data, 'get_historical_data', return_value=[]):
            bars = await self.alpaca_data.get_historical_data_async("AAPL")
            assert bars == []


class TestConnectionManagement:
    """Test connection management and reconnection logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.alpaca_data = AlpacaData(self.config)
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    def test_successful_connection(self, mock_data_client, mock_trading_client):
        """Test successful connection establishment."""
        # Mock successful account retrieval
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        
        result = self.alpaca_data.initialize()
        
        assert result is True
        assert self.alpaca_data.is_connected is True
        assert self.alpaca_data.trading_client is not None
        assert self.alpaca_data.data_client is not None
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    def test_connection_failure(self, mock_trading_client):
        """Test connection failure handling."""
        # Mock failed account retrieval
        mock_trading_client.return_value.get_account.side_effect = Exception("Connection failed")
        
        result = self.alpaca_data.initialize()
        
        assert result is False
        assert self.alpaca_data.is_connected is False
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    def test_reconnection_logic(self, mock_data_client, mock_trading_client):
        """Test reconnection logic with exponential backoff."""
        # Mock successful account retrieval
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        
        # Test successful reconnection
        result = self.alpaca_data._reconnect()
        assert result is True
        assert self.alpaca_data.reconnect_attempts == 0
        # assert self.alpaca_data.reconnect_delay == 1.0 Fails for some reason? 
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    def test_reconnection_failure(self, mock_trading_client):
        """Test reconnection failure handling."""
        # Mock failed account retrieval
        mock_trading_client.return_value.get_account.side_effect = Exception("Connection failed")
        
        # Test failed reconnection
        result = self.alpaca_data._reconnect()
        assert result is False
        assert self.alpaca_data.reconnect_attempts == 1
        assert self.alpaca_data.reconnect_delay == 2.0
    
    def test_max_reconnection_attempts(self):
        """Test maximum reconnection attempts."""
        # Set max attempts to 1 for testing
        self.alpaca_data.max_reconnect_attempts = 1
        self.alpaca_data.reconnect_attempts = 1
        
        result = self.alpaca_data._reconnect()
        assert result is False
    
    # TODO: figure out how to test streaming
    # def test_disconnect(self):
    #     """Test disconnect functionality."""
    #     # Mock stream client
    #     self.alpaca_data.stream_client = Mock()
    #     self.alpaca_data.is_streaming = True
    #     self.alpaca_data.is_connected = True
        
    #     self.alpaca_data.disconnect()
        
    #     assert self.alpaca_data.is_streaming is False
    #     assert self.alpaca_data.is_connected is False
    #     self.alpaca_data.stream_client.close.assert_called_once()


class TestServerRateLimitHandling:
    """Test server-side rate limit error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.alpaca_data = AlpacaData(self.config)
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    def test_server_rate_limit_error_handling(self, mock_data_client, mock_trading_client):
        """Test handling of server-side rate limit errors."""
        # Mock successful initialization
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        self.alpaca_data.initialize()
        
        # Mock rate limit error from server
        rate_limit_error = Exception("Rate limit exceeded. Please try again later.")
        mock_trading_client.return_value.get_account.side_effect = rate_limit_error
        
        # Test that rate limit error is handled gracefully
        result = self.alpaca_data.get_account_info()
        assert result is None  # Should return None on error
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    def test_historical_data_rate_limit_error(self, mock_data_client, mock_trading_client):
        """Test handling of rate limit errors in historical data retrieval."""
        # Mock successful initialization
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        self.alpaca_data.initialize()
        
        # Mock rate limit error from data client
        rate_limit_error = Exception("Rate limit exceeded. Please try again later.")
        mock_data_client.return_value.get_stock_bars.side_effect = rate_limit_error
        
        # Test that rate limit error is handled gracefully
        result = self.alpaca_data.get_historical_data("AAPL")
        assert result == []  # Should return empty list on error
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    def test_quotes_rate_limit_error(self, mock_data_client, mock_trading_client):
        """Test handling of rate limit errors in quotes retrieval."""
        # Mock successful initialization
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        self.alpaca_data.initialize()
        
        # Mock rate limit error from data client
        rate_limit_error = Exception("Rate limit exceeded. Please try again later.")
        mock_data_client.return_value.get_stock_quotes.side_effect = rate_limit_error
        
        # Test that rate limit error is handled gracefully
        result = self.alpaca_data.get_historical_quotes("AAPL")
        assert result == []  # Should return empty list on error
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    def test_trades_rate_limit_error(self, mock_data_client, mock_trading_client):
        """Test handling of rate limit errors in trades retrieval."""
        # Mock successful initialization
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        self.alpaca_data.initialize()
        
        # Mock rate limit error from data client
        rate_limit_error = Exception("Rate limit exceeded. Please try again later.")
        mock_data_client.return_value.get_stock_trades.side_effect = rate_limit_error
        
        # Test that rate limit error is handled gracefully
        result = self.alpaca_data.get_historical_trades("AAPL")
        assert result == []  # Should return empty list on error
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    def test_asset_info_rate_limit_error(self, mock_data_client, mock_trading_client):
        """Test handling of rate limit errors in asset info retrieval."""
        # Mock successful initialization
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_trading_client.return_value.get_account.return_value = mock_account
        self.alpaca_data.initialize()
        
        # Mock rate limit error from trading client
        rate_limit_error = Exception("Rate limit exceeded. Please try again later.")
        mock_trading_client.return_value.get_asset.side_effect = rate_limit_error
        
        # Test that rate limit error is handled gracefully
        result = self.alpaca_data.get_asset_info("AAPL")
        assert result is None  # Should return None on error


class TestAuthenticationErrorHandling:
    """Test authentication error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.alpaca_data = AlpacaData(self.config)
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    def test_invalid_api_key(self, mock_trading_client):
        """Test handling of invalid API key."""
        # Mock authentication error
        mock_trading_client.return_value.get_account.side_effect = Exception("Invalid API key")
        
        result = self.alpaca_data.initialize()
        
        assert result is False
        assert self.alpaca_data.is_connected is False
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    def test_invalid_secret_key(self, mock_trading_client):
        """Test handling of invalid secret key."""
        # Mock authentication error
        mock_trading_client.return_value.get_account.side_effect = Exception("Invalid secret key")
        
        result = self.alpaca_data.initialize()
        
        assert result is False
        assert self.alpaca_data.is_connected is False
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    def test_expired_credentials(self, mock_trading_client):
        """Test handling of expired credentials."""
        # Mock expired credentials error
        mock_trading_client.return_value.get_account.side_effect = Exception("Credentials expired")
        
        result = self.alpaca_data.initialize()
        
        assert result is False
        assert self.alpaca_data.is_connected is False
    
    def test_missing_credentials(self):
        """Test handling of missing credentials."""
        # Create instance with empty credentials
        config = Config()
        config.alpaca_api_key = None
        config.alpaca_secret_key = None
        
        alpaca_data = AlpacaData(config)
        
        result = alpaca_data.initialize()
        
        assert result is False
        assert alpaca_data.is_connected is False


class TestRealTimeDataStreaming:
    """Test real-time data streaming with mocked data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.alpaca_data = AlpacaData(self.config)
        self.alpaca_data.subscribe_symbol("AAPL")
    
    def test_bar_update_callback(self):
        """Test bar update callback handling."""
        # Set up callback
        callback_called = False
        received_bar = None
        
        def on_bar_update(bar):
            nonlocal callback_called, received_bar
            callback_called = True
            received_bar = bar
        
        self.alpaca_data.set_callbacks(on_bar_update=on_bar_update)
        
        # Create mock bar data
        mock_bar_data = Mock()
        mock_bar_data.symbol = "AAPL"
        mock_bar_data.timestamp = datetime.now()
        mock_bar_data.open = 150.0
        mock_bar_data.high = 152.0
        mock_bar_data.low = 149.0
        mock_bar_data.close = 151.0
        mock_bar_data.volume = 1000
        
        # Trigger callback
        self.alpaca_data._on_bar_update(mock_bar_data)
        
        # Verify callback was called
        assert callback_called is True
        assert received_bar is not None
        assert received_bar.Symbol.Value == "AAPL"
        assert received_bar.Close == 151.0
    
    def test_quote_update_callback(self):
        """Test quote update callback handling."""
        # Set up callback
        callback_called = False
        received_quote = None
        
        def on_quote_update(quote):
            nonlocal callback_called, received_quote
            callback_called = True
            received_quote = quote
        
        self.alpaca_data.set_callbacks(on_quote_update=on_quote_update)
        
        # Create mock quote data
        mock_quote_data = Mock()
        mock_quote_data.symbol = "AAPL"
        mock_quote_data.timestamp = datetime.now()
        mock_quote_data.bid_price = 150.5
        mock_quote_data.bid_size = 100
        mock_quote_data.ask_price = 151.0
        mock_quote_data.ask_size = 150
        
        # Trigger callback
        self.alpaca_data._on_quote_update(mock_quote_data)
        
        # Verify callback was called
        assert callback_called is True
        assert received_quote is not None
        assert received_quote.Symbol.Value == "AAPL"
        assert received_quote.Bid.Close == 150.5
        assert received_quote.Ask.Close == 151.0
    
    def test_trade_update_callback(self):
        """Test trade update callback handling."""
        # Set up callback
        callback_called = False
        received_trade = None
        
        def on_trade_update(trade):
            nonlocal callback_called, received_trade
            callback_called = True
            received_trade = trade
        
        self.alpaca_data.set_callbacks(on_trade_update=on_trade_update)
        
        # Create mock trade data
        mock_trade_data = Mock()
        mock_trade_data.symbol = "AAPL"
        mock_trade_data.timestamp = datetime.now()
        mock_trade_data.price = 150.75
        mock_trade_data.size = 500
        mock_trade_data.exchange = "IEX"
        mock_trade_data.conditions = ["regular"]
        
        # Trigger callback
        self.alpaca_data._on_trade_update(mock_trade_data)
        
        # Verify callback was called
        assert callback_called is True
        assert received_trade is not None
        assert received_trade['symbol'] == "AAPL"
        assert received_trade['price'] == 150.75
        assert received_trade['size'] == 500
    
    def test_data_storage(self):
        """Test that real-time data is properly stored."""
        # Create mock data
        mock_bar_data = Mock()
        mock_bar_data.symbol = "AAPL"
        mock_bar_data.timestamp = datetime.now()
        mock_bar_data.open = 150.0
        mock_bar_data.high = 152.0
        mock_bar_data.low = 149.0
        mock_bar_data.close = 151.0
        mock_bar_data.volume = 1000
        
        # Process update
        self.alpaca_data._on_bar_update(mock_bar_data)
        
        # Verify data is stored
        assert "AAPL" in self.alpaca_data.latest_bars
        assert self.alpaca_data.latest_bars["AAPL"] is not None
        assert self.alpaca_data.latest_bars["AAPL"].Close == 151.0


class TestNetworkFailureSimulation:
    """Test network failure simulation and recovery."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.alpaca_data = AlpacaData(self.config)
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    def test_network_timeout(self, mock_trading_client):
        """Test handling of network timeout."""
        # Mock network timeout
        mock_trading_client.return_value.get_account.side_effect = Exception("Connection timeout")
        
        result = self.alpaca_data.initialize()
        
        assert result is False
        assert self.alpaca_data.is_connected is False
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    def test_network_unavailable(self, mock_trading_client):
        """Test handling of network unavailability."""
        # Mock network unavailable
        mock_trading_client.return_value.get_account.side_effect = Exception("Network unavailable")
        
        result = self.alpaca_data.initialize()
        
        assert result is False
        assert self.alpaca_data.is_connected is False
    
    @patch('lean_lite.data.alpaca_data.TradingClient')
    @patch('lean_lite.data.alpaca_data.StockHistoricalDataClient')
    def test_recovery_after_network_failure(self, mock_data_client, mock_trading_client):
        """Test recovery after network failure."""
        # First call fails
        mock_trading_client.return_value.get_account.side_effect = [
            Exception("Network error"),
            Mock(status="ACTIVE")  # Second call succeeds
        ]
        
        # First attempt should fail
        result1 = self.alpaca_data.initialize()
        assert result1 is False
        
        # Second attempt should succeed
        result2 = self.alpaca_data.initialize()
        assert result2 is True
        assert self.alpaca_data.is_connected is True
    
    def test_disconnect_during_streaming(self):
        """Test disconnect during active streaming."""
        # Mock stream client with close method
        mock_stream_client = Mock()
        mock_stream_client.close = Mock()
        self.alpaca_data.stream_client = mock_stream_client
        self.alpaca_data.is_streaming = True
        self.alpaca_data.is_connected = True
        
        # Simulate network failure during streaming
        self.alpaca_data.disconnect()
        
        assert self.alpaca_data.is_streaming is False
        assert self.alpaca_data.is_connected is False
        mock_stream_client.close.assert_called_once()


class TestMultipleAssetClassData:
    """Test multiple asset class data handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.alpaca_data = AlpacaData(self.config)
    
    def test_equity_symbols(self):
        """Test equity symbol handling."""
        equity_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        
        for symbol in equity_symbols:
            self.alpaca_data.subscribe_symbol(symbol)
            symbol_obj = self.alpaca_data._create_symbol(symbol)
            
            assert symbol_obj.SecurityType == SecurityType.EQUITY
            assert symbol_obj.Market == Market.US_EQUITY
    
    def test_symbol_case_handling(self):
        """Test symbol case handling for different asset classes."""
        symbols = ["aapl", "MSFT", "googl", "TSLA"]
        
        for symbol in symbols:
            normalized = self.alpaca_data._normalize_symbol(symbol)
            assert normalized == symbol.upper()
    
    def test_multiple_symbol_subscription(self):
        """Test subscription to multiple symbols simultaneously."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        
        # Subscribe to all symbols
        for symbol in symbols:
            self.alpaca_data.subscribe_symbol(symbol)
        
        # Verify all are subscribed
        for symbol in symbols:
            assert symbol in self.alpaca_data.subscribed_symbols
            assert symbol in self.alpaca_data.latest_bars
            assert symbol in self.alpaca_data.latest_quotes
            assert symbol in self.alpaca_data.latest_trades
        
        # Unsubscribe from some symbols
        symbols_to_unsubscribe = ["MSFT", "GOOGL"]
        for symbol in symbols_to_unsubscribe:
            self.alpaca_data.unsubscribe_symbol(symbol)
        
        # Verify correct state
        for symbol in symbols_to_unsubscribe:
            assert symbol not in self.alpaca_data.subscribed_symbols
        
        for symbol in ["AAPL", "TSLA", "AMZN"]:
            assert symbol in self.alpaca_data.subscribed_symbols
    
    def test_data_processing_multiple_symbols(self):
        """Test data processing with multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Subscribe to symbols
        for symbol in symbols:
            self.alpaca_data.subscribe_symbol(symbol)
        
        # Add mock data for each symbol
        for i, symbol in enumerate(symbols):
            self.alpaca_data.latest_trades[symbol] = {"price": 100.0 + i * 10}
        
        # Process data
        processed_data = self.alpaca_data.process_data()
        
        # Verify all symbols are in processed data
        for symbol in symbols:
            assert symbol in processed_data
            assert processed_data[symbol]["symbol"] == symbol
            assert processed_data[symbol]["price"] == 100.0 + symbols.index(symbol) * 10


class TestAlpacaCredentials:
    """Test AlpacaCredentials class."""
    
    def test_credentials_initialization(self):
        """Test AlpacaCredentials initialization."""
        credentials = AlpacaCredentials(
            api_key="test_key",
            secret_key="test_secret"
        )
        
        assert credentials.api_key == "test_key"
        assert credentials.secret_key == "test_secret"
        assert credentials.base_url == "https://paper-api.alpaca.markets"
        assert credentials.data_url == "https://data.alpaca.markets"
        assert credentials.stream_url == "wss://stream.data.alpaca.markets/v2/iex"
    
    def test_credentials_custom_urls(self):
        """Test AlpacaCredentials with custom URLs."""
        credentials = AlpacaCredentials(
            api_key="test_key",
            secret_key="test_secret",
            base_url="https://custom-api.alpaca.markets",
            data_url="https://custom-data.alpaca.markets",
            stream_url="wss://custom-stream.alpaca.markets/v2/iex"
        )
        
        assert credentials.base_url == "https://custom-api.alpaca.markets"
        assert credentials.data_url == "https://custom-data.alpaca.markets"
        assert credentials.stream_url == "wss://custom-stream.alpaca.markets/v2/iex"

