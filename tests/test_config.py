"""
Tests for configuration module.
"""

import pytest
import os
from unittest.mock import patch

from lean_lite.config import Config


class TestConfig:
    """Test cases for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.broker_type == "alpaca"
        assert config.data_provider == "alpaca"
        assert config.log_level == "INFO"
        assert config.max_algorithm_capacity == 100
    
    @patch.dict(os.environ, {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_SECRET_KEY': 'test_secret',
        'BROKER_TYPE': 'alpaca',
        'LOG_LEVEL': 'DEBUG'
    })
    def test_environment_loading(self):
        """Test loading configuration from environment variables."""
        config = Config()
        
        assert config.alpaca_api_key == 'test_key'
        assert config.alpaca_secret_key == 'test_secret'
        assert config.broker_type == 'alpaca'
        assert config.log_level == 'DEBUG'
    
    def test_validation_with_valid_config(self):
        """Test configuration validation with valid settings."""
        config = Config()
        config.alpaca_api_key = "test_key"
        config.alpaca_secret_key = "test_secret"
        
        assert config.validate() is True
    
    def test_validation_without_credentials(self):
        """Test configuration validation without API credentials."""
        config = Config()
        config.alpaca_api_key = None
        config.alpaca_secret_key = None
        
        assert config.validate() is False 