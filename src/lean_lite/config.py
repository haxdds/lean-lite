"""
Configuration management for Lean-Lite.

This module handles configuration loading, validation, and management
for the Lean-Lite runtime.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for Lean-Lite runtime."""
    
    # Broker configuration
    broker_type: str = "alpaca"
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    
    # Data configuration
    data_provider: str = "alpaca"
    data_frequency: str = "minute"
    
    # Engine configuration
    max_algorithm_capacity: int = 100
    log_level: str = "INFO"
    
    # Strategy configuration
    strategy_path: str = "strategies/"
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY", self.alpaca_api_key)
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY", self.alpaca_secret_key)
        self.broker_type = os.getenv("BROKER_TYPE", self.broker_type)
        self.data_provider = os.getenv("DATA_PROVIDER", self.data_provider)
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        
        logger.info(f"Configuration loaded: broker={self.broker_type}, data_provider={self.data_provider}")
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if self.broker_type == "alpaca":
            if not self.alpaca_api_key or not self.alpaca_secret_key:
                logger.error("Alpaca API credentials are required")
                return False
        
        return True 