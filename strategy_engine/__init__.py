"""
Strategy Engine Module

A comprehensive framework for developing, backtesting, and executing trading strategies.
Provides standardized interfaces for signal generation, risk management, and performance tracking.
"""

from .base_strategy import BaseStrategy
from .signal import SignalDirection, TradingSignal
from .portfolio import PortfolioManager
from .risk import RiskEngine
from .performance import PerformanceTracker
from .exceptions import StrategyError, RiskViolationError
from .utils import (
    resample_data,
    calculate_pnl,
    optimize_position_size,
    validate_strategy_config
)

__all__ = [
    'BaseStrategy',
    'SignalDirection',
    'TradingSignal',
    'PortfolioManager',
    'RiskEngine',
    'PerformanceTracker',
    'StrategyError',
    'RiskViolationError',
    'resample_data',
    'calculate_pnl',
    'optimize_position_size',
    'validate_strategy_config'
]

# Package version
__version__ = '1.3.0'

# Initialize package logging
import logging
from logging.config import dictConfig

_log_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
    },
    'loggers': {
        'strategy_engine': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

dictConfig(_log_config)

# Package initialization
def initialize(settings: dict = None):
    """
    Initialize the strategy engine with optional configuration
    
    Args:
        settings: Dictionary of configuration settings including:
            - log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            - max_concurrent_strategies: Maximum strategies to run simultaneously
            - data_provider: Default data provider configuration
    """
    if settings:
        # Configure logging level
        if 'log_level' in settings:
            logger = logging.getLogger('strategy_engine')
            logger.setLevel(settings['log_level'].upper())
        
        # Initialize other package components
        if 'data_provider' in settings:
            from .data import DataManager
            DataManager.configure(settings['data_provider'])

    logging.info(f"Strategy Engine v{__version__} initialized")

# Import critical errors for easy access
class StrategyInitializationError(Exception):
    """Raised when strategy engine fails to initialize properly"""
    pass

class InvalidConfigurationError(Exception):
    """Raised when invalid configuration is provided"""
    pass
