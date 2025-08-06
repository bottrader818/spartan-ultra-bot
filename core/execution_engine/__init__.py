
"""
Execution Engine Module

Provides professional order execution capabilities including:
- Smart order routing
- Algorithmic execution (TWAP/VWAP)
- Multi-broker support
- Risk-aware execution
- Performance monitoring
"""

from .base_executor import BaseExecutor
from .alpaca_executor import AlpacaExecutor
from .ib_executor import IBExecutor
from .composite_executor import CompositeExecutor
from .models import (
    Order,
    ExecutionResult,
    OrderStatus,
    OrderType,
    TimeInForce,
    ExecutionStyle
)
from .utils import (
    calculate_slippage,
    estimate_market_impact,
    get_optimal_execution_path
)

__all__ = [
    'BaseExecutor',
    'AlpacaExecutor',
    'IBExecutor',
    'CompositeExecutor',
    'Order',
    'ExecutionResult',
    'OrderStatus',
    'OrderType',
    'TimeInForce',
    'ExecutionStyle',
    'calculate_slippage',
    'estimate_market_impact',
    'get_optimal_execution_path'
]

# Initialize package-level logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Execution Engine package initialized")

# Package version
__version__ = '1.0.0'
