from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from dataclasses import dataclass, field
import abc
import logging

logger = logging.getLogger(__name__)

# === ENUMS ===
class OrderStatus(Enum):
    PENDING = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()

class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TWAP = auto()
    VWAP = auto()
    ICEBERG = auto()

class TimeInForce(Enum):
    GTC = auto()  # Good Till Cancel
    IOC = auto()  # Immediate Or Cancel
    FOK = auto()  # Fill Or Kill
    DAY = auto()  # Day only
    GTD = auto()  # Good Till Date

# === EXECUTION RESULT ===
@dataclass
class ExecutionResult:
    order_id: str
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    avg_execution_price: float
    execution_time: datetime
    fees: float
    slippage: float
    metadata: Dict = field(default_factory=dict)

# === BASE EXECUTOR ===
class BaseExecutor(abc.ABC):
    """Abstract base class for execution engines"""
    def __init__(self, config: Dict):
        self.config = config
        self._active_orders: Dict[str, Dict] = {}
        self._execution_log: List[ExecutionResult] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def _generate_order_id(self) -> str:
        return f"ord-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    def _record_execution(self, result: ExecutionResult):
        self._execution_log.append(result)
        self.logger.info(f"Order {result.order_id} {result.status.name}")

    def calculate_fees(self, order: Dict, is_maker: bool = True) -> float:
        fee_rate = 0.0002 if is_maker else 0.0005
        return fee_rate * order.get("price", 100.0) * order.get("quantity", 0.0)

    @abc.abstractmethod
    def execute_order(self, order: Dict) -> ExecutionResult:
        pass

    @abc.abstractmethod
    def _do_cancel_order(self, order_id: str):
        pass

    def cancel_order(self, order_id: str):
        if order_id not in self._active_orders:
            self.logger.warning(f"Order {order_id} not found")
            return
        self._do_cancel_order(order_id)
