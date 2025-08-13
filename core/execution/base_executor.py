from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    DAY = "day"


class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class ExecutionResult:
    def __init__(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: float = 0.0,
        avg_price: Optional[float] = None,
        fees: Optional[float] = None,
        raw: Optional[Dict[str, Any]] = None,
    ):
        self.order_id = order_id
        self.status = status
        self.filled_quantity = filled_quantity
        self.avg_price = avg_price
        self.fees = fees
        self.raw = raw or {}

    def __repr__(self) -> str:
        return (
            f"<ExecutionResult(order_id={self.order_id}, status={self.status}, "
            f"filled_quantity={self.filled_quantity}, avg_price={self.avg_price}, "
            f"fees={self.fees})>"
        )


class ValidationError(Exception):
    """Raised when an order fails validation."""


class BaseExecutor(ABC):
    """
    Base class for executors. Provides:
      - _active_orders, _execution_log, _id_counter
      - thread-safe order id generation
      - fee calculation in basis points (bps)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._active_orders: Dict[str, Any] = {}
        self._execution_log: list = []
        self._id_counter: int = 0
        self._id_lock = threading.Lock()

        # Fee config (bps)
        self._maker_fee_bps: float = float(config.get("maker_fee_bps", 0.0))
        self._taker_fee_bps: float = float(config.get("taker_fee_bps", 0.0))

    def _generate_order_id(self) -> str:
        """Generates a sequential order ID with zero padding (thread-safe)."""
        with self._id_lock:
            self._id_counter += 1
            return f"ord-{self._id_counter:08d}"

    def calculate_fees(self, order: Dict[str, Any], is_maker: bool = False) -> float:
        """
        Calculate fees using basis points (bps).
        Raises ValidationError if price/quantity missing.
        """
        if "price" not in order or "quantity" not in order:
            raise ValidationError("Order must include 'price' and 'quantity'.")

        price = float(order["price"])
        quantity = float(order["quantity"])
        bps = self._maker_fee_bps if is_maker else self._taker_fee_bps
        fee_rate = bps / 10_000.0
        return price * quantity * fee_rate

    @abstractmethod
    def submit_order(self, *args, **kwargs) -> ExecutionResult:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        pass
