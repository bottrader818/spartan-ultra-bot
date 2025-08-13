from __future__ import annotations
import pytest
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from core.execution.base_executor import (
    BaseExecutor,
    ExecutionResult,
    OrderStatus,
    ValidationError
)


class TestExecutor(BaseExecutor):
    """Concrete implementation for testing base executor functionality"""
    def submit_order(self, *args, **kwargs) -> ExecutionResult:
        order = kwargs.get("order", {})
        oid = self._generate_order_id()
        self._active_orders[oid] = order
        return ExecutionResult(
            order_id=oid,
            status=OrderStatus.NEW,
            fees=self.calculate_fees(order, is_maker=True),
            raw=kwargs,
        )

    def cancel_order(self, order_id: str) -> bool:
        return self._active_orders.pop(order_id, None) is not None

    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.NEW if order_id in self._active_orders else OrderStatus.REJECTED


# ---- Initialization Tests ----
def test_base_executor_initialization():
    """Test that executor initializes with correct defaults"""
    executor = TestExecutor(config={
        "maker_fee_bps": 2.0,
        "taker_fee_bps": 5.0
    })
    assert executor._active_orders == {}
    assert executor._execution_log == []
    assert executor._id_counter == 0
    assert executor._maker_fee_bps == 2.0
    assert executor._taker_fee_bps == 5.0


# ---- Order ID Generation Tests ----
def test_generate_order_id_sequential():
    executor = TestExecutor(config={})
    assert executor._generate_order_id() == "ord-00000001"
    assert executor._generate_order_id() == "ord-00000002"
    assert executor._id_counter == 2


# ---- Fee Calculation Tests ----
@pytest.mark.parametrize("is_maker, bps, expected", [
    (True, 2.0, 0.20),
    (False, 5.0, 0.50),
    (True, 1.5, 0.15),
])
def test_calculate_fees(is_maker, bps, expected):
    executor = TestExecutor(config={
        "maker_fee_bps": bps if is_maker else 0,
        "taker_fee_bps": bps if not is_maker else 0
    })
    order = {"price": 100.0, "quantity": 10}
    assert pytest.approx(executor.calculate_fees(order, is_maker)) == expected


def test_calculate_fees_with_precision():
    executor = TestExecutor(config={
        "maker_fee_bps": 1.25,
        "taker_fee_bps": 3.75
    })
    order = {"price": 123.456, "quantity": 7.891}
    maker_fee = executor.calculate_fees(order, is_maker=True)
    expected_maker = 123.456 * 7.891 * (1.25 / 10_000)
    assert pytest.approx(maker_fee, abs=1e-10) == expected_maker


# ---- Error Handling ----
def test_calculate_fees_missing_fields():
    executor = TestExecutor(config={})
    with pytest.raises(ValidationError):
        executor.calculate_fees({"price": 100})
    with pytest.raises(ValidationError):
        executor.calculate_fees({"quantity": 10})


# ---- Execution Flow ----
def test_submit_order_flow():
    executor = TestExecutor(config={"maker_fee_bps": 2.0})
    order = {"price": 100.0, "quantity": 10, "symbol": "BTC/USD"}
    result = executor.submit_order(order=order)
    assert result.status == OrderStatus.NEW
    assert result.order_id.startswith("ord-")
    assert pytest.approx(result.fees) == 0.20
    assert result.raw["order"] == order


# ---- Concurrency Safety ----
def test_order_id_generation_thread_safety():
    executor = TestExecutor(config={})
    ids = set()

    def generate_id():
        ids.add(executor._generate_order_id())

    with ThreadPoolExecutor(max_workers=10) as pool:
        for _ in range(100):
            pool.submit(generate_id)

    assert len(ids) == 100
    assert executor._id_counter == 100


# ---- Retry Example ----
@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
def test_flaky_operation():
    assert True
