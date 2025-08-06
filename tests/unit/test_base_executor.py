import pytest
from datetime import datetime
from core.execution.base_executor import BaseExecutor, ExecutionResult, OrderStatus

class ConcreteExecutor(BaseExecutor):
    def execute_order(self, order):
        return ExecutionResult(
            order_id="test",
            status=OrderStatus.FILLED,
            filled_quantity=100,
            remaining_quantity=0,
            avg_execution_price=10.0,
            execution_time=datetime.now(),
            fees=0.0,
            slippage=0.0
        )

    def _do_cancel_order(self, order_id):
        pass

def test_base_executor_initialization():
    executor = ConcreteExecutor(config={})
    assert executor._active_orders == {}
    assert executor._execution_log == []

def test_generate_order_id():
    executor = ConcreteExecutor(config={})
    order_id = executor._generate_order_id()
    assert order_id.startswith("ord-")
    assert len(order_id) == 26

def test_calculate_fees():
    executor = ConcreteExecutor(config={})
    order = {"price": 100.0, "quantity": 10}
    assert executor.calculate_fees(order, is_maker=True) == 0.2
    assert executor.calculate_fees(order, is_maker=False) == 0.5
