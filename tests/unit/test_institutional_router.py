import pytest
from unittest.mock import patch
from core.execution.institutional_router import SmartOrderRouter
from core.execution.base_executor import OrderStatus, OrderType

@pytest.fixture
def router():
    return SmartOrderRouter(config={
        'default_order_type': OrderType.LIMIT,
        'default_time_in_force': 'GTC',
        'max_slippage': 0.002
    })

def test_large_order_execution(router):
    order = {
        "symbol": "TSLA",
        "quantity": 10000,
        "order_type": OrderType.VWAP,
        "price": 600.0,
        "urgency": 0.7,
        "liquidity_data": {"liquidity_score": 0.7, "adv": 1_000_000}
    }
    result = router.execute_order(order)
    assert result.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
    assert result.filled_quantity > 0
    assert result.avg_execution_price > 0

@patch("concurrent.futures.ThreadPoolExecutor.submit")
def test_concurrent_execution(mock_submit, router):
    order = {
        "symbol": "MSFT",
        "quantity": 5000,
        "order_type": OrderType.LIMIT,
        "price": 250.0,
        "urgency": 0.5,
        "liquidity_data": {"liquidity_score": 0.6, "adv": 1_000_000}
    }
    router.execute_order(order)
    assert mock_submit.called
