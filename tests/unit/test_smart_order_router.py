import pytest
from core.execution.smart_order_router import SmartOrderRouter
from core.execution.base_executor import OrderStatus, OrderType, TimeInForce

@pytest.fixture
def router():
    return SmartOrderRouter(config={
        'default_order_type': OrderType.LIMIT,
        'default_time_in_force': TimeInForce.GTC,
        'max_slippage': 0.002
    })

def test_smart_order_router_execution(router):
    order = {
        "symbol": "AAPL",
        "quantity": 100,
        "order_type": OrderType.MARKET,
        "price": 50.0,
        "urgency": 0.9,
        "liquidity_data": {"liquidity_score": 0.9, "adv": 1_000_000}
    }
    result = router.execute_order(order)

    assert result.status == OrderStatus.FILLED
    assert result.filled_quantity == 100
    assert result.remaining_quantity == 0
    assert result.avg_execution_price > 0
    assert "routing_strategy" in result.metadata

def test_order_cancellation(router):
    order = {
        "symbol": "AAPL",
        "quantity": 50,
        "order_type": OrderType.LIMIT,
        "price": 100.0,
        "urgency": 0.3,
        "liquidity_data": {"liquidity_score": 0.9, "adv": 1_000_000}
    }
    result = router.execute_order(order)
    router.cancel_order(result.order_id)
    assert result.order_id not in router._active_orders
