import pytest
from core.execution.smart_order_router import SmartOrderRouter
from core.execution.institutional_router import SmartOrderRouter as InstitutionalRouter
from core.execution.base_executor import OrderType

@pytest.mark.parametrize("Router", [SmartOrderRouter, InstitutionalRouter])
def test_order_lifecycle(Router):
    router = Router(config={
        'default_order_type': OrderType.LIMIT,
        'default_time_in_force': 'GTC',
        'max_slippage': 0.002
    })
    order = {
        "symbol": "GOOGL",
        "quantity": 100,
        "order_type": OrderType.LIMIT,
        "price": 120.0,
        "urgency": 0.6,
        "liquidity_data": {"liquidity_score": 0.8, "adv": 1_000_000}
    }
    result = router.execute_order(order)
    assert result.filled_quantity > 0
    router.cancel_order(result.order_id)
