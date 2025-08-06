# test_router.py
from core.execution.smart_order_router import SmartOrderRouter
from core.execution.base_executor import OrderType, TimeInForce

if __name__ == "__main__":
    router = SmartOrderRouter(config={
        'default_order_type': OrderType.LIMIT,
        'default_time_in_force': TimeInForce.GTC,
        'max_slippage': 0.002
    })

    order = {
        'symbol': 'AAPL',
        'quantity': 100,
        'order_type': OrderType.MARKET,
        'urgency': 0.9,
        'liquidity_data': {
            'liquidity_score': 0.8,
            'adv': 1_000_000
        }
    }

    result = router.execute_order(order)
    print("Execution Result:")
    print(f"Status: {result.status.name}")
    print(f"Filled Quantity: {result.filled_quantity}")
    print(f"Average Price: {result.avg_execution_price:.2f}")
    print(f"Fees: {result.fees:.4f}")
    print(f"Slippage: {result.slippage:.6f}")
    print(f"Metadata: {result.metadata}")

