import os
from core.strategies.dtw_neural import DTWNeuralStrategy
from core.engine.execution.order_router import SmartOrderRouter
from core.engine.data.stream_processor import DataProcessor

if __name__ == '__main__':
    os.environ['ENV_PATH'] = 'config/environments/live.env'

    print("🚀 Starting QuantBotPro in LIVE mode")

    strategy = DTWNeuralStrategy()
    router = SmartOrderRouter()
    data_processor = DataProcessor()

    # Fake tick for testing purposes
    test_tick = {
        'symbol': 'ETH-USD',
        'timestamp': '2025-07-24T12:00:00Z',
        'close': 3000.0,
        'volume': 10.5,
        'high': 3010.0,
        'low': 2985.0,
        'features': [0.1] * strategy.model.input_shape[1]
    }

    signal = strategy.generate_signal(test_tick)
    if signal:
        print(f"✅ Signal: {signal}")
        executions = router.route_order(signal)
        print(f"🧾 Executions: {executions}")
    else:
        print("❌ No signal generated")
