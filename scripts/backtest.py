from core.strategies.dtw_neural import DTWNeuralStrategy
import pandas as pd

def run_backtest(data_path):
    df = pd.read_csv(data_path)
    strategy = DTWNeuralStrategy()
    results = []

    for i in range(len(df)):
        window = df.iloc[max(0, i-25):i+1]
        tick = {
            'symbol': 'TEST',
            'timestamp': window.iloc[-1]['timestamp'],
            'close': list(window['close']),
            'features': [0.1] * strategy.model.input_shape[1]  # Replace with real feature logic
        }

        signal = strategy.generate_signal(tick)
        if signal:
            results.append(signal)

    print(f"📊 Generated {len(results)} signals from backtest")
    return results

if __name__ == '__main__':
    run_backtest('data/sample_backtest.csv')
