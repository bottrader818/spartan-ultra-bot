import numpy as np
from fastdtw import fastdtw
from tensorflow.keras.models import load_model
from core.risk.stop_system import HybridStopSystem

class DTWNeuralStrategy:
    def __init__(self):
        self.model = load_model('models/dtw_predictor.h5')
        self.stop_system = HybridStopSystem()
        self.dtw_window = 20
        self.volatility_window = 14

    def generate_signal(self, data):
        if len(data['close']) < self.dtw_window:
            return None

        prices = np.array(data['close'][-self.dtw_window:])
        mean_line = np.convolve(prices, np.ones(5) / 5, mode='valid')

        if len(mean_line) != len(prices):
            prices = prices[-len(mean_line):]

        distance, _ = fastdtw(prices, mean_line)
        features = self._create_features(data)

        if features.shape[0] != self.model.input_shape[1]:
            return None

        neural_conf = self.model.predict(features.reshape(1, -1))[0][0]
        volatility = np.std(prices[-self.volatility_window:])

        if distance > 2.0 and neural_conf > 0.7:
            action = 'buy' if prices[-1] < mean_line[-1] else 'sell'
            return {
                'symbol': data['symbol'],
                'action': action,
                'size': self._calculate_size(neural_conf, volatility),
                'stop': self.stop_system.calculate_stop(data, action),
                'take_profit': self._calculate_tp(prices, action),
                'confidence': min(0.99, distance * neural_conf)
            }
        return None

    def _calculate_size(self, confidence, volatility):
        base_size = min(0.25, 0.1 + (confidence * 0.15))
        return base_size * (1 + volatility)

    def _calculate_tp(self, prices, action):
        momentum = prices[-1] - np.mean(prices[-5:])
        return prices[-1] + momentum * 2.5 if action == 'buy' else prices[-1] - momentum * 2.5

    def _create_features(self, data):
        return np.array(data['features'][-self.model.input_shape[1]:])
