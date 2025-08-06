import time
import joblib
from collections import deque

class SmartOrderRouter:
    def __init__(self):
        self.order_history = deque(maxlen=1000)
        self.slippage_model = joblib.load('models/slippage_model.pkl')

    def route_order(self, order):
        if self._detect_predatory_conditions(order):
            return self._execute_defensive(order)
        return self._execute_aggressive(order)

    def _execute_aggressive(self, order):
        slices = self._time_slice_order(order)
        filled = []
        for slice in slices:
            execution = self._send_to_exchange(slice)
            X = self._extract_features(execution)
            execution['slippage'] = self.slippage_model.predict([X])[0]
            filled.append(execution)
        return filled

    def _detect_predatory_conditions(self, order):
        recent_orders = [
            o for o in self.order_history
            if o['symbol'] == order['symbol'] and time.time() - o['time'] < 60
        ]
        return len(recent_orders) > 5

    def _time_slice_order(self, order):
        return [order]

    def _send_to_exchange(self, order):
        order['time'] = time.time()
        return order

    def _extract_features(self, execution):
        return [
            execution.get('price', 0),
            execution.get('volume', 0),
            execution.get('latency', 0),
            execution.get('order_type', 0)
        ]
