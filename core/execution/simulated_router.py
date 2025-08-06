from datetime import datetime
from random import uniform

from core.execution.base_executor import BaseExecutor, ExecutionResult, OrderStatus, OrderType

class SmartOrderRouter(BaseExecutor):
    """
    Basic simulated Smart Order Router for Alpaca-style paper trading.
    Fills orders instantly with mock slippage/fees.
    """
    
    def execute_order(self, order):
        order_id = self._generate_order_id()
        symbol = order['symbol']
        quantity = order['quantity']
        order_type = order.get('order_type', self.config['default_order_type'])
        price = order.get('price', 100.0)
        tif = order.get('time_in_force', self.config['default_time_in_force'])

        slippage_pct = uniform(0.0001, self.config['max_slippage'])
        slippage = price * slippage_pct
        executed_price = price + slippage if order_type == OrderType.MARKET else price

        fee = self.calculate_fees(order, is_maker=(order_type != OrderType.MARKET))

        result = ExecutionResult(
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            remaining_quantity=0.0,
            avg_execution_price=executed_price,
            execution_time=datetime.now(),
            fees=fee,
            slippage=slippage,
            metadata={'symbol': symbol, 'tif': tif.name}
        )

        self._record_execution(result)
        return result

    def _do_cancel_order(self, order_id):
        if order_id in self._active_orders:
            self._active_orders.pop(order_id)
