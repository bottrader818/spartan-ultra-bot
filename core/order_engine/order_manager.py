from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import logging
import uuid
import pandas as pd
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()

class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()
    TWAP = auto()
    VWAP = auto()

class OrderSide(Enum):
    BUY = auto()
    SELL = auto()
    SHORT = auto()

@dataclass
class Order:
    order_id: str
    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_offset: Optional[float] = None
    time_in_force: timedelta = timedelta(days=1)
    strategy_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    fills: List[Dict] = field(default_factory=list)
    avg_fill_price: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None

class OrderManager:
    """
    Institutional Order Management System with:
    - Full order lifecycle tracking
    - Execution analytics
    - Strategy attribution
    - Performance metrics
    - Compliance checks
    """

    def __init__(self, max_history: int = 10_000):
        self._active_orders: Dict[str, Order] = {}
        self._order_history: deque = deque(maxlen=max_history)
        self._strategy_orders: Dict[str, List[str]] = defaultdict(list)
        self._symbol_orders: Dict[str, List[str]] = defaultdict(list)
        self._execution_analytics = {
            'total_orders': 0,
            'fills': 0,
            'volume': 0.0,
            'avg_slippage': 0.0,
            'avg_commission': 0.0
        }
        logger.info("OrderManager initialized")

    def create_order(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
        order_type: OrderType,
        strategy_id: Optional[str] = None,
        **kwargs
    ) -> Order:
        """
        Create and record a new order with institutional features
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: BUY/SELL/SHORT
            order_type: Order type
            strategy_id: Strategy identifier
            **kwargs: Additional order parameters
            
        Returns:
            Order object with unique ID
        """
        order_id = f"ord-{uuid.uuid4().hex[:8]}"
        order = Order(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            timestamp=datetime.utcnow(),
            strategy_id=strategy_id,
            **kwargs
        )
        
        self._active_orders[order_id] = order
        self._order_history.append(order)
        
        if strategy_id:
            self._strategy_orders[strategy_id].append(order_id)
        
        self._symbol_orders[symbol].append(order_id)
        self._execution_analytics['total_orders'] += 1
        
        logger.info(f"Created order {order_id} for {symbol} {side.name} {quantity}")
        return order

    def record_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: float,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> bool:
        """
        Record order fill with execution details
        
        Args:
            order_id: Order identifier
            fill_price: Execution price
            fill_quantity: Filled quantity
            commission: Trade commission
            slippage: Execution slippage
            
        Returns:
            True if fill was recorded successfully
        """
        if order_id not in self._active_orders:
            logger.error(f"Order {order_id} not found or already completed")
            return False
            
        order = self._active_orders[order_id]
        fill = {
            'timestamp': datetime.utcnow(),
            'price': fill_price,
            'quantity': fill_quantity,
            'commission': commission,
            'slippage': slippage
        }
        
        order.fills.append(fill)
        filled_qty = sum(f['quantity'] for f in order.fills)
        
        # Update order status
        if filled_qty >= order.quantity * 0.999:  # Allow for rounding
            order.status = OrderStatus.FILLED
            order.avg_fill_price = sum(f['price']*f['quantity'] for f in order.fills) / filled_qty
            order.commission = sum(f['commission'] for f in order.fills)
            order.slippage = sum(f['slippage'] for f in order.fills) / len(order.fills)
            self._active_orders.pop(order_id)
            
            # Update analytics
            self._execution_analytics['fills'] += 1
            self._execution_analytics['volume'] += filled_qty
            self._execution_analytics['avg_slippage'] = (
                self._execution_analytics['avg_slippage'] * (self._execution_analytics['fills'] - 1) + 
                slippage
            ) / self._execution_analytics['fills']
            self._execution_analytics['avg_commission'] = (
                self._execution_analytics['avg_commission'] * (self._execution_analytics['fills'] - 1) + 
                commission
            ) / self._execution_analytics['fills']
            
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            
        logger.info(f"Recorded fill for {order_id}: {fill_quantity}@{fill_price}")
        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if cancellation was successful
        """
        if order_id not in self._active_orders:
            logger.error(f"Order {order_id} not found or already completed")
            return False
            
        self._active_orders[order_id].status = OrderStatus.CANCELLED
        self._active_orders.pop(order_id)
        logger.info(f"Cancelled order {order_id}")
        return True

    def get_strategy_orders(self, strategy_id: str) -> List[Order]:
        """Get all orders for a specific strategy"""
        return [
            self._active_orders[oid] for oid in self._strategy_orders.get(strategy_id, [])
            if oid in self._active_orders
        ]

    def get_symbol_orders(self, symbol: str, active_only: bool = True) -> List[Order]:
        """Get all orders for a specific symbol"""
        orders = [
            self._active_orders[oid] for oid in self._symbol_orders.get(symbol, [])
            if oid in self._active_orders
        ]
        if not active_only:
            orders += [o for o in self._order_history if o.symbol == symbol]
        return orders

    def get_execution_analytics(self) -> Dict:
        """Get execution performance metrics"""
        return self._execution_analytics.copy()

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Check current order status"""
        if order_id in self._active_orders:
            return self._active_orders[order_id].status
        for order in self._order_history:
            if order.order_id == order_id:
                return order.status
        return None

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    om = OrderManager()
    
    # Create institutional orders
    order1 = om.create_order(
        symbol="AAPL",
        quantity=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=150.25,
        strategy_id="mean_reversion_001",
        tags={"portfolio": "tech", "risk_level": "medium"}
    )
    
    order2 = om.create_order(
        symbol="TSLA",
        quantity=50,
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LIMIT,
        limit_price=700.00,
        stop_price=695.50,
        strategy_id="momentum_004"
    )
    
    # Simulate fills
    om.record_fill(order1.order_id, 150.20, 100, commission=1.50, slippage=-0.05)
    om.record_fill(order2.order_id, 699.80, 25, commission=1.25, slippage=-0.20)
    
    # Analytics
    print(f"Execution Analytics: {om.get_execution_analytics()}")
    
    # Strategy orders
    strat_orders = om.get_strategy_orders("mean_reversion_001")
    print(f"Strategy Orders: {[o.order_id for o in strat_orders]}")
