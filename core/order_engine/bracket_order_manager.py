from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum, auto
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = auto()
    SELL = auto()

class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()

class BracketOrderStatus(Enum):
    PENDING = auto()
    ENTRY_FILLED = auto()
    TP_FILLED = auto()
    SL_FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()

@dataclass
class BracketOrder:
    """
    Institutional Bracket Order with:
    - Entry order
    - Take profit order
    - Stop loss order
    - Trailing stop options
    - Time-based triggers
    """
    order_id: str
    symbol: str
    entry_order: Dict
    take_profit_order: Dict
    stop_loss_order: Dict
    side: OrderSide
    quantity: float
    status: BracketOrderStatus = BracketOrderStatus.PENDING
    created_at: datetime = datetime.utcnow()
    tags: Dict[str, str] = field(default_factory=dict)
    trailing_stop: Optional[Dict] = None
    time_activation: Optional[datetime] = None
    strategy_id: Optional[str] = None

class BracketOrderManager:
    def __init__(self):
        self.active_orders: Dict[str, BracketOrder] = {}
        self.order_history: List[BracketOrder] = []
        logger.info("BracketOrderManager initialized")

    def create_bracket_order(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
        entry_type: OrderType,
        entry_price: Optional[float],
        take_profit_price: float,
        stop_loss_price: float,
        trailing_stop_config: Optional[Dict] = None,
        time_activation: Optional[datetime] = None,
        strategy_id: Optional[str] = None,
        tags: Optional[Dict] = None
    ) -> BracketOrder:
        """
        Create a professional bracket order with OCO (One-Cancels-Other) logic
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: BUY or SELL
            entry_type: Order type for entry
            entry_price: Price for limit orders (None for market)
            take_profit_price: Take profit price
            stop_loss_price: Stop loss price
            trailing_stop_config: Dict with 'activation_price' and 'trail_by'
            time_activation: Optional time-based activation
            strategy_id: Strategy identifier
            tags: Key-value metadata
            
        Returns:
            BracketOrder object
        """
        order_id = f"bracket-{uuid.uuid4().hex[:8]}"
        
        entry_order = {
            "order_id": f"{order_id}-entry",
            "type": entry_type,
            "price": entry_price,
            "quantity": quantity,
            "side": side
        }
        
        take_profit_order = {
            "order_id": f"{order_id}-tp",
            "type": OrderType.LIMIT,
            "price": take_profit_price,
            "quantity": quantity,
            "side": OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        }
        
        stop_loss_order = {
            "order_id": f"{order_id}-sl",
            "type": OrderType.STOP,
            "price": stop_loss_price,
            "quantity": quantity,
            "side": OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        }
        
        bracket_order = BracketOrder(
            order_id=order_id,
            symbol=symbol,
            entry_order=entry_order,
            take_profit_order=take_profit_order,
            stop_loss_order=stop_loss_order,
            side=side,
            quantity=quantity,
            trailing_stop=trailing_stop_config,
            time_activation=time_activation,
            strategy_id=strategy_id,
            tags=tags or {}
        )
        
        self.active_orders[order_id] = bracket_order
        logger.info(f"Created bracket order {order_id} for {symbol} {side.name} {quantity}")
        return bracket_order

    def update_bracket_order(
        self,
        order_id: str,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        trailing_stop_config: Optional[Dict] = None
    ) -> bool:
        """
        Update bracket order parameters
        
        Args:
            order_id: Bracket order ID
            take_profit_price: New take profit price
            stop_loss_price: New stop loss price
            trailing_stop_config: New trailing stop config
            
        Returns:
            True if update was successful
        """
        if order_id not in self.active_orders:
            logger.error(f"Bracket order {order_id} not found")
            return False
            
        order = self.active_orders[order_id]
        
        if take_profit_price is not None:
            order.take_profit_order["price"] = take_profit_price
            logger.debug(f"Updated TP for {order_id} to {take_profit_price}")
            
        if stop_loss_price is not None:
            order.stop_loss_order["price"] = stop_loss_price
            logger.debug(f"Updated SL for {order_id} to {stop_loss_price}")
            
        if trailing_stop_config is not None:
            order.trailing_stop = trailing_stop_config
            logger.debug(f"Updated trailing stop for {order_id}")
            
        return True

    def cancel_bracket_order(self, order_id: str) -> bool:
        """
        Cancel entire bracket order
        
        Args:
            order_id: Bracket order ID
            
        Returns:
            True if cancellation was successful
        """
        if order_id not in self.active_orders:
            logger.error(f"Bracket order {order_id} not found")
            return False
            
        order = self.active_orders.pop(order_id)
        order.status = BracketOrderStatus.CANCELLED
        self.order_history.append(order)
        logger.info(f"Cancelled bracket order {order_id}")
        return True

    def on_order_fill(self, filled_order_id: str) -> bool:
        """
        Handle order fill events and manage bracket order state
        
        Args:
            filled_order_id: The ID of the order that was filled
            
        Returns:
            True if state was updated successfully
        """
        # Implementation would:
        # 1. Determine which bracket order this fill belongs to
        # 2. Update the bracket order status
        # 3. Cancel the other legs if needed
        # 4. Move to order history if completed
        pass

    def get_active_bracket_orders(self, strategy_id: Optional[str] = None) -> List[BracketOrder]:
        """
        Get all active bracket orders, optionally filtered by strategy
        
        Args:
            strategy_id: Optional strategy filter
            
        Returns:
            List of active bracket orders
        """
        if strategy_id:
            return [o for o in self.active_orders.values() if o.strategy_id == strategy_id]
        return list(self.active_orders.values())

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = BracketOrderManager()
    
    # Create professional bracket order
    bracket = manager.create_bracket_order(
        symbol="AAPL",
        quantity=100,
        side=OrderSide.BUY,
        entry_type=OrderType.LIMIT,
        entry_price=150.25,
        take_profit_price=155.50,
        stop_loss_price=148.00,
        trailing_stop_config={"activation_price": 152.00, "trail_by": 0.5},  # Trail by $0.50
        strategy_id="mean_reversion_001",
        tags={"portfolio": "tech", "risk_level": "high"}
    )
    
    print(f"Created bracket order: {bracket.order_id}")
    print(f"Entry: {bracket.entry_order['price']}")
    print(f"TP: {bracket.take_profit_order['price']}")
    print(f"SL: {bracket.stop_loss_order['price']}")
    
    # Update order dynamically
    manager.update_bracket_order(
        order_id=bracket.order_id,
        take_profit_price=156.00,
        stop_loss_price=149.00
    )
    
    # Get active orders for strategy
    active_orders = manager.get_active_bracket_orders(strategy_id="mean_reversion_001")
    print(f"Active orders: {len(active_orders)}")
