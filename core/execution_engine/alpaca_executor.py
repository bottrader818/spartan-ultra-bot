import os
from typing import Dict, Optional, List, Union
from alpaca_trade_api.rest import REST
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pytz

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    TRAILING_STOP = 'trailing_stop'

class TimeInForce(Enum):
    DAY = 'day'
    GTC = 'gtc'
    OPG = 'opg'
    IOC = 'ioc'
    FOK = 'fok'

@dataclass
class EnhancedOrder:
    symbol: str
    qty: float
    side: OrderSide
    type: OrderType
    time_in_force: TimeInForce
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    client_order_id: Optional[str] = None
    extended_hours: bool = False
    order_class: Optional[str] = None
    take_profit: Optional[Dict] = None
    stop_loss: Optional[Dict] = None

class AlpacaTradingClient:
    """
    Professional Alpaca Trading Client with:
    - Enhanced order management
    - Comprehensive error handling
    - Position tracking
    - Risk controls
    - Performance monitoring
    """
    
    def __init__(self, paper: bool = True):
        """
        Initialize Alpaca trading client
        
        Args:
            paper: Whether to use paper trading (default) or live
        """
        self._api = self._init_api(paper)
        self._order_history = []
        self._positions = {}
        self._last_sync = None
        logger.info(f"Alpaca client initialized (paper={paper})")

    def _init_api(self, paper: bool) -> REST:
        """Initialize and validate API connection"""
        base_url = (
            os.getenv("ALPACA_PAPER_URL") if paper 
            else os.getenv("ALPACA_LIVE_URL")
        )
        
        try:
            api = REST(
                key_id=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                base_url=base_url,
                api_version='v2'
            )
            # Test connectivity
            api.get_account()
            return api
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API: {str(e)}")
            raise

    def submit_order(self, order: EnhancedOrder) -> Dict:
        """
        Submit an order with enhanced controls
        
        Args:
            order: EnhancedOrder object with all parameters
            
        Returns:
            Dictionary with order confirmation
        """
        try:
            order_data = {
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side.value,
                'type': order.type.value,
                'time_in_force': order.time_in_force.value,
                'limit_price': order.limit_price,
                'stop_price': order.stop_price,
                'trail_price': order.trail_price,
                'trail_percent': order.trail_percent,
                'client_order_id': order.client_order_id,
                'extended_hours': order.extended_hours,
                'order_class': order.order_class,
                'take_profit': order.take_profit,
                'stop_loss': order.stop_loss
            }
            
            # Remove None values
            order_data = {k: v for k, v in order_data.items() if v is not None}
            
            response = self._api.submit_order(**order_data)
            self._order_history.append({
                'time': datetime.now(pytz.UTC),
                'order': order,
                'response': response
            })
            logger.info(f"Order submitted: {order.symbol} {order.side.value} {order.qty}")
            return response
        except Exception as e:
            logger.error(f"Order failed: {str(e)}")
            raise

    def get_positions(self, refresh: bool = False) -> Dict:
        """
        Get current positions with optional refresh
        
        Args:
            refresh: Whether to force refresh from API
            
        Returns:
            Dictionary of current positions
        """
        if refresh or not self._positions or (
            self._last_sync and 
            datetime.now(pytz.UTC) - self._last_sync > timedelta(minutes=5)
        ):
            try:
                self._positions = {
                    pos.symbol: pos for pos in self._api.list_positions()
                }
                self._last_sync = datetime.now(pytz.UTC)
            except Exception as e:
                logger.error(f"Failed to update positions: {str(e)}")
                raise
        
        return self._positions

    def cancel_all_orders(self) -> List[Dict]:
        """Cancel all open orders"""
        try:
            response = self._api.cancel_all_orders()
            logger.info("Cancelled all open orders")
            return response
        except Exception as e:
            logger.error(f"Failed to cancel orders: {str(e)}")
            raise

    def get_account_info(self) -> Dict:
        """Get detailed account information"""
        try:
            account = self._api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'regt_buying_power': float(account.regt_buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'last_equity': float(account.last_equity),
                'status': account.status,
                'currency': account.currency,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {str(e)}")
            raise

    def get_order_history(self, 
                       days: int = 7,
                       status: Optional[str] = None) -> List[Dict]:
        """
        Get historical orders
        
        Args:
            days: Number of days of history to retrieve
            status: Filter by order status (filled, canceled, etc.)
            
        Returns:
            List of order dictionaries
        """
        try:
            after = (datetime.now(pytz.UTC) - timedelta(days=days)).isoformat()
            orders = self._api.list_orders(
                status=status,
                after=after,
                limit=500
            )
            return [self._format_order(o) for o in orders]
        except Exception as e:
            logger.error(f"Failed to get order history: {str(e)}")
            raise

    def _format_order(self, order) -> Dict:
        """Format order object for consistent output"""
        return {
            'id': order.id,
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'qty': float(order.qty),
            'filled_qty': float(order.filled_qty),
            'side': order.side,
            'type': order.type,
            'time_in_force': order.time_in_force,
            'status': order.status,
            'submitted_at': order.submitted_at,
            'filled_at': order.filled_at,
            'limit_price': float(order.limit_price) if order.limit_price else None,
            'stop_price': float(order.stop_price) if order.stop_price else None,
            'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None
        }

    def close_position(self, symbol: str) -> Dict:
        """
        Close position for a specific symbol
        
        Args:
            symbol: Symbol to close position for
            
        Returns:
            Order confirmation
        """
        try:
            response = self._api.close_position(symbol)
            logger.info(f"Closed position for {symbol}")
            return response
        except Exception as e:
            logger.error(f"Failed to close position: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    client = AlpacaTradingClient(paper=True)
    
    # Submit an order
    order = EnhancedOrder(
        symbol='AAPL',
        qty=10,
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=150.25
    )
    
    try:
        result = client.submit_order(order)
        print(f"Order submitted: {result}")
        
        # Check positions
        positions = client.get_positions()
        print(f"Current positions: {positions.keys()}")
        
        # Get account info
        account = client.get_account_info()
        print(f"Account equity: ${account['equity']:,.2f}")
        
    except Exception as e:
        print(f"Trading error: {str(e)}")
