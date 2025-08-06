from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import numpy as np

class PositionStatus(Enum):
    """Status of a position"""
    OPEN = auto()
    CLOSED = auto()
    FLIPPED = auto()
    PARTIALLY_CLOSED = auto()

class PositionDirection(Enum):
    """Direction of a position"""
    LONG = auto()
    SHORT = auto()
    NEUTRAL = auto()

@dataclass
class Trade:
    """Trade execution details"""
    timestamp: datetime
    symbol: str
    quantity: float
    price: float
    direction: PositionDirection
    commission: float = 0.0
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Detailed position information"""
    symbol: str
    quantity: float
    avg_price: float
    direction: PositionDirection
    status: PositionStatus = PositionStatus.OPEN
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    initial_value: float = 0.0
    current_value: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    trades: List[Trade] = field(default_factory=list)
    risk_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class PositionManager:
    """
    Professional-Grade Position Manager with:
    - Comprehensive position tracking
    - Trade history recording
    - PnL calculation
    - Risk metrics
    - Position status monitoring
    - Performance analytics
    """
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.closed_positions: Dict[str, List[Position]] = defaultdict(list)
        self.trade_history: List[Trade] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Statistics
        self._stats = {
            'total_positions': 0,
            'open_positions': 0,
            'closed_positions': 0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'winning_positions': 0,
            'losing_positions': 0,
            'long_positions': 0,
            'short_positions': 0
        }
        self._lock = threading.RLock()
    
    def update(self,
              symbol: str,
              quantity: float,
              price: float,
              direction: PositionDirection = PositionDirection.LONG,
              commission: float = 0.0,
              strategy: Optional[str] = None,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> Position:
        """
        Update or create a position
        
        Args:
            symbol: Trading symbol
            quantity: Quantity of the position
            price: Execution price
            direction: LONG or SHORT
            commission: Trade commission
            strategy: Strategy name
            tags: Position tags
            metadata: Additional metadata
            
        Returns:
            Updated or created Position
        """
        with self._lock:
            trade = Trade(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                quantity=quantity,
                price=price,
                direction=direction,
                commission=commission,
                strategy=strategy,
                tags=tags or [],
                metadata=metadata or {}
            )
            self.trade_history.append(trade)
            self._stats['total_trades'] += 1
            
            if symbol not in self.positions:
                return self._create_new_position(trade)
            else:
                return self._update_existing_position(trade)
    
    def _create_new_position(self, trade: Trade) -> Position:
        """Create a new position from a trade"""
        position = Position(
            symbol=trade.symbol,
            quantity=trade.quantity,
            avg_price=trade.price,
            direction=trade.direction,
            initial_value=trade.quantity * trade.price,
            current_value=trade.quantity * trade.price,
            trades=[trade],
            tags=trade.tags.copy(),
            metadata=trade.metadata.copy()
        )
        
        self.positions[trade.symbol] = position
        self._stats['total_positions'] += 1
        self._stats['open_positions'] += 1
        
        if trade.direction == PositionDirection.LONG:
            self._stats['long_positions'] += 1
        else:
            self._stats['short_positions'] += 1
            
        self.logger.info(f"Created new {trade.direction.name} position for {trade.symbol}: "
                        f"{trade.quantity} @ {trade.price}")
        
        return position
    
    def _update_existing_position(self, trade: Trade) -> Position:
        """Update an existing position with a new trade"""
        position = self.positions[trade.symbol]
        original_direction = position.direction
        
        # Calculate new average price and quantity
        if position.direction == trade.direction:
            # Adding to position
            total_qty = position.quantity + trade.quantity
            position.avg_price = (
                (position.avg_price * position.quantity + 
                 trade.price * trade.quantity) / total_qty
            )
            position.quantity = total_qty
            position.status = PositionStatus.OPEN
        else:
            # Reducing or flipping position
            if trade.quantity < position.quantity:
                # Partial close
                closed_qty = trade.quantity
                remaining_qty = position.quantity - trade.quantity
                closed_value = closed_qty * position.avg_price
                closed_pnl = self._calculate_pnl(
                    closed_qty, position.avg_price, trade.price, position.direction
                )
                
                position.quantity = remaining_qty
                position.realized_pnl += closed_pnl
                position.status = PositionStatus.PARTIALLY_CLOSED
                
                self.logger.info(f"Partially closed {position.direction.name} position for {trade.symbol}: "
                               f"Closed {closed_qty} @ {trade.price}, PnL: {closed_pnl:.2f}")
            else:
                # Full close or flip
                closed_pnl = self._calculate_pnl(
                    position.quantity, position.avg_price, trade.price, position.direction
                )
                position.realized_pnl += closed_pnl
                
                if trade.quantity == position.quantity:
                    # Full close
                    self._close_position(position, trade, closed_pnl)
                    return position
                else:
                    # Position flip
                    flip_qty = trade.quantity - position.quantity
                    position.quantity = flip_qty
                    position.avg_price = trade.price
                    position.direction = trade.direction
                    position.status = PositionStatus.FLIPPED
                    
                    self.logger.info(f"Flipped {trade.symbol} position from {original_direction.name} to "
                                   f"{trade.direction.name}: PnL: {closed_pnl:.2f}")
        
        # Update common fields
        position.current_value = position.quantity * trade.price
        position.trades.append(trade)
        position.updated_at = datetime.utcnow()
        
        # Update tags and metadata
        position.tags.extend(tag for tag in trade.tags if tag not in position.tags)
        position.metadata.update(trade.metadata)
        
        return position
    
    def _close_position(self, position: Position, trade: Trade, pnl: float) -> None:
        """Close an existing position"""
        self.closed_positions[position.symbol].append(position)
        del self.positions[position.symbol]
        
        self._stats['open_positions'] -= 1
        self._stats['closed_positions'] += 1
        self._stats['total_pnl'] += pnl
        
        if pnl > 0:
            self._stats['winning_positions'] += 1
        else:
            self._stats['losing_positions'] += 1
        
        self.logger.info(f"Closed {position.direction.name} position for {trade.symbol}: "
                        f"PnL: {pnl:.2f}, Total PnL: {self._stats['total_pnl']:.2f}")
    
    def _calculate_pnl(self, quantity: float, entry_price: float, exit_price: float, 
                      direction: PositionDirection) -> float:
        """Calculate PnL for a trade"""
        if direction == PositionDirection.LONG:
            return quantity * (exit_price - entry_price)
        else:
            return quantity * (entry_price - exit_price)
    
    def update_market_price(self, symbol: str, price: float) -> Optional[Position]:
        """
        Update position with current market price for unrealized PnL calculation
        
        Args:
            symbol: Trading symbol
            price: Current market price
            
        Returns:
            Updated Position if exists, None otherwise
        """
        with self._lock:
            if symbol not in self.positions:
                return None
                
            position = self.positions[symbol]
            position.current_value = position.quantity * price
            position.unrealized_pnl = self._calculate_pnl(
                position.quantity,
                position.avg_price,
                price,
                position.direction
            )
            position.updated_at = datetime.utcnow()
            
            return position
    
    def get(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position if exists, None otherwise
        """
        with self._lock:
            return self.positions.get(symbol, None)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        with self._lock:
            return self.positions.copy()
    
    def get_closed_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get closed positions for a symbol or all symbols
        
        Args:
            symbol: Optional trading symbol filter
            
        Returns:
            List of closed positions
        """
        with self._lock:
            if symbol:
                return self.closed_positions.get(symbol, [])
            return [pos for sym_pos in self.closed_positions.values() for pos in sym_pos]
    
    def get_position_history(self, symbol: str) -> List[Position]:
        """
        Get complete position history for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of all positions (open and closed) for the symbol
        """
        with self._lock:
            history = []
            if symbol in self.positions:
                history.append(self.positions[symbol])
            if symbol in self.closed_positions:
                history.extend(self.closed_positions[symbol])
            return history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current position statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['win_rate'] = (
                stats['winning_positions'] / max(1, stats['closed_positions'])
                if stats['closed_positions'] > 0 else 0.0
            )
            stats['avg_pnl'] = (
                stats['total_pnl'] / max(1, stats['closed_positions'])
                if stats['closed_positions'] > 0 else 0.0
            )
            return stats
    
    def get_performance_report(self) -> pd.DataFrame:
        """Generate comprehensive performance report"""
        with self._lock:
            data = []
            for symbol, positions in self.closed_positions.items():
                for pos in positions:
                    holding_period = (pos.updated_at - pos.created_at).total_seconds() / 86400
                    data.append({
                        'symbol': symbol,
                        'direction': pos.direction.name,
                        'quantity': pos.quantity,
                        'entry_price': pos.avg_price,
                        'exit_price': pos.current_value / pos.quantity,
                        'pnl': pos.realized_pnl,
                        'holding_period': holding_period,
                        'status': pos.status.name,
                        'strategy': pos.trades[0].strategy if pos.trades else None,
                        'created_at': pos.created_at,
                        'closed_at': pos.updated_at
                    })
            
            return pd.DataFrame(data)
    
    def reset(self) -> None:
        """Reset all positions and history"""
        with self._lock:
            self.positions.clear()
            self.closed_positions.clear()
            self.trade_history.clear()
            self._stats = {
                'total_positions': 0,
                'open_positions': 0,
                'closed_positions': 0,
                'total_trades': 0,
                'total_pnl': 0.0,
                'winning_positions': 0,
                'losing_positions': 0,
                'long_positions': 0,
                'short_positions': 0
            }
