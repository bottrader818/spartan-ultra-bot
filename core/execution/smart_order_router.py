from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    AGGRESSIVE = auto()        # Immediate execution at best available price
    PASSIVE = auto()           # Patient execution seeking price improvement
    DARK_POOL = auto()         # Seek block liquidity in dark pools
    LIQUIDITY_SEEKING = auto() # Actively search for hidden liquidity
    ICEBERG = auto()           # Break into smaller hidden orders
    TWAP = auto()              # Time-weighted average price execution
    VWAP = auto()             # Volume-weighted average price execution

@dataclass
class Venue:
    name: str
    latency_ms: int
    fee_structure: Dict[str, float]  # maker/taker fees
    fill_rate: float
    max_order_size: float
    min_order_size: float = 0.0
    avg_slippage: float = 0.0005
    uptime_pct: float = 99.9
    venue_type: str = "LIT"  # LIT/DARK/ECN

@dataclass
class VenuePerformance:
    fill_rate: float = 0.0
    avg_slippage: float = 0.0
    avg_fees: float = 0.0
    total_volume: float = 0.0
    last_used: Optional[datetime] = None

class SmartOrderRouter(BaseExecutor):
    """
    Advanced Smart Order Router with:
    - Multi-venue execution optimization
    - Dynamic venue scoring and selection
    - Contingency routing
    - Transaction cost analysis
    - Performance tracking
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.venues = self._init_venues()
        self.venue_performance = defaultdict(VenuePerformance)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.historical_executions = []

    def _init_venues(self) -> Dict[str, Venue]:
        """Initialize supported execution venues"""
        return {
            "ALPACA_PRIMARY": Venue(
                name="ALPACA_PRIMARY",
                latency_ms=25,
                fee_structure={"maker": 0.0002, "taker": 0.0005},
                fill_rate=0.98,
                max_order_size=15_000,
                venue_type="LIT"
            ),
            "ALPACA_DARK": Venue(
                name="ALPACA_DARK",
                latency_ms=45,
                fee_structure={"maker": 0.0001, "taker": 0.0003},
                fill_rate=0.85,
                max_order_size=8_000,
                venue_type="DARK"
            ),
            "ECN_PRO": Venue(
                name="ECN_PRO",
                latency_ms=35,
                fee_structure={"maker": 0.00015, "taker": 0.0004},
                fill_rate=0.95,
                max_order_size=12_000,
                venue_type="ECN"
            )
        }

    def execute_order(self, order: Dict) -> ExecutionResult:
        """
        Execute order with smart routing
        
        Args:
            order: Dictionary containing:
                - symbol: Instrument identifier
                - quantity: Order quantity
                - order_type: LIMIT/MARKET etc.
                - time_in_force: GTC/IOC/FOK etc.
                - price: Optional for limit orders
                - urgency: 0-1 scale of execution urgency
                
        Returns:
            ExecutionResult with detailed performance metrics
        """
        order_id = self._generate_order_id()
        strategy = self._determine_strategy(order)
        venues = self._select_venues(order, strategy)
        
        try:
            # Execute across multiple venues
            results = self._execute_across_venues(order, venues)
            combined_result = self._combine_results(results, order_id)
            
            # Record execution for performance analysis
            self._record_execution(combined_result)
            self._update_venue_performance(combined_result)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Order {order_id} failed: {str(e)}")
            return ExecutionResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                remaining_quantity=order['quantity'],
                avg_execution_price=0,
                execution_time=datetime.now(timezone.utc),
                fees=0,
                slippage=0,
                metadata={"error": str(e)}
            )

    def _determine_strategy(self, order: Dict) -> RoutingStrategy:
        """Determine optimal routing strategy based on order parameters"""
        urgency = order.get('urgency', 0.5)
        quantity = order['quantity']
        
        if urgency > 0.8:
            return RoutingStrategy.AGGRESSIVE
        elif quantity > 10000:
            return RoutingStrategy.DARK_POOL
        elif quantity > 5000 and urgency < 0.3:
            return RoutingStrategy.VWAP
        return RoutingStrategy.PASSIVE

    def _select_venues(self, order: Dict, strategy: RoutingStrategy) -> List[Venue]:
        """Select venues based on order characteristics and strategy"""
        quantity = order['quantity']
        
        if strategy == RoutingStrategy.AGGRESSIVE:
            return [self.venues["ALPACA_PRIMARY"], self.venues["ECN_PRO"]]
        elif strategy == RoutingStrategy.DARK_POOL:
            return [self.venues["ALPACA_DARK"], self.venues["ECN_PRO"]]
        else:
            return [self.venues["ECN_PRO"], self.venues["ALPACA_PRIMARY"]]

    def _execute_across_venues(self, order: Dict, venues: List[Venue]) -> List[ExecutionResult]:
        """Execute order parts across selected venues"""
        order_parts = self._split_order(order['quantity'], venues)
        futures = []
        
        for venue, quantity in order_parts:
            futures.append(
                self.executor.submit(
                    self._execute_at_venue,
                    order.copy(),
                    quantity,
                    venue
                )
            )
        
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.warning(f"Venue execution failed: {str(e)}")
                
        return results

    def _split_order(self, quantity: float, venues: List[Venue]) -> List[Tuple[Venue, float]]:
        """Split order across venues based on their characteristics"""
        parts = []
        remaining = quantity
        
        # Allocate 70% to primary venue
        primary_qty = min(quantity * 0.7, venues[0].max_order_size)
        parts.append((venues[0], primary_qty))
        remaining -= primary_qty
        
        # Allocate remaining to secondary venues
        for venue in venues[1:]:
            if remaining <= 0:
                break
            venue_qty = min(remaining, venue.max_order_size)
            parts.append((venue, venue_qty))
            remaining -= venue_qty
            
        return parts

    def _execute_at_venue(self, order: Dict, quantity: float, venue: Venue) -> ExecutionResult:
        """Simulate order execution at a specific venue"""
        # Simulate fill probability based on venue characteristics
        fill_prob = venue.fill_rate * (1 - min(1, quantity / venue.max_order_size))
        filled = quantity if np.random.random() < fill_prob else quantity * 0.8
        
        # Calculate slippage and fees
        slippage = venue.avg_slippage * (1 + np.random.normal(0, 0.2))
        is_maker = order.get('order_type', OrderType.MARKET) == OrderType.LIMIT
        fees = self.calculate_fees(order, venue, is_maker, filled)
        
        return ExecutionResult(
            order_id=order.get('client_order_id', str(uuid.uuid4())),
            status=OrderStatus.FILLED if filled == quantity else OrderStatus.PARTIALLY_FILLED,
            filled_quantity=filled,
            remaining_quantity=quantity - filled,
            avg_execution_price=order.get('price', 100.0) * (1 + slippage),
            execution_time=datetime.now(timezone.utc),
            fees=fees,
            slippage=slippage,
            metadata={"venue": venue.name}
        )

    def _combine_results(self, results: List[ExecutionResult], order_id: str) -> ExecutionResult:
        """Combine partial fills from multiple venues"""
        total_filled = sum(r.filled_quantity for r in results)
        total_fees = sum(r.fees for r in results)
        avg_price = sum(r.avg_execution_price * r.filled_quantity for r in results) / total_filled
        
        return ExecutionResult(
            order_id=order_id,
            status=OrderStatus.FILLED if total_filled > 0 else OrderStatus.REJECTED,
            filled_quantity=total_filled,
            remaining_quantity=results[0].remaining_quantity,  # Original quantity - filled
            avg_execution_price=avg_price,
            execution_time=datetime.now(timezone.utc),
            fees=total_fees,
            slippage=(avg_price / results[0].avg_execution_price - 1),
            metadata={"venues": [r.metadata['venue'] for r in results]}
        )

    def calculate_fees(self, order: Dict, venue: Venue, is_maker: bool, quantity: float) -> float:
        """Calculate estimated fees for this execution"""
        fee_pct = venue.fee_structure['maker'] if is_maker else venue.fee_structure['taker']
        return quantity * fee_pct

    def _record_execution(self, result: ExecutionResult):
        """Record execution details for performance analysis"""
        self.historical_executions.append({
            'order_id': result.order_id,
            'timestamp': result.execution_time,
            'quantity': result.filled_quantity,
            'slippage': result.slippage,
            'fees': result.fees,
            'venues': result.metadata.get('venues', [])
        })

    def _update_venue_performance(self, result: ExecutionResult):
        """Update venue performance metrics"""
        for venue_name in result.metadata.get('venues', []):
            perf = self.venue_performance[venue_name]
            perf.total_volume += result.filled_quantity
            perf.avg_slippage = (perf.avg_slippage * perf.total_volume + 
                                result.slippage * result.filled_quantity) / perf.total_volume
            perf.last_used = datetime.now(timezone.utc)
            
            if result.status == OrderStatus.FILLED:
                perf.fill_rate = (perf.fill_rate * (perf.total_volume - result.filled_quantity) + result.filled_quantity) / perf.total_volume

    def get_performance_report(self) -> Dict:
        """Generate performance analytics report"""
        if not self.historical_executions:
            return {}
            
        return {
            'total_orders': len(self.historical_executions),
            'avg_slippage': np.mean([e['slippage'] for e in self.historical_executions]),
            'avg_fees': np.mean([e['fees'] for e in self.historical_executions]),
            'venue_performance': {
                venue: {
                    'fill_rate': perf.fill_rate,
                    'avg_slippage': perf.avg_slippage,
                    'total_volume': perf.total_volume
                }
                for venue, perf in self.venue_performance.items()
            }
        }

    def _do_cancel_order(self, order_id: str):
        """Cancel an active order across all venues"""
        if order_id in self._active_orders:
            logger.info(f"Cancelling order {order_id} across all venues")
            self._active_orders.pop(order_id)
            
            # Record cancellation in history
            for execution in self.historical_executions:
                if execution['order_id'] == order_id:
                    execution['status'] = "CANCELLED"
