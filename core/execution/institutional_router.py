from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import uuid
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from core.execution.base_executor import (
    BaseExecutor,
    ExecutionResult,
    OrderStatus,
    OrderType,
    TimeInForce,
)

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    AGGRESSIVE = auto()        # Immediate execution at best available price
    PASSIVE = auto()           # Patient execution seeking price improvement
    DARK_POOL = auto()         # Seek block liquidity in dark pools
    LIQUIDITY_SEEKING = auto() # Actively search for hidden liquidity
    ICEBERG = auto()           # Break into smaller hidden orders
    TWAP = auto()              # Time-weighted average price execution
    VWAP = auto()              # Volume-weighted average price execution

class VenueType(Enum):
    LIT_EXCHANGE = auto()      # Public order book exchange
    DARK_POOL = auto()         # Private trading venues
    ECN = auto()               # Electronic communication network
    ATS = auto()               # Alternative trading system

@dataclass
class ExecutionVenue:
    name: str
    venue_type: VenueType
    fee_structure: Dict[str, float]  # Maker/taker fees, rebates
    latency_ms: float                # Round-trip latency
    fill_probability: float          # Historical fill rate
    max_order_size: float            # Maximum order size accepted
    min_order_size: float = 0.0      # Minimum order size
    avg_slippage: float = 0.0        # Historical average slippage
    volume_share: float = 0.0        # Market share for this instrument
    uptime_pct: float = 99.9         # Historical uptime percentage

@dataclass
class OrderRoutingPlan:
    order_id: str
    primary_venue: ExecutionVenue
    contingency_venues: List[Tuple[ExecutionVenue, float]]  # Venue + allocation %
    routing_strategy: RoutingStrategy
    expected_slippage: float
    expected_fees: float
    score: float                     # Quality score for this routing plan
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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
    - Contingency failover with real-time monitoring
    - Adaptive routing strategies
    - Transaction cost analysis (TCA)
    - Performance tracking and analytics
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.venues = self._init_venues()
        self.venue_performance = defaultdict(VenuePerformance)
        self.strategy_rules = self._init_strategy_rules()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.historical_orders = pd.DataFrame(
            columns=['order_id', 'venue', 'strategy', 'quantity', 'slippage', 'fees', 'timestamp']
        )

    def _init_venues(self) -> Dict[str, ExecutionVenue]:
        """Initialize supported execution venues with detailed parameters"""
        return {
            "ALPACA_PRIMARY": ExecutionVenue(
                name="ALPACA_PRIMARY",
                venue_type=VenueType.LIT_EXCHANGE,
                fee_structure={"maker": 0.0002, "taker": 0.0005, "rebate": 0.0001},
                latency_ms=25,
                fill_probability=0.98,
                max_order_size=15_000,
                avg_slippage=0.0005,
                volume_share=0.35
            ),
            "ALPACA_DARK": ExecutionVenue(
                name="ALPACA_DARK",
                venue_type=VenueType.DARK_POOL,
                fee_structure={"maker": 0.0001, "taker": 0.0003, "rebate": 0.00005},
                latency_ms=45,
                fill_probability=0.82,
                max_order_size=8_000,
                avg_slippage=0.0008,
                volume_share=0.15
            ),
            "ECN_PRO": ExecutionVenue(
                name="ECN_PRO",
                venue_type=VenueType.ECN,
                fee_structure={"maker": 0.00015, "taker": 0.0004, "rebate": 0.00012},
                latency_ms=35,
                fill_probability=0.95,
                max_order_size=12_000,
                avg_slippage=0.0006,
                volume_share=0.25
            )
        }

    def _init_strategy_rules(self) -> Dict[RoutingStrategy, Dict]:
        """Define rules for each routing strategy"""
        return {
            RoutingStrategy.AGGRESSIVE: {
                'max_slippage': 0.001,
                'timeout': timedelta(seconds=5),
                'venue_priority': ['ALPACA_PRIMARY', 'ECN_PRO']
            },
            RoutingStrategy.PASSIVE: {
                'max_slippage': 0.0002,
                'timeout': timedelta(minutes=5),
                'venue_priority': ['ECN_PRO', 'ALPACA_DARK']
            },
            RoutingStrategy.DARK_POOL: {
                'max_slippage': 0.0005,
                'timeout': timedelta(minutes=10),
                'venue_priority': ['ALPACA_DARK']
            },
            RoutingStrategy.VWAP: {
                'max_slippage': 0.0003,
                'timeout': timedelta(minutes=30),
                'venue_priority': ['ALPACA_PRIMARY', 'ECN_PRO']
            }
        }

    def execute_order(self, order: Dict) -> ExecutionResult:
        """
        Execute order with smart routing and real-time monitoring
        
        Args:
            order: Dictionary containing:
                - symbol: Instrument identifier
                - quantity: Order quantity
                - order_type: LIMIT/MARKET etc.
                - time_in_force: GTC/IOC/FOK etc.
                - price: Optional for limit orders
                - urgency: 0-1 scale of execution urgency
                - strategy: Optional explicit routing strategy
                
        Returns:
            ExecutionResult with detailed performance metrics
        """
        order_id = self._generate_order_id()
        routing_plan = self._create_routing_plan(order, order_id)
        
        try:
            # Execute across multiple venues concurrently
            results = self._execute_across_venues(order, routing_plan)
            combined_result = self._combine_results(results, order_id)
            
            # Record execution for performance analysis
            self._record_execution(combined_result, routing_plan)
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

    def _execute_across_venues(self, order: Dict, routing_plan: OrderRoutingPlan) -> List[ExecutionResult]:
        """Execute order across primary and contingency venues"""
        futures = []
        order_parts = self._split_order(order['quantity'], routing_plan)
        
        # Submit all order parts concurrently
        for venue, quantity in order_parts:
            futures.append(
                self.executor.submit(
                    self._execute_at_venue,
                    order.copy(),
                    quantity,
                    venue
                )
            )
        
        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.warning(f"Venue execution failed: {str(e)}")
                
        return results

    def _split_order(self, total_qty: float, routing_plan: OrderRoutingPlan) -> List[Tuple[ExecutionVenue, float]]:
        """Split order according to routing plan allocations"""
        parts = [(routing_plan.primary_venue, total_qty * 0.7)]  # 70% to primary
        
        for venue, allocation in routing_plan.contingency_venues:
            parts.append((venue, total_qty * allocation))
            
        return parts

    def _execute_at_venue(self, order: Dict, quantity: float, venue: ExecutionVenue) -> ExecutionResult:
        """Simulate order execution at a specific venue"""
        # In production, this would make actual API calls to the venue
        fill_prob = venue.fill_probability * (1 - min(1, quantity / venue.max_order_size))
        filled = quantity if np.random.random() < fill_prob else quantity * 0.8
        
        slippage = venue.avg_slippage * (1 + np.random.normal(0, 0.2))
        fees = self.calculate_fees(order, venue, filled)
        
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

    def _create_routing_plan(self, order: Dict, order_id: str) -> OrderRoutingPlan:
        """Create optimized routing plan based on order characteristics"""
        strategy = self._select_strategy(order)
        venues = self._select_venues(strategy, order['quantity'])
        expected_slippage = self._estimate_slippage(strategy, order['quantity'])
        fees = self._estimate_fees(order, venues[0])
        
        return OrderRoutingPlan(
            order_id=order_id,
            primary_venue=venues[0],
            contingency_venues=[(v, 0.1) for v in venues[1:]],
            routing_strategy=strategy,
            expected_slippage=expected_slippage,
            expected_fees=fees,
            score=self._score_routing_plan(strategy, venues, order['quantity'])
        )

    def _select_strategy(self, order: Dict) -> RoutingStrategy:
        """Determine optimal routing strategy based on order parameters"""
        if 'strategy' in order:
            return order['strategy']
            
        urgency = order.get('urgency', 0.5)
        quantity = order['quantity']
        
        if urgency > 0.8:
            return RoutingStrategy.AGGRESSIVE
        elif quantity > 10000:
            return RoutingStrategy.DARK_POOL
        elif quantity > 5000 and urgency < 0.3:
            return RoutingStrategy.VWAP
        return RoutingStrategy.PASSIVE

    def _select_venues(self, strategy: RoutingStrategy, quantity: float) -> List[ExecutionVenue]:
        """Select venues based on strategy and order size"""
        venue_names = self.strategy_rules[strategy]['venue_priority']
        return [self.venues[name] for name in venue_names 
                if self.venues[name].max_order_size >= quantity * 0.5]

    def _estimate_slippage(self, strategy: RoutingStrategy, quantity: float) -> float:
        """Predict expected slippage based on historical data"""
        base_slippage = {
            RoutingStrategy.AGGRESSIVE: 0.0008,
            RoutingStrategy.PASSIVE: 0.0002,
            RoutingStrategy.DARK_POOL: 0.0005,
            RoutingStrategy.VWAP: 0.0003
        }.get(strategy, 0.0005)
        
        # Scale by order size relative to typical market depth
        size_factor = min(1.0, quantity / 5000)
        return base_slippage * (1 + size_factor)

    def _score_routing_plan(self, strategy: RoutingStrategy, venues: List[ExecutionVenue], quantity: float) -> float:
        """Score routing plan quality (0-1)"""
        fill_score = sum(v.fill_probability * 0.5 ** i for i, v in enumerate(venues))
        fee_score = 1 - min(0.001, venues[0].fee_structure['taker'])
        size_score = min(1.0, quantity / venues[0].max_order_size)
        
        return (fill_score * 0.5 + fee_score * 0.3 + size_score * 0.2)

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

    def _record_execution(self, result: ExecutionResult, plan: OrderRoutingPlan):
        """Record execution details for performance analysis"""
        new_row = {
            'order_id': result.order_id,
            'venue': plan.primary_venue.name,
            'strategy': plan.routing_strategy.name,
            'quantity': result.filled_quantity,
            'slippage': result.slippage,
            'fees': result.fees,
            'timestamp': result.execution_time
        }
        self.historical_orders.loc[len(self.historical_orders)] = new_row

    def _update_venue_performance(self, result: ExecutionResult):
        """Update venue performance metrics"""
        for venue_name in result.metadata.get('venues', []):
            perf = self.venue_performance[venue_name]
            perf.total_volume += result.filled_quantity
            perf.avg_slippage = (perf.avg_slippage * perf.total_volume + 
                                result.slippage * result.filled_quantity) / perf.total_volume
            perf.last_used = datetime.now(timezone.utc)
            
            # Simple fill rate tracking (would be more sophisticated in production)
            if result.status == OrderStatus.FILLED:
                perf.fill_rate = (perf.fill_rate * (perf.total_volume - result.filled_quantity) + 
                                1 * result.filled_quantity) / perf.total_volume

    def calculate_fees(self, order: Dict, venue: ExecutionVenue, quantity: float) -> float:
        """Calculate estimated fees for this execution"""
        is_maker = order.get('order_type', OrderType.MARKET) == OrderType.LIMIT
        fee_pct = venue.fee_structure['maker'] if is_maker else venue.fee_structure['taker']
        return quantity * fee_pct

    def get_performance_report(self) -> Dict:
        """Generate performance analytics report"""
        if self.historical_orders.empty:
            return {}
            
        recent = self.historical_orders[self.historical_orders['timestamp'] > 
                                       datetime.now(timezone.utc) - timedelta(days=1)]
        
        return {
            'total_orders': len(self.historical_orders),
            'avg_slippage': self.historical_orders['slippage'].mean(),
            'avg_fees': self.historical_orders['fees'].mean(),
            'venue_performance': {
                venue: {
                    'fill_rate': perf.fill_rate,
                    'avg_slippage': perf.avg_slippage,
                    'total_volume': perf.total_volume
                }
                for venue, perf in self.venue_performance.items()
            },
            'recent_stats': {
                'order_count': len(recent),
                'fill_rate': len(recent[recent['quantity'] > 0]) / len(recent) if len(recent) > 0 else 0
            }
        }

    def _do_cancel_order(self, order_id: str):
        """Cancel an active order across all venues"""
        if order_id in self._active_orders:
            # In production, would send cancel requests to each venue
            logger.info(f"Cancelling order {order_id} across all venues")
            self._active_orders.pop(order_id)
            
            # Record cancellation in history
            if order_id in self.historical_orders['order_id'].values:
                idx = self.historical_orders[self.historical_orders['order_id'] == order_id].index
                self.historical_orders.loc[idx, 'status'] = OrderStatus.CANCELLED
