from core.execution.routers.adaptive_router import AdaptiveRouter
from core.execution.venues.venue_base import ExecutionVenue
import pytest
from dataclasses import dataclass

@dataclass
class MockOrder:
    """Production-like order object"""
    symbol: str
    quantity: int
    order_type: str = 'LIMIT'
    price: float = None

class TestVenue(ExecutionVenue):
    """Enhanced mock venue with realistic properties"""
    def __init__(self, name, fill_rate=0.95, fee=0.0005):
        self.name = name
        self._fill_rate = fill_rate
        self._available = True
        self._fee = fee
        self._latency_ms = 50
    
    def is_available(self):
        return self._available
    
    def get_latency(self):
        return self._latency_ms
    
    def calculate_fee(self, order):
        return self._fee * order.quantity

@pytest.fixture
def default_router():
    """Production-equivalent router fixture"""
    return AdaptiveRouter(
        venues=[
            TestVenue("ALPACA", fill_rate=0.98, fee=0.0003),  # Best venue
            TestVenue("SIM", fill_rate=0.95, fee=0.0002),     # Low fee
            TestVenue("DARK", fill_rate=0.85, fee=0.0001)     # Fallback
        ],
        config={'max_latency_ms': 100}
    )

class TestAdaptiveRouter:
    """Comprehensive test suite following production patterns"""

    def test_primary_routing(self, default_router):
        """Verify optimal venue selection"""
        order = MockOrder(symbol='SPY', quantity=100)
        plan = default_router.route_order(order)
        assert plan.primary.name == "ALPACA"
        assert plan.primary_reason == "best_fill_rate"

    def test_fallback_mechanism(self, default_router):
        """Test circuit breaker functionality"""
        default_router.venues[0]._available = False  # Disable ALPACA
        order = MockOrder(symbol='SPY', quantity=100)
        plan = default_router.route_order(order)
        assert plan.primary.name == "SIM"
        assert plan.fallback_reason == "primary_unavailable"

    def test_latency_threshold(self, default_router):
        """Verify latency constraints"""
        default_router.venues[0]._latency_ms = 150  # Exceeds threshold
        order = MockOrder(symbol='SPY', quantity=100)
        plan = default_router.route_order(order)
        assert plan.primary.name == "SIM"
        assert plan.primary_reason == "within_latency"

    def test_order_allocation(self, default_router):
        """Test large order splitting logic"""
        large_order = MockOrder(symbol='SPY', quantity=10000)
        plan = default_router.route_order(large_order)
        assert len(plan.allocations) == 2
        assert sum(a.quantity for a in plan.allocations) == 10000

    def test_cost_optimization(self, default_router):
        """Verify fee-sensitive routing"""
        low_cost_order = MockOrder(symbol='SPY', quantity=5000)
        default_router.config['cost_sensitive'] = True
        plan = default_router.route_order(low_cost_order)
        assert plan.primary.name == "DARK"
        assert plan.primary_reason == "lowest_cost"

    @pytest.mark.parametrize("qty,expected_venues", [
        (100, ["ALPACA"]),
        (5000, ["ALPACA", "SIM"]),
        (20000, ["ALPACA", "SIM", "DARK"])
    ])
    def test_quantity_based_routing(self, default_router, qty, expected_venues):
        """Parameterized quantity threshold testing"""
        order = MockOrder(symbol='SPY', quantity=qty)
        plan = default_router.route_order(order)
        assert [v.name for v in plan.allocations] == expected_venues

    def test_venue_health_stats(self, default_router):
        """Verify health monitoring integration"""
        for _ in range(5):
            default_router.route_order(MockOrder(symbol='SPY', quantity=100))
        stats = default_router.get_venue_stats()
        assert stats["ALPACA"]["usage_count"] == 5
        assert stats["DARK"]["usage_count"] == 0
