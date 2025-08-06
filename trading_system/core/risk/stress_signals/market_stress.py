import pytest
import asyncio
import tracemalloc
from decimal import Decimal
from unittest.mock import patch, MagicMock
from core.risk.stress_signals.market_stress import MarketStressMonitor
from enum import Enum, auto

class StressState(Enum):
    NORMAL = auto()
    ELEVATED = auto()
    STRESS = auto()
    EXTREME = auto()

# ----------------------------
# Enterprise-Grade Fixtures
# ----------------------------

@pytest.fixture(params=[
    {'market_type': 'equities', 'vix_threshold': 30},
    {'market_type': 'crypto', 'vix_threshold': 45},
    {'market_type': 'fx', 'vix_threshold': 25}
], ids=['equities', 'crypto', 'fx'])
def config_stress_monitor(request):
    """Monitor with different market configurations"""
    return MarketStressMonitor(
        market_type=request.param['market_type'],
        vix_threshold=request.param['vix_threshold']
    )

@pytest.fixture
def chaos_monitor():
    """Monitor with failure injection capabilities"""
    monitor = MarketStressMonitor()
    monitor._fail_next_update = False
    return monitor

# ----------------------------
# Chaos Engineering Tests  
# ----------------------------

class TestFailureScenarios:
    """Tests for system failure modes"""
    
    @pytest.mark.asyncio
    async def test_signal_processing_failure(self, chaos_monitor):
        """Test corrupted signal handling"""
        with patch.object(chaos_monitor, '_process_signals', 
                         side_effect=Exception("Signal corruption")):
            with pytest.raises(Exception, match="Signal corruption"):
                await chaos_monitor.update_signals_async({'vix': 40})
                
        assert chaos_monitor._state == "degraded"

    def test_missing_data_resilience(self, config_stress_monitor):
        """Test partial signal handling"""
        with patch('core.risk.stress_signals.validate_inputs', 
                 return_value={'vix': 40}):  # Missing other fields
            config_stress_monitor.update_signals({'vix': 40})
            assert config_stress_monitor._state != "error"

# ----------------------------
# Performance Tests
# ----------------------------

class TestStressMonitorPerformance:
    """Benchmarking and load testing"""
    
    @pytest.mark.stress
    def test_high_frequency_updates(self, config_stress_monitor):
        """Process 10K updates under 500ms"""
        start = time.perf_counter()
        for i in range(10_000):
            config_stress_monitor.update_signals({
                'vix': 20 + (i % 30),
                'correlation': 0.3 + (i % 7)/10,
                'liquidity': 0.5 + (i % 3)/10
            })
        duration = time.perf_counter() - start
        assert duration < 0.5
        
    def test_memory_usage(self, config_stress_monitor):
        """Verify <1MB memory growth after 1K updates"""
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
        for _ in range(1000):
            config_stress_monitor.update_signals({
                'vix': 25,
                'correlation': 0.5,
                'liquidity': 0.4
            })
            
        snapshot2 = tracemalloc.take_snapshot()
        assert (snapshot2.compare_to(snapshot1, 'lineno')[0].size_diff < 1_000_000
        tracemalloc.stop()

# ----------------------------
# Integration Layer
# ----------------------------

@pytest.mark.integration
class TestMarketDataIntegration:
    """Live data integration tests"""
    
    @pytest.fixture
    def mock_market_data(self):
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_real_time_feeds(self, config_stress_monitor, mock_market_data):
        """Test end-to-end with mock market feed"""
        mock_market_data.subscribe.return_value = [
            {'vix': 25, 'correlation': 0.5, 'liquidity': 0.4},
            {'vix': 35, 'correlation': 0.8, 'liquidity': 0.2},
            {'vix': 45, 'correlation': 0.9, 'liquidity': 0.1}
        ]
        
        states = []
        config_stress_monitor.add_subscriber(lambda x: states.append(x['state']))
        
        async for data in mock_market_data.subscribe():
            await config_stress_monitor.update_signals_async(data)
            
        assert states == ['normal', 'stress', 'extreme']

# ----------------------------
# Core Tests (Enhanced)
# ----------------------------

class TestStressScenarios:
    """Comprehensive state transition testing"""
    
    @pytest.mark.parametrize("signals,expected_state,alert_severity", [
        (
            {'vix': 25, 'correlation': 0.5, 'liquidity': 0.4},
            StressState.NORMAL,
            None
        ),
        (
            {'vix': 32, 'correlation': 0.7, 'liquidity': 0.3},
            StressState.ELEVATED,
            'warning'
        ),
        (
            {'vix': 40, 'correlation': 0.9, 'liquidity': 0.1},
            StressState.EXTREME,
            'critical'
        )
    ])
    def test_state_machine(self, config_stress_monitor, 
                         signals, expected_state, alert_severity):
        """Verify full state machine behavior"""
        alerts = []
        config_stress_monitor.add_subscriber(lambda x: alerts.append(x))
        
        config_stress_monitor.update_signals(signals)
        
        assert config_stress_monitor._state == expected_state.name.lower()
        if alert_severity:
            assert alerts[0]['severity'] == alert_severity
