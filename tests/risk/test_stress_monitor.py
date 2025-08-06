import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from core.market.stress_monitor import MarketStressMonitor, StressState

## ----------------------------
## Advanced Fixtures
## ----------------------------

@pytest.fixture(params=[
    {'asset_class': 'equities', 'expected_thresholds': (25, 40)},
    {'asset_class': 'crypto', 'expected_thresholds': (35, 60)},
    {'asset_class': 'fx', 'expected_thresholds': (15, 30)}
], ids=['equities', 'crypto', 'fx'])
def configured_monitor(request):
    """Monitor fixture with different asset class configurations"""
    return MarketStressMonitor(
        asset_class=request.param['asset_class'],
        vix_thresholds=request.param['expected_thresholds']
    )

@pytest.fixture
def regime_data():
    """Generate market regime test data"""
    return {
        'normal': {'vix': 15, 'correlation': 0.3, 'liquidity': 0.8},
        'elevated': {'vix': 30, 'correlation': 0.5, 'liquidity': 0.6},
        'stress': {'vix': 45, 'correlation': 0.8, 'liquidity': 0.3},
        'extreme': {'vix': 60, 'correlation': 0.9, 'liquidity': 0.1}
    }

## ----------------------------
## Core Functionality Tests
## ----------------------------

class TestStressCalculations:
    """Tests for stress index calculations"""
    
    def test_vix_weighting_nonlinearity(self, configured_monitor, regime_data):
        """Verify VIX has disproportionate weighting in stress index"""
        normal_score = configured_monitor.calculate_stress_index(regime_data['normal'])
        stress_score = configured_monitor.calculate_stress_index(regime_data['stress'])
        
        vix_diff = regime_data['stress']['vix'] - regime_data['normal']['vix']
        score_diff = stress_score - normal_score
        
        # Verify nonlinear response - stress score increases faster than VIX
        assert score_diff / vix_diff > 1.0
        
    def test_liquidity_crisis_detection(self, configured_monitor):
        """Test liquidity crunch detection even with moderate VIX"""
        data = {
            'vix': 25,  # Moderate
            'correlation': 0.7,  # Elevated
            'liquidity': 0.2  # Crisis
        }
        score = configured_monitor.calculate_stress_index(data)
        assert score > configured_monitor.stress_threshold
        
    @pytest.mark.parametrize("correlation,expected_impact", [
        (0.3, 0.2),  # Low correlation - minimal impact
        (0.6, 0.5),  # Moderate - noticeable
        (0.9, 0.8)   # High - major impact
    ])
    def test_correlation_impact(self, configured_monitor, correlation, expected_impact):
        """Test cross-asset correlation impact on stress score"""
        base_data = {'vix': 30, 'liquidity': 0.5}
        base_score = configured_monitor.calculate_stress_index({**base_data, 'correlation': 0.3})
        
        test_score = configured_monitor.calculate_stress_index({**base_data, 'correlation': correlation})
        impact = (test_score - base_score) / base_score
        
        assert impact == pytest.approx(expected_impact, abs=0.1)

## ----------------------------
## State Transition Tests
## ----------------------------

class TestStateTransitions:
    """Tests for market state machine behavior"""
    
    def test_hysteresis_effect(self, configured_monitor, regime_data):
        """Verify state transitions require meaningful changes"""
        # Transition to stress
        configured_monitor.update_signals(regime_data['stress'])
        assert configured_monitor.current_state == StressState.STRESS
        
        # Small improvement shouldn't exit stress
        slightly_better = {
            'vix': regime_data['stress']['vix'] * 0.95,
            'correlation': regime_data['stress']['correlation'] * 0.95,
            'liquidity': regime_data['stress']['liquidity'] * 1.05
        }
        configured_monitor.update_signals(slightly_better)
        assert configured_monitor.current_state == StressState.STRESS
        
        # Requires significant improvement to normalize
        configured_monitor.update_signals(regime_data['normal'])
        assert configured_monitor.current_state == StressState.NORMAL
        
    def test_rapid_escalation(self, configured_monitor, regime_data):
        """Test fast transition to extreme state"""
        # From normal directly to extreme
        configured_monitor.update_signals(regime_data['normal'])
        configured_monitor.update_signals(regime_data['extreme'])
        
        assert configured_monitor.current_state == StressState.EXTREME
        assert configured_monitor.state_duration < 2  # Should happen immediately
        
    @pytest.mark.parametrize("improvement_rate,expected_state", [
        ('slow', StressState.ELEVATED),  # Gradual improvement
        ('fast', StressState.NORMAL)     # Rapid recovery
    ])
    def test_recovery_patterns(self, configured_monitor, improvement_rate, expected_state):
        """Test different market recovery scenarios"""
        # Start in stress
        configured_monitor.update_signals({
            'vix': 45,
            'correlation': 0.8,
            'liquidity': 0.3
        })
        
        if improvement_rate == 'slow':
            # Gradual improvement over 3 updates
            for i in range(3):
                configured_monitor.update_signals({
                    'vix': 45 - (i+1)*5,
                    'correlation': 0.8 - (i+1)*0.1,
                    'liquidity': 0.3 + (i+1)*0.1
                })
        else:
            # Immediate improvement
            configured_monitor.update_signals({
                'vix': 20,
                'correlation': 0.4,
                'liquidity': 0.7
            })
            
        assert configured_monitor.current_state == expected_state

## ----------------------------
## Asset-Class Specific Tests
## ----------------------------

class TestAssetClassBehavior:
    """Tests for asset-class specific behaviors"""
    
    def test_crypto_volatility_scaling(self):
        """Verify crypto markets have higher volatility tolerance"""
        crypto_monitor = MarketStressMonitor('crypto')
        equities_monitor = MarketStressMonitor('equities')
        
        # Same market conditions
        data = {'vix': 35, 'correlation': 0.6, 'liquidity': 0.5}
        
        crypto_score = crypto_monitor.calculate_stress_index(data)
        equities_score = equities_monitor.calculate_stress_index(data)
        
        # Crypto should score this as less stressful
        assert crypto_score < equities_score
        
    def test_fx_liquidity_sensitivity(self):
        """Verify FX markets are more sensitive to liquidity"""
        fx_monitor = MarketStressMonitor('fx')
        equities_monitor = MarketStressMonitor('equities')
        
        # Low liquidity scenario
        data = {'vix': 25, 'correlation': 0.5, 'liquidity': 0.3}
        
        fx_score = fx_monitor.calculate_stress_index(data)
        equities_score = equities_monitor.calculate_stress_index(data)
        
        # FX should score this as more stressful
        assert fx_score > equities_score

## ----------------------------
## Alerting & Actions Tests
## ----------------------------

class TestStressResponses:
    """Tests for stress-triggered actions"""
    
    @pytest.mark.parametrize("state,expected_actions", [
        (StressState.NORMAL, []),
        (StressState.ELEVATED, ['increase_monitoring']),
        (StressState.STRESS, ['reduce_risk', 'enable_circuit_breakers']),
        (StressState.EXTREME, ['halt_trading', 'activate_emergency_protocols'])
    ])
    def test_state_actions(self, configured_monitor, state, expected_actions):
        """Verify correct actions for each stress state"""
        with patch.object(configured_monitor, 'trigger_actions') as mock_trigger:
            configured_monitor.current_state = state
            configured_monitor.execute_state_actions()
            
            assert mock_trigger.call_count == 1
            triggered_actions = mock_trigger.call_args[0][0]
            assert set(expected_actions).issubset(triggered_actions)
            
    def test_volatility_spike_alert(self, configured_monitor):
        """Test alerts for rapid volatility increases"""
        with patch.object(configured_monitor, 'send_alert') as mock_alert:
            # Normal market
            configured_monitor.update_signals({'vix': 20, 'correlation': 0.4, 'liquidity': 0.7})
            
            # Sudden spike
            configured_monitor.update_signals({'vix': 45, 'correlation': 0.5, 'liquidity': 0.6})
            
            assert mock_alert.called
            assert "volatility spike" in mock_alert.call_args[0][0]['message']
            assert mock_alert.call_args[0][0]['severity'] == 'high'

## ----------------------------
## Performance Tests
## ----------------------------

class TestMonitorPerformance:
    """Performance and scaling tests"""
    
    @pytest.mark.stress
    def test_high_frequency_updates(self, configured_monitor):
        """Test monitor handles high-frequency signal updates"""
        import time
        
        start = time.perf_counter()
        for i in range(1000):
            configured_monitor.update_signals({
                'vix': 20 + (i % 30),
                'correlation': 0.3 + (i % 50)/100,
                'liquidity': 0.5 + (i % 20)/100
            })
            
        duration = time.perf_counter() - start
        assert duration < 0.5  # 1000 updates in <500ms
        assert configured_monitor.signal_history.maxlen == 1000
        
    def test_memory_efficiency(self, configured_monitor):
        """Verify memory usage doesn't grow with history"""
        import tracemalloc
        
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
        # Process updates
        for i in range(1000):
            configured_monitor.update_signals({
                'vix': 20 + (i % 30),
                'correlation': 0.3 + (i % 50)/100,
                'liquidity': 0.5 + (i % 20)/100
            })
            
        snapshot2 = tracemalloc.take_snapshot()
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        mem_increase = sum(stat.size_diff for stat in stats)
        assert mem_increase < 1024 * 1024  # <1MB increase
        
        tracemalloc.stop()

## ----------------------------
## Integration Tests
## ----------------------------

@pytest.mark.integration
class TestIntegration:
    """Integration with market data systems"""
    
    @pytest.fixture
    def live_data_adapter(self):
        """Mock live data feed adapter"""
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, configured_monitor, live_data_adapter):
        """Test end-to-end monitoring with live data"""
        with patch.object(configured_monitor, 'update_signals') as mock_update:
            # Simulate live data feed
            live_data_adapter.subscribe.return_value = [
                {'vix': 20, 'correlation': 0.4, 'liquidity': 0.7},
                {'vix': 25, 'correlation': 0.5, 'liquidity': 0.6},
                {'vix': 35, 'correlation': 0.7, 'liquidity': 0.4}
            ]
            
            # Process updates
            for data in live_data_adapter.subscribe():
                configured_monitor.update_signals(data)
                
            assert mock_update.call_count == 3
            assert configured_monitor.current_state != StressState.NORMAL
