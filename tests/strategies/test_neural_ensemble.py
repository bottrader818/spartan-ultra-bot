import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from core.strategies.neural_ensemble import NeuralEnsembleStrategy

## ----------------------------
## Advanced Fixtures
## ----------------------------

@pytest.fixture(params=[
    {'ensemble_type': 'conservative', 'expected_max_vol': 0.3},
    {'ensemble_type': 'aggressive', 'expected_max_vol': 0.5},
    {'ensemble_type': 'market_neutral', 'expected_max_vol': 0.2}
], ids=['conservative', 'aggressive', 'market_neutral'])
def configured_strategy(request):
    """Strategy fixture with different risk profiles"""
    config = {
        'ensemble_config': {
            'type': request.param['ensemble_type'],
            'max_volatility': request.param['expected_max_vol']
        }
    }
    return NeuralEnsembleStrategy(config)

@pytest.fixture
def mock_models():
    """Mock ensemble models with configurable behavior"""
    models = {
        'temporal': MagicMock(),
        'graph': MagicMock(),
        'fundamental': MagicMock()
    }
    return models

## ----------------------------
## Core Functionality Tests
## ----------------------------

class TestEnsembleWeighting:
    """Tests for model weighting and signal generation"""
    
    @pytest.mark.parametrize("uncertainties,expected_ratios", [
        ([(0.1, 0.4), (0.8, 0.6)],  # Temporal vs Graph
        ([(0.05, 0.3), (0.2, 0.9)],  # Temporal vs Fundamental
        ([(0.15, 0.2, 0.4), (0.5, 0.4, 0.1)]  # All three models
    ])
    def test_uncertainty_weighting(self, configured_strategy, mock_models, 
                                 uncertainties, expected_ratios):
        """Verify inverse uncertainty weighting across models"""
        # Configure mock returns
        for model, (std, weight) in zip(mock_models.values(), zip(uncertainties, expected_ratios)):
            model.predict_with_uncertainty.return_value = {
                'mean': 0.5, 
                'std': std
            }
        
        with patch.object(configured_strategy, 'models', mock_models):
            signal = configured_strategy.generate_signal({})
            
            # Verify weighting ratios match expected proportions
            weights = [signal[f'{name}_weight'] for name in mock_models.keys()]
            actual_ratios = [w/max(weights) for w in weights]
            
            assert np.allclose(actual_ratios, expected_ratios, rtol=0.1), \
                f"Weight ratios {actual_ratios} don't match expected {expected_ratios}"

    def test_consensus_boosting(self, configured_strategy):
        """Test confidence boost when models agree"""
        with patch.object(configured_strategy.models['temporal'], 'predict_with_uncertainty') as mock_temp, \
             patch.object(configured_strategy.models['graph'], 'predict_with_uncertainty') as mock_graph:
            
            # Consensus case
            mock_temp.return_value = {'mean': 0.7, 'std': 0.1}
            mock_graph.return_value = {'mean': 0.68, 'std': 0.15}
            consensus_signal = configured_strategy.generate_signal({})
            
            # Divergent case
            mock_temp.return_value = {'mean': 0.7, 'std': 0.1}
            mock_graph.return_value = {'mean': 0.3, 'std': 0.15}
            divergent_signal = configured_strategy.generate_signal({})
            
            assert consensus_signal['confidence'] > divergent_signal['confidence'], \
                "Consensus signals should have higher confidence"

## ----------------------------
## Risk Management Tests
## ----------------------------

class TestRiskOverlays:
    """Tests for volatility and position scaling"""
    
    @pytest.mark.parametrize("vix,expected_reduction", [
        (15, 1.0),    # No reduction
        (25, 0.8),    # Moderate reduction
        (35, 0.5),    # Significant reduction
        (45, 0.3)     # Extreme reduction
    ])
    def test_volatility_scaling(self, configured_strategy, vix, expected_reduction):
        """Verify non-linear position scaling"""
        base_signal = {'direction': 1.0, 'confidence': 0.8}
        scaled_signal = configured_strategy._apply_risk_overlay(
            base_signal,
            {'vix': vix}
        )
        
        reduction = abs(scaled_signal['direction'] / base_signal['direction'])
        assert reduction == pytest.approx(expected_reduction, abs=0.05), \
            f"VIX {vix} should reduce position by {expected_reduction}x"
    
    def test_strategy_specific_vol_limits(self, configured_strategy):
        """Verify config-specific volatility handling"""
        max_vol = configured_strategy.config['ensemble_config']['max_volatility']
        extreme_signal = configured_strategy._apply_risk_overlay(
            {'direction': 1.0, 'confidence': 0.9},
            {'vix': 40}
        )
        
        assert abs(extreme_signal['direction']) <= max_vol, \
            f"Position size {extreme_signal['direction']} exceeds configured max {max_vol}"

## ----------------------------
## Performance Tests
## ----------------------------

class TestEnsemblePerformance:
    """Benchmarking and efficiency tests"""
    
    @pytest.mark.benchmark
    def test_signal_generation_speed(self, configured_strategy):
        """Generate 1000 signals under 500ms"""
        import time
        
        # Warm up
        configured_strategy.generate_signal({})
        
        start = time.perf_counter()
        for _ in range(1000):
            configured_strategy.generate_signal({
                'price': 150.0,
                'volume': 1e6,
                'sentiment': 0.5
            })
        duration = time.perf_counter() - start
        
        assert duration < 0.5, f"Signal generation too slow: {duration:.3f}s"
    
    def test_model_parallelism(self, configured_strategy):
        """Verify models execute in parallel"""
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_pool:
            configured_strategy.generate_signal({})
            assert mock_pool.return_value.__enter__().submit.call_count == \
                   len(configured_strategy.models), \
                   "Not all models executed in parallel"

## ----------------------------
## Failure Mode Tests
## ----------------------------

class TestFailureModes:
    """Tests for error handling and fallbacks"""
    
    def test_model_failure_fallback(self, configured_strategy):
        """Verify graceful handling of model failures"""
        with patch.object(configured_strategy.models['temporal'], 
                         'predict_with_uncertainty',
                         side_effect=Exception("Model failed")) as mock_fail:
            
            # Should disable temporal model but still produce signal
            signal = configured_strategy.generate_signal({})
            assert signal['temporal_weight'] == 0
            assert 'direction' in signal
            assert mock_fail.call_count == 1
    
    def test_missing_data_handling(self, configured_strategy):
        """Test operation with partial market data"""
        with patch.dict('os.environ', {'ALLOW_PARTIAL_SIGNALS': 'true'}):
            signal = configured_strategy.generate_signal({
                'price': None,  # Missing critical field
                'volume': 1e6
            })
            assert signal['confidence'] < 0.5, \
                "Should reduce confidence on missing data"

## ----------------------------
## Integration Tests
## ----------------------------

@pytest.mark.integration
class TestTradingIntegration:
    """Integration with trading execution systems"""
    
    @pytest.fixture
    def mock_executor(self):
        return MagicMock()
    
    def test_full_signal_pipeline(self, configured_strategy, mock_executor):
        """Test end-to-end signal generation and execution"""
        # Generate signal
        signal = configured_strategy.generate_signal({
            'price': 150.0,
            'volume': 2e6,
            'sentiment': 0.7,
            'vix': 25
        })
        
        # Execute
        configured_strategy.execute_signal(signal, mock_executor)
        
        # Verify execution
        assert mock_executor.submit_order.call_count == 1
        order = mock_executor.submit_order.call_args[0][0]
        assert order['size'] == pytest.approx(signal['direction'] * signal['confidence'], abs=0.01)
