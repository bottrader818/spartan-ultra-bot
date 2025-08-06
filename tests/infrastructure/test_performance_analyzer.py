import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from performance_analyzer import PerformanceAnalyzer, OnlineStats

# ========== FIXTURES ==========
@pytest.fixture
def analyzer():
    """Default analyzer with conservative thresholds"""
    return PerformanceAnalyzer()

@pytest.fixture
def aggressive_analyzer():
    """Analyzer with tight alert thresholds"""
    return PerformanceAnalyzer({
        'alert_thresholds': {
            'slippage': 0.001,
            'latency_ms': 50
        }
    })

@pytest.fixture
def mock_executions():
    """Generate realistic execution samples"""
    base_time = datetime.utcnow()
    return [
        {
            'latency_ms': 30 + i,
            'slippage': 0.0005 + (i * 0.0001),
            'venue': 'ALPACA',
            'timestamp': (base_time + timedelta(seconds=i)).isoformat()
        }
        for i in range(100)
    ]

# ========== CORE FUNCTIONALITY TESTS ==========
class TestCoreTracking:
    def test_basic_execution_tracking(self, analyzer):
        analyzer.track_execution({
            'latency_ms': 45,
            'slippage': 0.0012
        })
        report = analyzer.generate_report()
        assert report['latency']['mean'] == 45.0
        assert report['slippage']['mean'] == pytest.approx(0.0012)

    def test_invalid_execution_rejection(self, analyzer):
        with patch.object(analyzer, '_validate_execution', return_value=False):
            analyzer.track_execution({'invalid': 'data'})
            assert len(analyzer._execution_log) == 0

# ========== ALERTING TESTS ==========
class TestAlertSystem:
    def test_slippage_alert_triggering(self, aggressive_analyzer):
        aggressive_analyzer.track_execution({
            'latency_ms': 40,
            'slippage': 0.0015  # Above 0.001 threshold
        })
        assert len(aggressive_analyzer._alerts) == 1
        alert = aggressive_analyzer._alerts[0]
        assert alert['metric'] == 'slippage'
        assert alert['severity'] == 'medium'

    def test_latency_alert_severity(self, analyzer):
        analyzer.track_execution({
            'latency_ms': 250,  # 2.5x threshold (100ms)
            'slippage': 0.001
        })
        assert analyzer._alerts[0]['severity'] == 'high'

# ========== STATISTICAL TESTS ==========
class TestStatisticalMethods:
    def test_online_stats_calculation(self):
        stats = OnlineStats()
        values = [10, 20, 30, 40, 50]
        for v in values:
            stats.update(v)
        
        assert stats.mean == pytest.approx(30.0)
        assert stats.variance == pytest.approx(200.0)
        assert stats.stddev == pytest.approx(14.142, abs=0.001)

    def test_rolling_window_calculation(self, analyzer, mock_executions):
        for exec_data in mock_executions:
            analyzer.track_execution(exec_data)
        
        rolling_stats = analyzer.get_rolling_stats(window=10)
        assert rolling_stats['max_latency'] == 139  # Last 10: 130..139
        assert rolling_stats['max_slippage'] == pytest.approx(0.0144)

# ========== PERFORMANCE TESTS ==========
class TestPerformanceBenchmarks:
    @pytest.mark.benchmark
    def test_high_throughput_tracking(self, analyzer):
        """Process 10K executions in <100ms"""
        import time
        start = time.perf_counter()
        
        for i in range(10_000):
            analyzer.track_execution({
                'latency_ms': 30 + (i % 5),
                'slippage': 0.0005 + (i % 10)/10000
            })
        
        duration = time.perf_counter() - start
        assert duration < 0.1, f"Tracking too slow: {duration*1000:.2f}ms"
        assert len(analyzer._execution_log) == 10_000

# ========== INTEGRATION TESTS ==========
class TestTelemetryIntegration:
    def test_alert_publishing(self, analyzer):
        mock_publisher = MagicMock()
        analyzer.config['alert_publisher'] = mock_publisher
        
        analyzer.track_execution({
            'latency_ms': 150,  # Triggers alert
            'slippage': 0.001
        })
        
        assert mock_publisher.publish.call_count == 1
        published_alert = mock_publisher.publish.call_args[0][0]
        assert published_alert['metric'] == 'latency_ms'
        assert published_alert['severity'] == 'medium'

    def test_custom_telemetry_hooks(self, analyzer):
        hook_calls = []
        analyzer.add_telemetry_hook(lambda alert: hook_calls.append(alert))
        
        analyzer.track_execution({
            'latency_ms': 120,
            'slippage': 0.0025
        })
        
        assert len(hook_calls) == 2  # latency + slippage alerts
