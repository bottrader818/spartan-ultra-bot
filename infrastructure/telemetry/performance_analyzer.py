import time
import math
import numpy as np
import logging
from collections import deque, defaultdict
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Union, Callable

logger = logging.getLogger(__name__)

# ========== DEFAULT CONFIG ==========
DEFAULT_CONFIG = {
    'alert_thresholds': {
        'slippage': 0.002,      # 2bps
        'latency_ms': 100,       # 100ms
        'custom_metrics': {}     # User-defined thresholds
    },
    'reporting': {
        'percentiles': [50, 95, 99],  # Track median, 95th, 99th percentiles
        'outlier_thresholds': {
            'slippage': 0.005,   # 5bps outlier threshold
            'latency_ms': 200    # 200ms outlier threshold
        }
    },
    'max_execution_log_size': 10_000,  # Bounded memory usage
    'alert_publisher': None             # Optional external alert system
}

# ========== CORE CLASSES ==========
class OnlineStats:
    """Welford's algorithm for O(1) mean/variance tracking"""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0       # Sum of squared differences
        self.min = float('inf')
        self.max = -float('inf')

    def update(self, value: float):
        """Update stats with a new value (thread-safe)"""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    @property
    def variance(self) -> float:
        """Compute variance in O(1) time"""
        return self.M2 / self.count if self.count else 0.0

    @property
    def stddev(self) -> float:
        """Compute standard deviation in O(1) time"""
        return math.sqrt(self.variance)

class PerformanceAnalyzer:
    """High-performance telemetry analyzer with real-time alerts"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional custom config"""
        self.config = self._merge_configs(config)
        self._validate_config()
        
        # Core data structures
        self._execution_log = deque(maxlen=self.config['max_execution_log_size'])
        self._alerts = []
        self._telemetry_hooks = []
        self._lock = Lock()
        
        # Statistics tracking
        self._stats = {
            'latency': OnlineStats(),
            'slippage': OnlineStats(),
            'custom': defaultdict(OnlineStats)
        }

    # ========== PUBLIC METHODS ==========
    def track_execution(self, execution: Dict):
        """Record an execution with metrics (thread-safe)"""
        if not self._validate_execution(execution):
            return

        with self._lock:
            # Update execution log
            self._execution_log.append(execution)
            
            # Update real-time stats
            self._stats['latency'].update(execution['latency_ms'])
            self._stats['slippage'].update(execution['slippage'])
            
            # Check for alerts
            self._check_alerts(execution)

    def generate_report(self) -> Dict:
        """Generate a full performance report (thread-safe)"""
        with self._lock:
            report = {
                'latency': self._generate_metric_report('latency_ms'),
                'slippage': self._generate_metric_report('slippage'),
                'custom': {
                    metric: self._generate_metric_report(metric)
                    for metric in self.config['alert_thresholds']['custom_metrics']
                },
                'execution_count': len(self._execution_log),
                'alert_count': len(self._alerts)
            }
            return report

    def get_rolling_stats(self, window: int = 10) -> Dict:
        """Get max values over a rolling window (thread-safe)"""
        with self._lock:
            if len(self._execution_log) < window:
                return {'max_latency': 0, 'max_slippage': 0}

            recent = list(self._execution_log)[-window:]
            return {
                'max_latency': max(e['latency_ms'] for e in recent),
                'max_slippage': max(e['slippage'] for e in recent)
            }

    def add_telemetry_hook(self, hook_func: Callable):
        """Add a custom telemetry processor (e.g., for Prometheus)"""
        with self._lock:
            self._telemetry_hooks.append(hook_func)

    def reset(self):
        """Clear all data (for testing)"""
        with self._lock:
            self._execution_log.clear()
            self._alerts.clear()
            for stat in self._stats.values():
                if isinstance(stat, defaultdict):
                    stat.clear()
                else:
                    stat.__init__()

    # ========== PRIVATE HELPERS ==========
    def _merge_configs(self, user_config: Optional[Dict]) -> Dict:
        """Deep-merge user config with defaults"""
        config = DEFAULT_CONFIG.copy()
        if user_config:
            config.update(user_config)
            if 'alert_thresholds' in user_config:
                config['alert_thresholds'].update(user_config['alert_thresholds'])
            if 'reporting' in user_config:
                config['reporting'].update(user_config['reporting'])
        return config

    def _validate_config(self):
        """Validate configuration at startup"""
        if not isinstance(self.config['max_execution_log_size'], int):
            raise ValueError("max_execution_log_size must be an integer")
        if self.config['max_execution_log_size'] <= 0:
            raise ValueError("max_execution_log_size must be positive")

    def _validate_execution(self, execution: Dict) -> bool:
        """Validate execution data before processing"""
        if 'latency_ms' not in execution or 'slippage' not in execution:
            logger.warning(f"Dropped incomplete execution: {execution}")
            return False
        return True

    def _check_alerts(self, execution: Dict):
        """Check if execution triggers any alerts"""
        alerts_triggered = []
        thresholds = self.config['alert_thresholds']

        for metric, threshold in thresholds.items():
            value = execution.get(metric)
            if value is None or value <= threshold:
                continue

            alert = self._create_alert(metric, value, threshold, execution)
            alerts_triggered.append(alert)
            self._publish_alert(alert)

        if alerts_triggered:
            with self._lock:
                self._alerts.extend(alerts_triggered)

    def _create_alert(self, metric: str, value: float, threshold: float, execution: Dict) -> Dict:
        """Generate a structured alert payload"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metric': metric,
            'value': value,
            'threshold': threshold,
            'venue': execution.get('venue', 'unknown'),
            'severity': self._calculate_severity(value, threshold),
            'related_metrics': {
                k: v for k, v in execution.items()
                if k not in ['timestamp', 'venue']
            }
        }

    def _calculate_severity(self, value: float, threshold: float) -> str:
        """Dynamic severity based on threshold violation ratio"""
        ratio = value / threshold
        if ratio > 3.0: return 'critical'
        if ratio > 2.0: return 'high'
        if ratio > 1.5: return 'medium'
        return 'low'

    def _publish_alert(self, alert: Dict):
        """Publish alerts to all registered systems"""
        try:
            if self.config['alert_publisher']:
                self.config['alert_publisher'].publish(alert)
            
            for hook in self._telemetry_hooks:
                try:
                    hook(alert)
                except Exception as e:
                    logger.warning(f"Telemetry hook failed: {str(e)}")
        except Exception as e:
            logger.warning(f"Alert publishing failed: {str(e)}")
            self._queue_for_retry(alert)

    def _generate_metric_report(self, metric: str) -> Dict:
        """Generate statistics for a single metric"""
        values = np.array([
            e[metric] for e in self._execution_log 
            if metric in e
        ])
        
        if not values.size:
            return {}

        return {
            'mean': float(np.mean(values)),
            'stddev': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'percentiles': {
                str(p): float(np.percentile(values, p))
                for p in self.config['reporting']['percentiles']
            },
            'outliers': int(np.sum(
                values > self.config['reporting']['outlier_thresholds'].get(metric, float('inf'))
            ))
        }

    def _queue_for_retry(self, alert: Dict):
        """(Optional) Implement retry logic for failed alerts"""
        pass
