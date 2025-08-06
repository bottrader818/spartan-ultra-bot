from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
import logging
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field

class DrawdownSeverity(Enum):
    """Classification levels for drawdown severity"""
    NORMAL = auto()        # < 5%
    WATCH = auto()         # 5-10%
    WARNING = auto()       # 10-15%
    ALERT = auto()         # 15-20%
    CRITICAL = auto()      # > 20%

@dataclass
class DrawdownEvent:
    """Detailed drawdown event tracking"""
    start_time: datetime
    peak_time: datetime
    max_drawdown: float
    duration: timedelta
    recovery_time: Optional[datetime] = None
    severity: DrawdownSeverity = DrawdownSeverity.NORMAL
    market_conditions: Dict = field(default_factory=dict)

class DrawdownManager:
    """
    Professional Drawdown Management System with:
    - Historical drawdown tracking
    - Severity classification
    - Recovery analysis
    - Automated risk controls
    - Performance attribution
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = self._validate_config(config or {})
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._peak_value = 1.0
        self._drawdown_start = None
        self._recovery_start = None
        self._events: List[DrawdownEvent] = []
        self._active_event: Optional[DrawdownEvent] = None
        self._metrics = {
            'total_drawdown_time': timedelta(0),
            'worst_drawdown': 0.0,
            'recovery_rate': 0.0,
            'severity_counts': {level: 0 for level in DrawdownSeverity}
        }

    def _validate_config(self, config: Dict) -> Dict:
        """Ensure valid configuration with defaults"""
        defaults = {
            'severity_levels': {
                DrawdownSeverity.NORMAL: 0.05,
                DrawdownSeverity.WATCH: 0.10,
                DrawdownSeverity.WARNING: 0.15,
                DrawdownSeverity.ALERT: 0.20,
                DrawdownSeverity.CRITICAL: 0.25
            },
            'risk_controls': {
                'reduce_position': 0.15,
                'hedge_required': 0.20,
                'stop_trading': 0.25
            },
            'recovery_buffer': 0.02,
            'reporting_period': timedelta(days=30)
        }
        return {**defaults, **config}

    def update(self, current_value: float, market_data: Optional[Dict] = None) -> Tuple[float, DrawdownSeverity]:
        """Update drawdown calculations with current portfolio value"""
        now = datetime.now()
        
        if current_value > self._peak_value:
            self._peak_value = current_value
            self._drawdown_start = None
            self._current_drawdown = 0.0
            if self._active_event and not self._active_event.recovery_time:
                self._active_event.recovery_time = now
                self._metrics['total_drawdown_time'] += now - self._active_event.start_time
            self._active_event = None

        else:
            drawdown = (self._peak_value - current_value) / self._peak_value
            self._current_drawdown = drawdown
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown
                self._metrics['worst_drawdown'] = drawdown

            severity = self._classify_severity(drawdown)
            if not self._active_event:
                self._active_event = DrawdownEvent(
                    start_time=now,
                    peak_time=now,
                    max_drawdown=drawdown,
                    duration=timedelta(0),
                    severity=severity,
                    market_conditions=market_data or {}
                )
                self._drawdown_start = now
                self._events.append(self._active_event)
            else:
                self._active_event.max_drawdown = max(self._active_event.max_drawdown, drawdown)
                self._active_event.duration = now - self._active_event.start_time
                self._active_event.severity = severity

            self._metrics['severity_counts'][severity] += 1

        return self._current_drawdown, self._active_event.severity if self._active_event else DrawdownSeverity.NORMAL

    def _classify_severity(self, drawdown: float) -> DrawdownSeverity:
        levels = self.config['severity_levels']
        for severity in reversed(DrawdownSeverity):
            if drawdown >= levels.get(severity, 0.0):
                return severity
        return DrawdownSeverity.NORMAL

    def get_active_event(self) -> Optional[DrawdownEvent]:
        """Return current active drawdown event if any"""
        return self._active_event

    def get_metrics(self) -> Dict:
        """Return drawdown performance metrics"""
        return self._metrics

    def reset(self):
        """Clear state and history"""
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._peak_value = 1.0
        self._drawdown_start = None
        self._active_event = None
        self._events.clear()
        self._metrics = {
            'total_drawdown_time': timedelta(0),
            'worst_drawdown': 0.0,
            'recovery_rate': 0.0,
            'severity_counts': {level: 0 for level in DrawdownSeverity}
        }
