
we can. Implement after I just wanna finish these last couple files from typing import List, Dict, Optional, Tuple
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
        
        # State tracking
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._peak_value = 1.0
        self._drawdown_start = None
        self._recovery_start = None
        
        # Event history
        self._events: List[DrawdownEvent] = []
        self._active_event: Optional[DrawdownEvent] = None
        
        # Performance metrics
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
            'recovery_buffer': 0.02,  # 2% recovery buffer
            'reporting_period': timedelta(days=30)
        }
        return {**defaults, **config}

    def update(self, current_value: float, market_data: Optional[Dict] = None) -> Tuple[float, DrawdownSeverity]:
        """
        Update drawdown calculations with current portfolio value
        
        Args:
            current_value: Current portfolio value (normalized)
            market_data: Optional market conditions data
            
        Returns:
            Tuple of (current_drawdown, severity_level)
        """
        # Calculate current drawdown
        self._peak_value = max(self._peak_value, current_value)
        new_drawdown = (self._peak_value - current_value) / self._peak_value
        
        # Detect drawdown start
        if new_drawdown > 0 and self._current_drawdown == 0:
            self._drawdown_start = datetime.now()
            self._active_event = DrawdownEvent(
                start_time=self._drawdown_start,
                peak_time=datetime.now(),
                max_drawdown=new_drawdown,
                duration=timedelta(0),
                market_conditions=market_data or {}
            )
        
        # Update max drawdown
        self._current_drawdown = new_drawdown
        self._max_drawdown = max(self._max_drawdown, new_drawdown)
        
        # Update active event
        if self._active_event:
            self._active_event.max_drawdown = max(self._active_event.max_drawdown, new_drawdown)
            self._active_event.duration = datetime.now() - self._active_event.start_time
            
            # Check for new peak (potential recovery)
            if current_value >= self._peak_value * (1 - self.config['recovery_buffer']):
                if not self._recovery_start:
                    self._recovery_start = datetime.now()
            else:
                self._recovery_start = None
        
        # Check for completed recovery
        if (self._current_drawdown == 0 and 
            self._active_event and 
            not self._active_event.recovery_time):
            
            self._active_event.recovery_time = datetime.now()
            recovery_duration = self._active_event.recovery_time - self._active_event.start_time
            
            # Update metrics
            self._metrics['total_drawdown_time'] += recovery_duration
            if self._active_event.max_drawdown > self._metrics['worst_drawdown']:
                self._metrics['worst_drawdown'] = self._active_event.max_drawdown
            
            # Calculate recovery rate if there was significant drawdown
            if self._active_event.max_drawdown > 0.05:
                recovery_rate = self._active_event.max_drawdown / recovery_duration.total_seconds()
                self._metrics['recovery_rate'] = (
                    self._metrics['recovery_rate'] * len(self._events) + recovery_rate
                ) / (len(self._events) + 1)
            
            # Store completed event
            self._active_event.severity = self._classify_severity(self._active_event.max_drawdown)
            self._metrics['severity_counts'][self._active_event.severity] += 1
            self._events.append(self._active_event)
            self._active_event = None
        
        # Classify current severity
        severity = self._classify_severity(new_drawdown)
        return new_drawdown, severity

    def _classify_severity(self, drawdown: float) -> DrawdownSeverity:
        """Classify drawdown severity based on configured levels"""
        for severity, threshold in sorted(
            self.config['severity_levels'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if drawdown >= threshold:
                return severity
        return DrawdownSeverity.NORMAL

    def get_risk_actions(self) -> List[str]:
        """Get recommended risk control actions based on current drawdown"""
        actions = []
        if self._current_drawdown >= self.config['risk_controls']['stop_trading']:
            actions.append("STOP_ALL_TRADING")
        elif self._current_drawdown >= self.config['risk_controls']['hedge_required']:
            actions.append("HEDGE_POSITIONS")
            actions.append("REDUCE_LEVERAGE")
        elif self._current_drawdown >= self.config['risk_controls']['reduce_position']:
            actions.append("REDUCE_RISK_EXPOSURE")
        
        if self._current_drawdown > 0.1 and not any(
            e.severity == DrawdownSeverity.CRITICAL 
            for e in self._events[-3:]
        ):
            actions.append("INCREASE_MONITORING")
        
        return actions

    def get_performance_report(self) -> Dict:
        """Generate drawdown performance report"""
        recent_events = [
            e for e in self._events 
            if e.start_time > datetime.now() - self.config['reporting_period']
        ]
        
        return {
            'current_drawdown': self._current_drawdown,
            'max_drawdown': self._max_drawdown,
            'active_event': bool(self._active_event),
            'recent_events_count': len(recent_events),
            'avg_recovery_time': self._metrics['recovery_rate'],
            'severity_distribution': self._metrics['severity_counts'],
            'worst_drawdown': self._metrics['worst_drawdown'],
            'total_drawdown_time': self._metrics['total_drawdown_time']
        }

    def plot_drawdown_history(self):
        """Generate visualization of drawdown history"""
        # Would implement actual plotting in production
        pass

# Example usage
if __name__ == "__main__":
    manager = DrawdownManager()
    
    # Simulate portfolio values
    portfolio_values = [1.0, 0.98, 0.95, 0.92, 0.88, 0.85, 0.82, 
                       0.85, 0.88, 0.92, 0.95, 0.98, 1.0]
    
    for value in portfolio_values:
        drawdown, severity = manager.update(value)
        print(f"Value: {value:.2f} | Drawdown: {drawdown:.2%} | Severity: {severity.name}")
        
        if drawdown > 0:
            actions = manager.get_risk_actions()
            if actions:
                print("ACTIONS:", ", ".join(actions))
    
    print("\nPerformance Report:")
    print(manager.get_performance_report())
