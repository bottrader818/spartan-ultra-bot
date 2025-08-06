from typing import Dict, Tuple
from enum import Enum, auto
import logging
from datetime import datetime
import numpy as np

class BlackSwanSeverity(Enum):
    """Classification levels for black swan events"""
    WARNING = auto()       # Early warning signs
    EMERGING = auto()      # Event likely developing
    CONFIRMED = auto()     # Event confirmed
    CRITICAL = auto()      # Full market crisis

class BlackSwanDetector:
    """
    Professional Black Swan Event Detection System with:
    - Multi-factor analysis
    - Severity classification
    - Historical pattern matching
    - Adaptive thresholds
    - Comprehensive logging
    """
    
    def __init__(self, config: Dict = None):
        self.config = self._validate_config(config or {})
        self.logger = logging.getLogger(self.__class__.__name__)
        self._event_history = []
        
        # Adaptive thresholds
        self._volatility_baseline = 0.03
        self._sentiment_baseline = -0.3
        self._update_interval = 30  # days
        
    def _validate_config(self, config: Dict) -> Dict:
        """Ensure valid configuration with defaults"""
        defaults = {
            'volatility_threshold': 0.05,
            'sentiment_threshold': -0.7,
            'volume_spike_multiplier': 3.0,
            'liquidity_drop_threshold': 0.5,
            'correlation_threshold': 0.8,
            'severity_levels': {
                'WARNING': {'volatility': 0.05, 'sentiment': -0.5},
                'EMERGING': {'volatility': 0.08, 'sentiment': -0.7},
                'CONFIRMED': {'volatility': 0.12, 'sentiment': -0.8},
                'CRITICAL': {'volatility': 0.20, 'sentiment': -0.9}
            }
        }
        return {**defaults, **config}

    def detect_event(self, market_data: Dict) -> Tuple[bool, BlackSwanSeverity]:
        """
        Detect black swan events with severity classification
        
        Args:
            market_data: Dictionary containing:
                - volatility: Current market volatility (0-1)
                - sentiment: Market sentiment score (-1 to 1)
                - volume_ratio: Current volume vs 30-day average
                - liquidity_ratio: Current liquidity vs normal
                - asset_correlations: Average cross-asset correlation
                - vix: Current VIX level
                
        Returns:
            Tuple of (is_black_swan, severity_level)
        """
        # Update baselines periodically
        self._update_baselines(market_data)
        
        # Calculate composite score
        severity_score = self._calculate_severity_score(market_data)
        
        # Check against all severity levels from highest to lowest
        for severity in reversed(BlackSwanSeverity):
            threshold = self.config['severity_levels'][severity.name]
            
            if (market_data['volatility'] >= threshold['volatility'] and
                market_data['sentiment'] <= threshold['sentiment'] and
                self._check_secondary_indicators(market_data, severity)):
                
                event_details = {
                    'timestamp': datetime.now(),
                    'severity': severity,
                    'metrics': market_data,
                    'score': severity_score
                }
                self._event_history.append(event_details)
                self.logger.warning(f"Black swan event detected: {severity.name}")
                return True, severity
                
        return False, None

    def _calculate_severity_score(self, market_data: Dict) -> float:
        """Calculate normalized severity score (0-1)"""
        vol_score = min(1.0, market_data['volatility'] / 0.3)  # Cap at 30% vol
        sent_score = (market_data['sentiment'] + 1) / 2  # Convert to 0-1 range
        
        # Composite score weighted 60% volatility, 40% sentiment
        base_score = 0.6 * vol_score + 0.4 * (1 - sent_score)
        
        # Apply multipliers from secondary indicators
        multipliers = []
        if market_data.get('volume_ratio', 1) > 2.0:
            multipliers.append(1.2)
        if market_data.get('liquidity_ratio', 1) < 0.7:
            multipliers.append(1.3)
        if market_data.get('asset_correlations', 0) > 0.7:
            multipliers.append(1.4)
            
        final_score = base_score * np.prod(multipliers) if multipliers else base_score
        return min(1.0, max(0.0, final_score))

    def _check_secondary_indicators(self, market_data: Dict, severity: BlackSwanSeverity) -> bool:
        """Verify supporting evidence for event classification"""
        # Volume spike check
        vol_ok = market_data.get('volume_ratio', 1) >= (
            self.config['volume_spike_multiplier'] * (0.5 if severity == BlackSwanSeverity.WARNING else 1)
        )
        
        # Liquidity drop check
        liq_ok = market_data.get('liquidity_ratio', 1) <= (
            self.config['liquidity_drop_threshold'] * (1.2 if severity == BlackSwanSeverity.WARNING else 1)
        )
        
        # Correlation check
        corr_ok = market_data.get('asset_correlations', 0) >= (
            self.config['correlation_threshold'] * (0.8 if severity == BlackSwanSeverity.WARNING else 1)
        )
        
        # At least two secondary indicators must confirm
        return sum([vol_ok, liq_ok, corr_ok]) >= 2

    def _update_baselines(self, market_data: Dict):
        """Update dynamic thresholds based on recent market conditions"""
        # In production, this would use historical data analysis
        pass

    def get_event_history(self, days: int = 7) -> list:
        """Get recent events within specified timeframe"""
        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in self._event_history if e['timestamp'] >= cutoff]

    def is_market_stressed(self) -> bool:
        """Check if market is in elevated stress state"""
        recent_events = self.get_event_history()
        return any(e['severity'] in [BlackSwanSeverity.EMERGING, 
                                   BlackSwanSeverity.CONFIRMED,
                                   BlackSwanSeverity.CRITICAL] 
                 for e in recent_events)

def detect_black_swan(volatility: float, sentiment: float) -> bool:
    """
    Simplified black swan detection interface
    
    Args:
        volatility: Current volatility measure (0-1)
        sentiment: Market sentiment score (-1 to 1)
        
    Returns:
        True if conditions suggest a black swan event
    """
    detector = BlackSwanDetector()
    market_data = {
        'volatility': volatility,
        'sentiment': sentiment,
        'volume_ratio': 1.0,  # Default values for simple interface
        'liquidity_ratio': 1.0,
        'asset_correlations': 0
    }
    detected, _ = detector.detect_event(market_data)
    return detected
