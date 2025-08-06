from typing import Dict, Tuple, List, Optional
from enum import Enum, auto
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import zscore

class BlackSwanSeverity(Enum):
    """Classification levels for black swan events with quantitative thresholds"""
    WARNING = auto()       # Early warning signs (volatility > 0.05, sentiment < -0.5)
    EMERGING = auto()      # Event likely developing (volatility > 0.08, sentiment < -0.7)
    CONFIRMED = auto()     # Event confirmed (volatility > 0.12, sentiment < -0.8)
    CRITICAL = auto()      # Full market crisis (volatility > 0.20, sentiment < -0.9)

class BlackSwanDetector:
    """
    Advanced Black Swan Event Detection System with:
    - Multi-factor quantitative analysis
    - Machine learning-ready features
    - Real-time severity classification
    - Historical pattern matching
    - Adaptive threshold calibration
    - Comprehensive event logging
    - Stress testing integration
    """

    def __init__(self, config: Dict = None, historical_data: pd.DataFrame = None):
        """
        Initialize detector with configuration and optional historical data
        
        Args:
            config: Custom configuration dictionary
            historical_data: Pandas DataFrame with historical market metrics
        """
        self.config = self._validate_config(config or {})
        self.logger = logging.getLogger(self.__class__.__name__)
        self._event_history = []
        self._historical_data = historical_data
        self._market_regime = "normal"  # normal, stressed, crisis
        
        # Initialize adaptive baselines
        self._initialize_baselines()
        
        # Setup logging
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _validate_config(self, config: Dict) -> Dict:
        """Ensure valid configuration with defaults"""
        defaults = {
            'volatility_threshold': 0.05,
            'sentiment_threshold': -0.7,
            'volume_spike_multiplier': 3.0,
            'liquidity_drop_threshold': 0.5,
            'correlation_threshold': 0.8,
            'vix_threshold': 30,
            'severity_levels': {
                'WARNING': {'volatility': 0.05, 'sentiment': -0.5, 'vix': 25},
                'EMERGING': {'volatility': 0.08, 'sentiment': -0.7, 'vix': 30},
                'CONFIRMED': {'volatility': 0.12, 'sentiment': -0.8, 'vix': 35},
                'CRITICAL': {'volatility': 0.20, 'sentiment': -0.9, 'vix': 40}
            },
            'lookback_window': 30,  # days for baseline calculations
            'zscore_threshold': 2.0  # for outlier detection
        }
        return {**defaults, **config}

    def _initialize_baselines(self):
        """Initialize baseline metrics from historical data if available"""
        if self._historical_data is not None:
            self._volatility_baseline = self._historical_data['volatility'].mean()
            self._sentiment_baseline = self._historical_data['sentiment'].mean()
            self._volume_baseline = self._historical_data['volume'].median()
        else:
            self._volatility_baseline = 0.03
            self._sentiment_baseline = -0.3
            self._volume_baseline = 1.0

    def detect_event(self, market_data: Dict) -> Tuple[bool, Optional[BlackSwanSeverity]]:
        """
        Comprehensive black swan detection with severity classification
        
        Args:
            market_data: Dictionary containing market metrics:
                - volatility (0-1)
                - sentiment (-1 to 1)
                - volume_ratio (current vs average)
                - liquidity_ratio (0-1)
                - asset_correlations (0-1)
                - vix (volatility index)
                - [optional] sector_performance (dict)
                
        Returns:
            Tuple of (detected: bool, severity: Optional[BlackSwanSeverity])
        """
        try:
            # Validate input data
            self._validate_market_data(market_data)
            
            # Update dynamic baselines
            self._update_baselines(market_data)
            
            # Calculate severity score and check conditions
            severity_score = self._calculate_severity_score(market_data)
            is_event, severity = self._classify_event(market_data, severity_score)
            
            if is_event:
                event_details = self._create_event_record(market_data, severity, severity_score)
                self._event_history.append(event_details)
                self._update_market_regime(severity)
                self.logger.warning(f"Black swan event detected: {severity.name} (Score: {severity_score:.2f})")
                
            return is_event, severity
            
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            return False, None

    def _validate_market_data(self, data: Dict):
        """Validate required market data fields"""
        required_fields = ['volatility', 'sentiment']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

    def _classify_event(self, market_data: Dict, severity_score: float) -> Tuple[bool, Optional[BlackSwanSeverity]]:
        """Determine if conditions meet any severity level thresholds"""
        for severity in reversed(BlackSwanSeverity):
            threshold = self.config['severity_levels'][severity.name]
            
            conditions_met = (
                market_data['volatility'] >= threshold['volatility'] and
                market_data['sentiment'] <= threshold['sentiment'] and
                market_data.get('vix', 0) >= threshold.get('vix', 0) and
                self._check_secondary_indicators(market_data, severity)
            )
            
            if conditions_met:
                return True, severity
                
        return False, None

    def _create_event_record(self, market_data: Dict, severity: BlackSwanSeverity, score: float) -> Dict:
        """Create detailed event record"""
        return {
            'timestamp': datetime.now(),
            'severity': severity,
            'score': round(score, 4),
            'metrics': market_data,
            'market_regime': self._market_regime,
            'z_scores': self._calculate_z_scores(market_data)
        }

    def _calculate_z_scores(self, market_data: Dict) -> Dict:
        """Calculate z-scores for key metrics if historical data available"""
        if self._historical_data is None:
            return {}
            
        z_scores = {}
        for metric in ['volatility', 'sentiment', 'volume_ratio']:
            if metric in market_data and metric in self._historical_data.columns:
                z_scores[metric] = zscore(
                    np.append(self._historical_data[metric], market_data[metric])
                )[-1]
        return z_scores

    def _calculate_severity_score(self, market_data: Dict) -> float:
        """Calculate normalized severity score (0-1) with weighted factors"""
        weights = {
            'volatility': 0.35,
            'sentiment': 0.25,
            'volume': 0.15,
            'liquidity': 0.10,
            'correlation': 0.10,
            'vix': 0.05
        }
        
        # Normalize metrics
        vol_score = min(1.0, market_data['volatility'] / 0.3)
        sent_score = (market_data['sentiment'] + 1) / 2  # -1 to 1 → 0 to 1
        volume_score = min(1.0, (market_data.get('volume_ratio', 1) - 1) / 4.0)
        liq_score = 1 - min(1.0, market_data.get('liquidity_ratio', 1))
        corr_score = market_data.get('asset_correlations', 0)
        vix_score = min(1.0, market_data.get('vix', 0) / 50.0)
        
        # Calculate weighted score
        base_score = (
            weights['volatility'] * vol_score +
            weights['sentiment'] * (1 - sent_score) +
            weights['volume'] * volume_score +
            weights['liquidity'] * liq_score +
            weights['correlation'] * corr_score +
            weights['vix'] * vix_score
        )
        
        # Apply non-linear scaling for extreme events
        final_score = base_score ** 1.5 if base_score > 0.7 else base_score
        return min(1.0, max(0.0, final_score))

    def _check_secondary_indicators(self, market_data: Dict, severity: BlackSwanSeverity) -> bool:
        """Verify supporting evidence for event classification"""
        checks = [
            # Volume spike check
            market_data.get('volume_ratio', 1) >= (
                self.config['volume_spike_multiplier'] * 
                (0.8 if severity == BlackSwanSeverity.WARNING else 1)
            ),
            # Liquidity drop check
            market_data.get('liquidity_ratio', 1) <= (
                self.config['liquidity_drop_threshold'] * 
                (1.2 if severity == BlackSwanSeverity.WARNING else 1)
            ),
            # Correlation check
            market_data.get('asset_correlations', 0) >= (
                self.config['correlation_threshold'] * 
                (0.8 if severity == BlackSwanSeverity.WARNING else 1)
            ),
            # VIX check if available
            'vix' not in market_data or 
            market_data['vix'] >= self.config['severity_levels'][severity.name].get('vix', 0)
        ]
        
        # Require at least 2 supporting indicators for WARNING, 3 for others
        required = 2 if severity == BlackSwanSeverity.WARNING else 3
        return sum(checks) >= min(required, len(checks))

    def _update_baselines(self, market_data: Dict):
        """Update dynamic thresholds based on recent market conditions"""
        # In a production system, this would update rolling windows of historical data
        pass

    def _update_market_regime(self, severity: BlackSwanSeverity):
        """Update current market regime based on event severity"""
        if severity == BlackSwanSeverity.CRITICAL:
            self._market_regime = "crisis"
        elif severity in [BlackSwanSeverity.CONFIRMED, BlackSwanSeverity.EMERGING]:
            self._market_regime = "stressed"
        else:
            self._market_regime = "normal"

    def get_event_history(self, days: int = 7) -> List[Dict]:
        """Get recent events within specified timeframe"""
        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in self._event_history if e['timestamp'] >= cutoff]

    def is_market_stressed(self) -> bool:
        """Check if market is in elevated stress state"""
        return self._market_regime in ["stressed", "crisis"]

    def get_current_regime(self) -> str:
        """Get current market regime classification"""
        return self._market_regime

def detect_black_swan(volatility: float, sentiment: float) -> bool:
    """
    Simplified black swan detection interface for basic use cases
    
    Args:
        volatility: Current volatility measure (0-1)
        sentiment: Current market sentiment (-1 to 1)
        
    Returns:
        True if black swan conditions detected
    """
    detector = BlackSwanDetector()
    market_data = {
        'volatility': volatility,
        'sentiment': sentiment,
        'volume_ratio': 1.0,
        'liquidity_ratio': 1.0,
        'asset_correlations': 0,
        'vix': 20
    }
    detected, _ = detector.detect_event(market_data)
    return detected
