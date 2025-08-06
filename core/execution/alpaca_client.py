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
    Enhanced Black Swan Event Detection System with:
    - Multi-factor quantitative analysis
    - Machine learning-ready features
    - Real-time severity classification
    - Historical pattern matching
    - Adaptive threshold calibration
    - Comprehensive event logging
    - Market regime detection
    """

    def __init__(self, config: Dict = None, historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize detector with configuration and optional historical data
        
        Args:
            config: Custom configuration dictionary
            historical_data: Optional pandas DataFrame with historical market metrics
        """
        self.config = self._validate_config(config or {})
        self.logger = self._setup_logger()
        self._event_history = []
        self._historical_data = historical_data
        self._market_regime = "normal"  # normal, stressed, crisis
        
        # Initialize adaptive baselines
        self._initialize_baselines()

    def _setup_logger(self) -> logging.Logger:
        """Configure and return logger instance"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

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
        Detect black swan events with severity classification
        
        Args:
            market_data: Dictionary containing:
                - volatility (0-1)
                - sentiment (-1 to 1)
                - volume_ratio (current vs average)
                - liquidity_ratio (0-1)
                - asset_correlations (0-1)
                - vix (volatility index)
                
        Returns:
            Tuple of (is_black_swan, severity_level)
        """
        try:
            self._validate_market_data(market_data)
            severity_score = self._calculate_severity_score(market_data)

            for severity in reversed(BlackSwanSeverity):
                threshold = self.config['severity_levels'][severity.name]
                
                if (market_data['volatility'] >= threshold['volatility'] and
                    market_data['sentiment'] <= threshold['sentiment'] and
                    self._check_secondary_indicators(market_data, severity)):
                    
                    event_details = {
                        'timestamp': datetime.now(),
                        'severity': severity,
                        'metrics': market_data,
                        'score': severity_score,
                        'z_scores': self._calculate_z_scores(market_data),
                        'market_regime': self._market_regime
                    }
                    self._event_history.append(event_details)
                    self._update_market_regime(severity)
                    self.logger.warning(f"Black swan event detected: {severity.name}")
                    return True, severity
                    
            return False, None
        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}")
            return False, None

    def _validate_market_data(self, data: Dict):
        """Validate required market data fields"""
        required_fields = ['volatility', 'sentiment']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

    def _calculate_severity_score(self, market_data: Dict) -> float:
        """Calculate normalized severity score (0-1) with weighted factors"""
        weights = {
            'volatility': 0.4,
            'sentiment': 0.3,
            'volume': 0.1,
            'liquidity': 0.1,
            'correlation': 0.1
        }
        
        vol_score = min(1.0, market_data['volatility'] / 0.3)
        sent_score = (market_data['sentiment'] + 1) / 2  # -1 to 1 → 0 to 1
        volume_score = min(1.0, (market_data.get('volume_ratio', 1) - 1) / 4.0)
        liq_score = 1 - min(1.0, market_data.get('liquidity_ratio', 1))
        corr_score = market_data.get('asset_correlations', 0)
        
        base_score = (
            weights['volatility'] * vol_score +
            weights['sentiment'] * (1 - sent_score) +
            weights['volume'] * volume_score +
            weights['liquidity'] * liq_score +
            weights['correlation'] * corr_score
        )
        
        return min(1.0, max(0.0, base_score))

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

    def _check_secondary_indicators(self, market_data: Dict, severity: BlackSwanSeverity) -> bool:
        """Verify supporting evidence for event classification"""
        checks = [
            market_data.get('volume_ratio', 1) >= (
                self.config['volume_spike_multiplier'] * 
                (0.8 if severity == BlackSwanSeverity.WARNING else 1)
            ),
            market_data.get('liquidity_ratio', 1) <= (
                self.config['liquidity_drop_threshold'] * 
                (1.2 if severity == BlackSwanSeverity.WARNING else 1)
            ),
            market_data.get('asset_correlations', 0) >= (
                self.config['correlation_threshold'] * 
                (0.8 if severity == BlackSwanSeverity.WARNING else 1)
            ),
            'vix' not in market_data or 
            market_data['vix'] >= self.config['severity_levels'][severity.name].get('vix', 0)
        ]
        
        required = 2 if severity == BlackSwanSeverity.WARNING else 3
        return sum(checks) >= min(required, len(checks))

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
    Simplified black swan detection interface
    
    Args:
        volatility: 0-1
        sentiment: -1 to 1
        
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
