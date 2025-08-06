import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from enum import Enum, auto
from scipy.stats import linregress
from sklearn.cluster import KMeans
import logging


class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_BULL = auto()
    WEAK_BULL = auto()
    SIDEWAYS = auto()
    WEAK_BEAR = auto()
    STRONG_BEAR = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()


def detect_market_regime(
    data: pd.DataFrame,
    lookback_window: int = 90,
    volatility_thresholds: Tuple[float, float] = (0.15, 0.05),
    trend_strength_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Advanced market regime detection using multiple indicators
    """
    required_columns = {'close', 'high', 'low', 'volume'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Data must contain {required_columns} columns")

    if len(data) < lookback_window:
        raise ValueError(f"Need at least {lookback_window} data points")

    recent_data = data.iloc[-lookback_window:]
    closes = recent_data['close'].values
    highs = recent_data['high'].values
    lows = recent_data['low'].values
    volumes = recent_data['volume'].values

    results = {
        'trend_strength': _calculate_trend_strength(closes),
        'volatility': _calculate_volatility(highs, lows, closes),
        'volume_profile': _analyze_volume_profile(volumes),
        'cluster_regime': _cluster_based_regime(closes),
        'candle_pattern': _analyze_candle_patterns(recent_data)
    }

    primary_regime = _determine_primary_regime(
        results['trend_strength'],
        results['volatility'],
        volatility_thresholds,
        trend_strength_threshold
    )

    secondary_regime = _determine_volatility_regime(
        results['volatility'],
        volatility_thresholds
    )

    confidence = _calculate_regime_confidence(
        primary_regime,
        results['trend_strength'],
        results['volatility'],
        volatility_thresholds
    )

    return {
        'primary_regime': primary_regime,
        'secondary_regime': secondary_regime or MarketRegime.SIDEWAYS,
        'confidence': confidence,
        'trend_strength': results['trend_strength'],
        'volatility': results['volatility'],
        'metrics': results
    }


def _calculate_trend_strength(prices: np.ndarray) -> float:
    """Calculate normalized trend strength (-1 to 1)"""
    x = np.arange(len(prices))
    slope, _, _, _, _ = linregress(x, prices)
    std_dev = np.std(prices)
    return slope / std_dev if std_dev > 0 else 0.0def _calculate_volatility(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> float:
    """Calculate normalized volatility (0-1 scale)"""
    true_ranges = np.maximum(
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1])
    )
    atr = np.mean(true_ranges[-period:]) if len(true_ranges) >= period else np.mean(true_ranges)
    return atr / np.mean(closes) if np.mean(closes) > 0 else 0.0


def _analyze_volume_profile(volumes: np.ndarray) -> Dict[str, float]:
    """Analyze volume characteristics"""
    recent_volume = volumes[-14:].mean()
    avg_volume = volumes.mean()
    return {
        'volume_ratio': recent_volume / avg_volume if avg_volume > 0 else 1.0,
        'volume_trend': _calculate_trend_strength(volumes)
    }


def _cluster_based_regime(prices: np.ndarray, n_clusters: int = 3) -> MarketRegime:
    """Use K-means clustering to identify price regimes"""
    try:
        prices_reshaped = prices.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(prices_reshaped)

        sorted_centers = sorted(kmeans.cluster_centers_.flatten())
        current_price = prices[-1]

        if current_price >= sorted_centers[-1]:
            return MarketRegime.STRONG_BULL
        elif current_price >= sorted_centers[-2]:
            return MarketRegime.WEAK_BULL
        elif current_price <= sorted_centers[0]:
            return MarketRegime.STRONG_BEAR
        elif current_price <= sorted_centers[1]:
            return MarketRegime.WEAK_BEAR
        else:
            return MarketRegime.SIDEWAYS
    except Exception as e:
        logging.warning(f"Clustering failed: {str(e)}")
        return MarketRegime.SIDEWAYS


def _analyze_candle_patterns(data: pd.DataFrame) -> Dict[str, float]:
    """Analyze candlestick patterns"""
    up_days = (data['close'] > data['open']).sum()
    down_days = (data['close'] < data['open']).sum()
    return {
        'bullish_ratio': up_days / len(data),
        'bearish_ratio': down_days / len(data)
    }


def _determine_primary_regime(
    trend_strength: float,
    volatility: float,
    volatility_thresholds: Tuple[float, float],
    trend_threshold: float
) -> MarketRegime:
    high_vol, low_vol = volatility_thresholds

    if volatility > high_vol:
        return MarketRegime.HIGH_VOLATILITY
    elif volatility < low_vol:
        return MarketRegime.LOW_VOLATILITY
    elif trend_strength > trend_threshold:
        return MarketRegime.STRONG_BULL if trend_strength > trend_threshold * 2 else MarketRegime.WEAK_BULL
    elif trend_strength < -trend_threshold:
        return MarketRegime.STRONG_BEAR if trend_strength < -trend_threshold * 2 else MarketRegime.WEAK_BEAR
    else:
        return MarketRegime.SIDEWAYS


def _determine_volatility_regime(
    volatility: float,
    volatility_thresholds: Tuple[float, float]
) -> Optional[MarketRegime]:
    high_vol, low_vol = volatility_thresholds
    if volatility > high_vol:
        return MarketRegime.HIGH_VOLATILITY
    elif volatility < low_vol:
        return MarketRegime.LOW_VOLATILITY
    return None


def _calculate_regime_confidence(
    regime: MarketRegime,
    trend_strength: float,
    volatility: float,
    volatility_thresholds: Tuple[float, float]
) -> float:
    high_vol, low_vol = volatility_thresholds

    if regime == MarketRegime.HIGH_VOLATILITY:
        return min(1.0, (volatility - high_vol) / high_vol + 0.5)
    elif regime == MarketRegime.LOW_VOLATILITY:
        return min(1.0, (low_vol - volatility) / low_vol + 0.5)
    else:
        return min(1.0, abs(trend_strength) * 2)
