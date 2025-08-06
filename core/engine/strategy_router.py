from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
import logging
import numpy as np


class MarketRegime(Enum):
    STRONG_BULL = 1
    WEAK_BULL = 2
    SIDEWAYS = 3
    WEAK_BEAR = 4
    STRONG_BEAR = 5
    HIGH_VOLATILITY = 6
    LOW_VOLATILITY = 7


class StrategyType(Enum):
    MOMENTUM = "MomentumStrategy"
    MEAN_REVERSION = "MeanReversionStrategy"
    VOLATILITY = "VolatilityStrategy"
    BREAKOUT = "BreakoutStrategy"
    ARBITRAGE = "ArbitrageStrategy"
    MACHINE_LEARNING = "MLStrategy"
    HYBRID = "HybridStrategy"


@dataclass
class StrategyConfig:
    """Configuration for strategy routing"""
    base_risk: float = 0.01
    position_size_multipliers: Optional[Dict[MarketRegime, float]] = None
    preferred_strategies: Optional[Dict[MarketRegime, StrategyType]] = None
    volatility_threshold: float = 0.15
    trend_strength_threshold: float = 0.3

    def __post_init__(self):
        if self.position_size_multipliers is None:
            self.position_size_multipliers = {
                MarketRegime.STRONG_BULL: 1.5,
                MarketRegime.WEAK_BULL: 1.2,
                MarketRegime.SIDEWAYS: 0.8,
                MarketRegime.WEAK_BEAR: 0.7,
                MarketRegime.STRONG_BEAR: 0.5,
                MarketRegime.HIGH_VOLATILITY: 0.6,
                MarketRegime.LOW_VOLATILITY: 1.1
            }

        if self.preferred_strategies is None:
            self.preferred_strategies = {
                MarketRegime.STRONG_BULL: StrategyType.MOMENTUM,
                MarketRegime.WEAK_BULL: StrategyType.HYBRID,
                MarketRegime.SIDEWAYS: StrategyType.MEAN_REVERSION,
                MarketRegime.WEAK_BEAR: StrategyType.VOLATILITY,
                MarketRegime.STRONG_BEAR: StrategyType.VOLATILITY,
                MarketRegime.HIGH_VOLATILITY: StrategyType.BREAKOUT,
                MarketRegime.LOW_VOLATILITY: StrategyType.ARBITRAGE
            }


def route_strategy(
    market_regime: MarketRegime,
    config: Optional[StrategyConfig] = None,
    additional_factors: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if config is None:
        config = StrategyConfig()
    if additional_factors is None:
        additional_factors = {}

    if not isinstance(market_regime, MarketRegime):
        raise ValueError("market_regime must be a MarketRegime enum value")

    recommended_strategy = config.preferred_strategies.get(market_regime, StrategyType.HYBRID)
    risk_multiplier = config.position_size_multipliers.get(market_regime, 1.0)
    adjusted_risk = config.base_risk * risk_multiplier

    confidence = _calculate_routing_confidence(market_regime, additional_factors)
    strategy_config = _get_strategy_config(recommended_strategy, market_regime, additional_factors)

    return {
        "strategy": recommended_strategy,
        "strategy_class": recommended_strategy.value,
        "config": strategy_config,
        "risk_parameters": {
            "base_risk": config.base_risk,
            "adjusted_risk": adjusted_risk,
            "position_size_multiplier": risk_multiplier,
            "max_leverage": _get_max_leverage(market_regime)
        },
        "confidence": confidence,
        "market_regime": market_regime.name,
        "additional_factors": additional_factors
    }def _calculate_routing_confidence(
    market_regime: MarketRegime,
    factors: Dict[str, Any]
) -> float:
    confidence_map = {
        MarketRegime.STRONG_BULL: 0.9,
        MarketRegime.WEAK_BULL: 0.7,
        MarketRegime.SIDEWAYS: 0.8,
        MarketRegime.WEAK_BEAR: 0.7,
        MarketRegime.STRONG_BEAR: 0.9,
        MarketRegime.HIGH_VOLATILITY: 0.85,
        MarketRegime.LOW_VOLATILITY: 0.75
    }

    base_confidence = confidence_map.get(market_regime, 0.5)
    volatility = factors.get('volatility', 0.1)
    trend_strength = factors.get('trend_strength', 0)
    volume_ratio = factors.get('volume_ratio', 1.0)

    if market_regime in [MarketRegime.STRONG_BULL, MarketRegime.STRONG_BEAR]:
        vol_adjustment = min(1.0, volatility / 0.2)
    else:
        vol_adjustment = 1.0 - min(1.0, abs(volatility - 0.1) / 0.15)

    if market_regime in [MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL]:
        trend_adjustment = min(1.0, trend_strength / 0.5)
    elif market_regime in [MarketRegime.STRONG_BEAR, MarketRegime.WEAK_BEAR]:
        trend_adjustment = min(1.0, abs(trend_strength) / 0.5)
    else:
        trend_adjustment = 1.0 - min(1.0, abs(trend_strength) / 0.3)

    volume_adjustment = min(1.0, volume_ratio / 1.5)

    adjusted_confidence = (
        base_confidence * 0.5 +
        vol_adjustment * 0.2 +
        trend_adjustment * 0.2 +
        volume_adjustment * 0.1
    )

    return max(0.1, min(1.0, adjusted_confidence))


def _get_strategy_config(
    strategy: StrategyType,
    regime: MarketRegime,
    factors: Dict[str, Any]
) -> Dict[str, Any]:
    base_config = {
        "lookback_window": 50,
        "risk_per_trade": 0.01,
        "max_position_size": 0.1
    }

    if strategy == StrategyType.MOMENTUM:
        return {
            **base_config,
            "lookback_window": 30,
            "trend_confirmation": True,
            "exit_on_trend_reversal": True,
            "volatility_filter": factors.get('volatility', 0.1) < 0.25
        }
    elif strategy == StrategyType.MEAN_REVERSION:
        return {
            **base_config,
            "lookback_window": 100,
            "zscore_threshold": 2.0,
            "atr_multiplier": 1.5,
            "require_oversold": regime in [MarketRegime.STRONG_BEAR, MarketRegime.WEAK_BEAR]
        }
    elif strategy == StrategyType.VOLATILITY:
        return {
            **base_config,
            "lookback_window": 20,
            "volatility_window": 14,
            "target_multiplier": 1.8,
            "use_iv_rank": factors.get('iv_rank_available', False)
        }
    elif strategy == StrategyType.BREAKOUT:
        return {
            **base_config,
            "lookback_window": 20,
            "consolidation_period": 10,
            "volume_threshold": 1.5,
            "use_atr_stops": True
        }
    else:
        return base_config


def _get_max_leverage(regime: MarketRegime) -> float:
    leverage_map = {
        MarketRegime.STRONG_BULL: 3.0,
        MarketRegime.WEAK_BULL: 2.0,
        MarketRegime.SIDEWAYS: 1.5,
        MarketRegime.WEAK_BEAR: 2.0,
        MarketRegime.STRONG_BEAR: 1.0,
        MarketRegime.HIGH_VOLATILITY: 1.0,
        MarketRegime.LOW_VOLATILITY: 2.0
    }
    return leverage_map.get(regime, 1.0)


# Example usage
if __name__ == "__main__":
    regime = MarketRegime.STRONG_BULL
    factors = {
        'volatility': 0.18,
        'trend_strength': 0.45,
        'volume_ratio': 1.8
    }
    routing = route_strategy(regime, additional_factors=factors)
    print(routing)
