import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import timedelta

from core.strategy_engine.base_strategy import BaseStrategy
from core.utils.technical_indicators import (
    calculate_ema, 
    calculate_atr,
    calculate_macd,
    calculate_volume_profile
)

logger = logging.getLogger(__name__)

class MomentumSignalType(Enum):
    BREAKOUT = auto()
    TREND_CONTINUATION = auto()
    VOLATILITY_EXPANSION = auto()
    REVERSION = auto()

@dataclass
class MomentumSignal:
    symbol: str
    signal_type: MomentumSignalType
    direction: int  # 1 for long, -1 for short
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    time_to_hold: timedelta
    volatility_ratio: float
    volume_confirmation: bool
    component_scores: Dict[str, float]

class AdvancedMomentumStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "advanced_momentum"
        self.parameters = self._validate_parameters(config.get('strategies', {}).get(self.name, {}))
        self.required_bars = self._calculate_required_bars()
        self._setup_indicators()
        
    def _validate_parameters(self, params: Dict) -> Dict:
        defaults = {
            'short_ema': 12,
            'long_ema': 26,
            'signal_line': 9,
            'atr_period': 14,
            'volatility_lookback': 63,
            'min_volume_ratio': 1.5,
            'risk_reward_ratio': 2.0,
            'max_position_pct': 0.15,
            'confidence_threshold': 0.65,
            'hold_period_days': 5,
            'require_volume_confirmation': True,
            'require_volatility_expansion': False
        }
        return {**defaults, **params}
    
    def _calculate_required_bars(self) -> int:
        return max(
            self.parameters['long_ema'] * 3,
            self.parameters['volatility_lookback'],
            200
        )
    
    def _setup_indicators(self):
        self.indicators = {
            'macd': lambda p: calculate_macd(
                p, 
                self.parameters['short_ema'],
                self.parameters['long_ema'],
                self.parameters['signal_line']
            ),
            'atr': lambda h, l, c: calculate_atr(
                h, l, c,
                self.parameters['atr_period']
            ),
            'volume_profile': lambda v: calculate_volume_profile(
                v,
                lookback=self.parameters['volatility_lookback']
            )
        }
    
    def _analyze_trend(self, prices: pd.Series) -> Dict:
        trends = {}
        timeframes = [50, 100, 200]
        for tf in timeframes:
            if len(prices) >= tf:
                ema = calculate_ema(prices, tf)
                slope = (ema.iloc[-1] - ema.iloc[-5]) / ema.iloc[-5]
                trends[tf] = {
                    'value': ema.iloc[-1],
                    'slope': slope,
                    'distance': (prices.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1],
                    'momentum': slope * (1 + abs(prices.iloc[-1] - ema.iloc[-1]))
                }
        return trends
    
    def _assess_volatility(self, highs, lows, closes) -> Dict:
        atr = self.indicators['atr'](highs, lows, closes)
        return {
            'current': atr.iloc[-1],
            'ratio': atr.iloc[-1] / atr.rolling(21).mean().iloc[-1],
            'regime': 'high' if atr.iloc[-1] > atr.quantile(0.75) else 
                      'low' if atr.iloc[-1] < atr.quantile(0.25) else 'normal'
        }
    
    def _generate_signal_components(self, symbol: str, data: Dict) -> Dict:
        closes = data['close']
        highs = data['high']
        lows = data['low']
        volumes = data['volume']
        
        macd = self.indicators['macd'](closes)
        volatility = self._assess_volatility(highs, lows, closes)
        trends = self._analyze_trend(closes)
        volume_profile = self.indicators['volume_profile'](volumes)
        
        return {
            'symbol': symbol,
            'macd': macd,
            'volatility': volatility,
            'trends': trends,
            'volume': volume_profile,
            'price': closes.iloc[-1]
        }

    def _calculate_signal_quality(self, components: Dict) -> Tuple[Optional[MomentumSignalType], float]:
        try:
            macd_hist = components['macd']['histogram']
            macd_score = min(1.0, abs(macd_hist) / 0.02)
            
            trend_scores = components['trends']
            trend_strength = trend_scores.get(200, {}).get('momentum', 0) + trend_scores.get(50, {}).get('momentum', 0)
            trend_score = min(1.0, max(0.0, trend_strength))
            
            vol_ratio = components['volatility']['ratio']
            vol_score = min(1.0, vol_ratio / 1.5)
            
            volume_ok = components['volume']['confirmed'] if self.parameters['require_volume_confirmation'] else True
            if not volume_ok:
                return None, 0.0
            
            confidence = 0.4 * trend_score + 0.4 * macd_score + 0.2 * vol_score
            
            if trend_score > 0.5 and vol_score > 0.8:
                signal_type = MomentumSignalType.BREAKOUT
            elif trend_score > 0.4:
                signal_type = MomentumSignalType.TREND_CONTINUATION
            elif macd_score < 0.3 and vol_score < 0.8:
                signal_type = MomentumSignalType.REVERSION
            else:
                signal_type = MomentumSignalType.TREND_CONTINUATION
            
            return signal_type, confidence
        except Exception as e:
            logger.error(f"Signal quality assessment failed: {str(e)}", exc_info=True)
            return None, 0.0

    def _calculate_position_size(self, components: Dict, confidence: float) -> float:
        risk_amount = self.parameters['max_position_pct'] * confidence
        atr_multiple = 1.5 if components['volatility']['regime'] == 'high' else 2.5
        position_size = risk_amount / (components['volatility']['current'] * atr_multiple)
        return min(position_size, self.parameters['max_position_pct'])
    
    def generate_signals(self, market_data: Dict) -> Dict[str, MomentumSignal]:
        signals = {}
        for symbol, data in market_data.items():
            try:
                if len(data['close']) < self.required_bars:
                    continue
                
                components = self._generate_signal_components(symbol, data)
                signal_type, confidence = self._calculate_signal_quality(components)
                
                if not signal_type or confidence < self.parameters['confidence_threshold']:
                    continue
                
                entry = components['price']
                atr = components['volatility']['current']
                stop_loss = entry - (2 * atr) if signal_type == MomentumSignalType.BREAKOUT else entry - (1.5 * atr)
                take_profit = entry + (self.parameters['risk_reward_ratio'] * (entry - stop_loss))
                
                signals[symbol] = MomentumSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    direction=1 if components['macd']['histogram'] > 0 else -1,
                    confidence=confidence,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=self._calculate_position_size(components, confidence),
                    time_to_hold=timedelta(days=self.parameters['hold_period_days']),
                    volatility_ratio=components['volatility']['ratio'],
                    volume_confirmation=components['volume']['confirmed'],
                    component_scores={
                        'trend_strength': components['trends'][200]['momentum'],
                        'macd_strength': abs(components['macd']['histogram']),
                        'volume_ratio': components['volume']['ratio']
                    }
                )
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {str(e)}", exc_info=True)
        return signals
