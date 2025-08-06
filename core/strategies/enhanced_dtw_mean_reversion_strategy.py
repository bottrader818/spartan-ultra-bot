import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from core.strategy_engine.base_strategy import BaseStrategy
from core.strategy_engine import SignalDirection
from core.utils.technical_indicators import calculate_rsi, calculate_dtw_distance

@dataclass
class MeanReversionSignal:
    symbol: str
    direction: SignalDirection
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    rsi_value: float
    dtw_distance: float
    sma_deviation: float
    volume_ratio: float

class EnhancedDTWMeanReversionStrategy(BaseStrategy):
    """
    Enhanced Mean Reversion Strategy with:
    - Dynamic Time Warping (DTW) for pattern recognition
    - Multi-factor confirmation system
    - Adaptive position sizing
    - Comprehensive risk management
    - Volume confirmation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "dtw_mean_reversion"
        self.parameters = self._validate_parameters(config.get('strategies', {}).get(self.name, {}))
        self.required_bars = max(50, self.parameters['sma_period'] * 3)
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Initialize performance tracking metrics"""
        self.metrics = {
            'signals_generated': 0,
            'win_rate': 0.0,
            'avg_holding_period': pd.Timedelta(0),
            'recent_signals': []
        }

    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize strategy parameters"""
        return {
            'sma_period': max(10, min(params.get('sma_period', 20), 50)),
            'rsi_period': max(5, min(params.get('rsi_period', 14), 30)),
            'rsi_oversold': max(10, min(params.get('rsi_oversold', 30), 40)),
            'rsi_overbought': min(90, max(params.get('rsi_overbought', 70), 60)),
            'dtw_threshold': max(1.0, min(params.get('dtw_threshold', 5.0), 10.0)),
            'deviation_threshold': max(0.5, min(params.get('deviation_threshold', 2.0), 5.0)),
            'min_volume_ratio': max(1.0, min(params.get('min_volume_ratio', 1.2), 3.0)),
            'risk_reward_ratio': max(1.5, min(params.get('risk_reward_ratio', 2.0), 5.0)),
            'max_position_pct': min(0.2, max(params.get('max_position_pct', 0.1), 0.01))
        }

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, MeanReversionSignal]:
        """
        Generate mean reversion signals with comprehensive validation
        Args:
            market_data: Dictionary of symbol to DataFrame with OHLCV data
        Returns:
            Dictionary of signals keyed by symbol
        """
        signals = {}
        
        for symbol, data in market_data.items():
            if len(data) < self.required_bars:
                continue
                
            try:
                signal = self._generate_signal(symbol, data)
                if signal:
                    signals[symbol] = signal
                    self._update_metrics(symbol, signal.direction)
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
                
        return signals

    def _generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[MeanReversionSignal]:
        """Generate a single signal with multiple confirmation checks"""
        close = data['close']
        volume = data['volume']
        
        # Calculate indicators
        sma = close.rolling(window=self.parameters['sma_period']).mean()
        rsi = calculate_rsi(close, self.parameters['rsi_period'])
        dtw_distance = calculate_dtw_distance(close.values[-self.parameters['sma_period']:], 
                                            sma.values[-self.parameters['sma_period']:])
        
        # Calculate additional metrics
        current_deviation = (close.iloc[-1] - sma.iloc[-1]) / sma.iloc[-1] * 100
        volume_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
        
        # Generate signal direction
        direction = None
        if (rsi.iloc[-1] < self.parameters['rsi_oversold'] and 
            dtw_distance > self.parameters['dtw_threshold'] and 
            current_deviation < -self.parameters['deviation_threshold'] and
            volume_ratio > self.parameters['min_volume_ratio']):
            direction = SignalDirection.LONG
        elif (rsi.iloc[-1] > self.parameters['rsi_overbought'] and 
              dtw_distance > self.parameters['dtw_threshold'] and 
              current_deviation > self.parameters['deviation_threshold'] and
              volume_ratio > self.parameters['min_volume_ratio']):
            direction = SignalDirection.SHORT
            
        if not direction:
            return None
            
        # Calculate risk parameters
        atr = self.calculate_atr(data['high'], data['low'], close)
        entry_price = close.iloc[-1]
        
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - (atr.iloc[-1] * 1.5)
            take_profit = entry_price + (atr.iloc[-1] * self.parameters['risk_reward_ratio'] * 1.5)
        else:
            stop_loss = entry_price + (atr.iloc[-1] * 1.5)
            take_profit = entry_price - (atr.iloc[-1] * self.parameters['risk_reward_ratio'] * 1.5)
        
        # Calculate confidence score (0-1)
        confidence = self._calculate_confidence(
            rsi.iloc[-1],
            dtw_distance,
            current_deviation,
            volume_ratio
        )
        
        return MeanReversionSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rsi_value=rsi.iloc[-1],
            dtw_distance=dtw_distance,
            sma_deviation=current_deviation,
            volume_ratio=volume_ratio
        )

    def _calculate_confidence(self, 
                            rsi: float, 
                            dtw_distance: float,
                            deviation: float,
                            volume_ratio: float) -> float:
        """Calculate signal confidence score (0-1)"""
        # RSI component (0-0.4 weight)
        rsi_component = 0.4 * (1 - min(
            abs(rsi - 30) / 30 if rsi < 50 else 
            abs(rsi - 70) / 30
        ))
        
        # DTW component (0-0.3 weight)
        dtw_component = 0.3 * min(dtw_distance / self.parameters['dtw_threshold'], 1.0)
        
        # Deviation component (0-0.2 weight)
        dev_component = 0.2 * min(
            abs(deviation) / self.parameters['deviation_threshold'], 1.0
        )
        
        # Volume component (0-0.1 weight)
        vol_component = 0.1 * min(
            (volume_ratio - 1) / (self.parameters['min_volume_ratio'] - 1), 1.0
        )
        
        return min(rsi_component + dtw_component + dev_component + vol_component, 1.0)

    def _update_metrics(self, symbol: str, direction: SignalDirection) -> None:
        """Update performance tracking metrics"""
        self.metrics['signals_generated'] += 1
        self.metrics['recent_signals'].append({
            'symbol': symbol,
            'time': pd.Timestamp.now(),
            'direction': direction,
            'confidence': None  # Will be updated after trade closes
        })
        
        # Keep only last 100 signals
        if len(self.metrics['recent_signals']) > 100:
            self.metrics['recent_signals'] = self.metrics['recent_signals'][-100:]

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
