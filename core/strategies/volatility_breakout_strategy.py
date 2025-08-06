import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, time
from core.strategy_engine.base_strategy import BaseStrategy
from core.strategy_engine import SignalDirection
from core.risk_management.position_sizer import VolatilityPositionSizer
from core.data_adapters import NewsAnalyzer, LiquidityAnalyzer

logger = logging.getLogger('VolatilityBreakoutStrategy')
logger.setLevel(logging.INFO)

@dataclass
class BreakoutSignal:
    symbol: str
    direction: SignalDirection
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    volatility_ratio: float
    timeframe: str
    position_size: float
    confirmed: bool = False
    confidence: float = 0.0
    last_confirmation: Optional[datetime] = None

class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Enhanced Volatility Breakout Strategy with:
    - Multi-timeframe confirmation system
    - Adaptive volatility filters
    - Smart session detection
    - News sentiment integration
    - Liquidity screening
    - Position sizing based on account risk
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "volatility_breakout"
        self._init_dependencies(config)
        self.parameters = self._validate_parameters(config)
        self.required_bars = self._calculate_required_bars()
        self._setup_state_tracking()
        self._init_metrics()

    def _init_dependencies(self, config: Dict[str, Any]) -> None:
        """Initialize external service adapters"""
        self.position_sizer = VolatilityPositionSizer(config)
        self.news_analyzer = NewsAnalyzer(config['news_api'])
        self.liquidity_analyzer = LiquidityAnalyzer(config['liquidity_thresholds'])

    def _init_metrics(self) -> None:
        """Initialize performance tracking metrics"""
        self.metrics = {
            'signals_generated': 0,
            'signals_triggered': 0,
            'win_rate': 0.0,
            'avg_holding_period': pd.Timedelta(0),
            'recent_breakouts': []
        }

    def _setup_state_tracking(self) -> None:
        """Initialize strategy state tracking"""
        self.breakout_levels = {}
        self.last_breakouts = {}
        self.session_high_low = {}
        self.volatility_regimes = {}
        self.confirmation_status = {}

    def _calculate_required_bars(self) -> int:
        """Determine minimum bars needed for reliable calculations"""
        return max(
            100,
            self.parameters['atr_period'] * 3,
            *[self._timeframe_to_bars(tf) for tf in self.parameters['confirmation_timeframes']],
            self.parameters['volatility_lookback'] * 2
        )

    def _validate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize strategy parameters"""
        params = config['strategies'][self.name]
        
        return {
            'atr_period': self._clamp(params.get('atr_period', 14), 5, 50),
            'breakout_multiplier': self._clamp(params.get('breakout_multiplier', 1.0), 0.5, 3.0),
            'risk_pct': self._clamp(params.get('risk_pct', 1.0), 0.1, 5.0),
            'timeframe': params.get('timeframe', '1d'),
            'confirmation_timeframes': self._validate_timeframes(
                params.get('confirmation_timeframes', ['4h', '1d'])
            ),
            'session_breakout': params.get('session_breakout', True),
            'volatility_lookback': self._clamp(params.get('volatility_lookback', 20), 10, 100),
            'min_volume_ratio': self._clamp(params.get('min_volume_ratio', 1.2), 1.0, 3.0),
            'penetration_threshold': self._clamp(params.get('penetration_threshold', 0.7), 0.3, 1.5),
            'max_daily_breakouts': self._clamp(params.get('max_daily_breakouts', 3), 1, 10),
            'adaptive_multiplier': params.get('adaptive_multiplier', True),
            'require_confirmation': params.get('require_confirmation', True)
        }

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Ensure value stays within bounds"""
        return max(min_val, min(value, max_val))

    def calculate_volatility_regime(self, close: pd.Series) -> float:
        """
        Calculate current volatility regime (0.5-2.0)
        1.0 = normal volatility
        <1.0 = low volatility
        >1.0 = high volatility
        """
        short_term = close.pct_change().rolling(5).std()
        long_term = close.pct_change().rolling(50).std()
        return (short_term.iloc[-1] / (long_term.iloc[-1] + 1e-6)).clip(0.5, 2.0)

    def calculate_breakout_levels(self, 
                                high: pd.Series, 
                                low: pd.Series, 
                                close: pd.Series,
                                volatility_regime: float) -> Tuple[float, float]:
        """Calculate dynamic breakout levels based on volatility"""
        atr = self.calculate_atr(high, low, close)
        multiplier = self.parameters['breakout_multiplier']
        if self.parameters['adaptive_multiplier']:
            multiplier *= volatility_regime
            
        upper = close.iloc[-1] + (atr.iloc[-1] * multiplier)
        lower = close.iloc[-1] - (atr.iloc[-1] * multiplier)
        
        return upper, lower

    def check_liquidity(self, symbol: str, volume: pd.Series) -> bool:
        """Verify sufficient liquidity for the instrument"""
        return self.liquidity_analyzer.validate(
            symbol,
            volume.iloc[-1],
            volume.mean()
        )

    def check_news_impact(self, symbol: str) -> bool:
        """Check for significant news events that may affect volatility"""
        return self.news_analyzer.is_tradeable(symbol)

    def check_session_breakout(self, symbol: str, timeframe: str) -> bool:
        """Validate breakout during optimal session times"""
        if not self.parameters['session_breakout']:
            return True
            
        now = pd.Timestamp.now()
        if timeframe.endswith('d'):
            # Daily breakouts - only valid in first 4 hours
            return now.hour < 12
        elif timeframe.endswith('h'):
            # Intraday breakouts - avoid overnight
            return 8 <= now.hour < 20
        return True

    def check_breakout_quality(self, 
                             symbol: str, 
                             direction: SignalDirection,
                             close: pd.Series,
                             volume: pd.Series,
                             atr: pd.Series) -> bool:
        """Validate breakout meets quality thresholds"""
        # Volume check
        vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
        if vol_ratio < self.parameters['min_volume_ratio']:
            return False
            
        # Price penetration check
        penetration = (
            (close.iloc[-1] - close.iloc[-2]) / atr.iloc[-2] 
            if direction == SignalDirection.LONG else
            (close.iloc[-2] - close.iloc[-1]) / atr.iloc[-2]
        )
        if penetration < self.parameters['penetration_threshold']:
            return False
            
        return True

    def check_confirmation(self, 
                         symbol: str,
                         direction: SignalDirection,
                         market_data: Dict[str, Dict[str, pd.DataFrame]]) -> bool:
        """Check for multi-timeframe confirmation"""
        if not self.parameters['require_confirmation']:
            return True
            
        confirmed = False
        for tf in self.parameters['confirmation_timeframes']:
            if tf in market_data[symbol]:
                tf_data = market_data[symbol][tf]
                if len(tf_data) > 20:
                    ma_direction = self._get_trend_direction(tf_data)
                    if ma_direction == direction:
                        confirmed = True
                        break
        return confirmed

    def _get_trend_direction(self, data: pd.DataFrame) -> SignalDirection:
        """Determine trend direction using moving averages"""
        ma_fast = data['close'].rolling(9).mean()
        ma_slow = data['close'].rolling(21).mean()
        
        if ma_fast.iloc[-1] > ma_slow.iloc[-1]:
            return SignalDirection.LONG
        return SignalDirection.SHORT

    def generate_signals(self, 
                       market_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, BreakoutSignal]:
        """Generate volatility breakout signals with comprehensive validation"""
        signals = {}
        
        for symbol, timeframes in market_data.items():
            if self.parameters['timeframe'] not in timeframes:
                continue
                
            data = timeframes[self.parameters['timeframe']]
            if len(data) < self.required_bars:
                continue
                
            try:
                # Pre-filtering checks
                if not all([
                    self.check_liquidity(symbol, data['volume']),
                    self.check_news_impact(symbol),
                    self.check_session_breakout(symbol, self.parameters['timeframe']),
                    self._check_daily_breakout_limit(symbol)
                ]):
                    continue
                    
                # Calculate volatility regime and breakout levels
                volatility_regime = self.calculate_volatility_regime(data['close'])
                upper, lower = self.calculate_breakout_levels(
                    data['high'], data['low'], data['close'], volatility_regime
                )
                
                # Check for breakout
                direction = None
                if data['close'].iloc[-1] > upper:
                    direction = SignalDirection.LONG
                elif data['close'].iloc[-1] < lower:
                    direction = SignalDirection.SHORT
                    
                if direction and self.check_breakout_quality(
                    symbol, direction, data['close'], data['volume'], 
                    self.calculate_atr(data['high'], data['low'], data['close'])
                ):
                    # Calculate position size
                    position_size = self.position_sizer.calculate(
                        symbol,
                        data['close'].iloc[-1],
                        self.calculate_atr(data['high'], data['low'], data['close']).iloc[-1],
                        self.parameters['risk_pct']
                    )
                    
                    # Create signal
                    signals[symbol] = BreakoutSignal(
                        symbol=symbol,
                        direction=direction,
                        strength=min(volatility_regime, 1.5),
                        entry_price=data['close'].iloc[-1],
                        stop_loss=(
                            data['close'].iloc[-1] - (1.5 * atr.iloc[-1])
                            if direction == SignalDirection.LONG else
                            data['close'].iloc[-1] + (1.5 * atr.iloc[-1])
                        ),
                        take_profit=(
                            data['close'].iloc[-1] + (3 * atr.iloc[-1])
                            if direction == SignalDirection.LONG else
                            data['close'].iloc[-1] - (3 * atr.iloc[-1])
                        ),
                        atr_value=atr.iloc[-1],
                        volatility_ratio=volatility_regime,
                        timeframe=self.parameters['timeframe'],
                        position_size=position_size,
                        confirmed=self.check_confirmation(symbol, direction, market_data),
                        confidence=self._calculate_confidence(symbol, direction, market_data)
                    )
                    
                    self._update_state(symbol, direction, data['close'].iloc[-1])
                    self._update_metrics(symbol, True)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
                self._update_metrics(symbol, False)
                
        return signals

    def _check_daily_breakout_limit(self, symbol: str) -> bool:
        """Enforce maximum daily breakouts per symbol"""
        today = datetime.now().date()
        recent = [
            b for b in self.metrics['recent_breakouts']
            if b['symbol'] == symbol and b['time'].date() == today
        ]
        return len(recent) < self.parameters['max_daily_breakouts']

    def _calculate_confidence(self, 
                            symbol: str,
                            direction: SignalDirection,
                            market_data: Dict) -> float:
        """Calculate signal confidence score (0-1)"""
        # Base confidence based on volatility regime
        confidence = min(self.volatility_regimes.get(symbol, 1.0), 1.0)
        
        # Increase confidence with multi-timeframe confirmation
        if self.check_confirmation(symbol, direction, market_data):
            confidence *= 1.2
            
        # Decrease confidence if recent opposing breakouts
        if symbol in self.last_breakouts:
            last_dir = self.last_breakouts[symbol]['direction']
            if last_dir != direction:
                confidence *= 0.8
                
        return confidence.clip(0.1, 1.0)

    def _update_state(self, symbol: str, direction: SignalDirection, price: float) -> None:
        """Update internal state tracking"""
        self.last_breakouts[symbol] = {
            'time': datetime.now(),
            'direction': direction,
            'price': price
        }
        self.metrics['recent_breakouts'].append({
            'symbol': symbol,
            'time': datetime.now(),
            'direction': direction
        })

    def _update_metrics(self, symbol: str, success: bool) -> None:
        """Update performance metrics"""
        self.metrics['signals_generated'] += 1
        if success:
            self.metrics['signals_triggered'] += 1
