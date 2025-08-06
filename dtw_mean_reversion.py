import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from datetime import datetime, timedelta
from dataclasses import asdict
import json
from .base_strategy import BaseStrategy, TradingSignal, SignalDirection
from .utils import calculate_atr, calculate_sma, calculate_zscore, calculate_rolling_volatility

class DTWMeanReversionStrategy(BaseStrategy):
    """
    Enhanced Dynamic Time Warping Mean Reversion Strategy
    
    Features:
    - DTW pattern recognition with adaptive thresholds
    - Multi-timeframe confirmation system
    - Volatility-adjusted position sizing
    - Dynamic risk management
    - Performance analytics
    - State persistence
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Enhanced configuration with validation
        self.lookback_window = self._validate_param(config, 'lookback_window', 14, 5, 50)
        self.zscore_threshold = self._validate_param(config, 'zscore_threshold', 2.0, 1.0, 3.0)
        self.atr_period = self._validate_param(config, 'atr_period', 14, 5, 30)
        self.sma_period = self._validate_param(config, 'sma_period', 20, 10, 50)
        self.max_position_size = self._validate_param(config, 'max_position_size', 0.1, 0.01, 0.3)
        self.risk_per_trade = self._validate_param(config, 'risk_per_trade', 0.01, 0.001, 0.05)
        self.volatility_lookback = config.get('volatility_lookback', 30)
        self.min_volume_ratio = config.get('min_volume_ratio', 1.5)
        
        # Enhanced state management
        self._price_history = {}
        self._volume_history = {}
        self._signal_history = []
        self._performance_metrics = {
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'avg_holding_period': timedelta(0),
            'sharpe_ratio': 0.0
        }
        
        # Multi-timeframe analysis
        self.timeframes = ['1h', '4h', '1d']  # Supported timeframes
        
        self.logger.info(f"Initialized enhanced DTWMeanReversionStrategy with config: {config}")

    def _validate_param(self, config: Dict, param: str, default: float, 
                       min_val: float, max_val: float) -> float:
        """Validate and constrain configuration parameters"""
        value = config.get(param, default)
        return max(min_val, min(max_val, value))

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> List[TradingSignal]:
        """
        Generate signals with multi-timeframe confirmation
        
        Args:
            data: Nested dictionary of {symbol: {timeframe: DataFrame}}
            
        Returns:
            List of validated trading signals
        """
        signals = []
        
        for symbol, timeframe_data in data.items():
            try:
                # Primary analysis on 1h timeframe
                primary_df = timeframe_data.get('1h')
                if primary_df is None or len(primary_df) < self.lookback_window:
                    continue
                
                # Update market data
                self._update_market_data(symbol, primary_df)
                
                # Check volume filter
                if not self._passes_volume_filter(symbol):
                    continue
                
                # Multi-timeframe confirmation
                confirmation = self._check_multi_timeframe_confirmation(symbol, timeframe_data)
                if not confirmation['confirmed']:
                    continue
                
                # Generate signal if conditions met
                signal = self._generate_signal(symbol, primary_df, confirmation)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
        
        return signals

    def _update_market_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Update price and volume history"""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            self._volume_history[symbol] = []
            
        self._price_history[symbol].extend(df['close'].tolist())
        self._volume_history[symbol].extend(df['volume'].tolist())
        
        # Maintain rolling window
        max_window = max(self.lookback_window, self.sma_period, self.volatility_lookback) * 2
        self._price_history[symbol] = self._price_history[symbol][-max_window:]
        self._volume_history[symbol] = self._volume_history[symbol][-max_window:]

    def _passes_volume_filter(self, symbol: str) -> bool:
        """Check if volume meets minimum ratio requirement"""
        if len(self._volume_history.get(symbol, [])) < 20:
            return False
            
        current_volume = self._volume_history[symbol][-1]
        avg_volume = np.mean(self._volume_history[symbol][-20:-1])
        return current_volume >= avg_volume * self.min_volume_ratio

    def _check_multi_timeframe_confirmation(self, symbol: str, 
                                          timeframe_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Check for signal confirmation across multiple timeframes
        
        Returns:
            Dictionary with confirmation status and analysis details
        """
        confirmation = {
            'confirmed': False,
            'timeframes_analyzed': 0,
            'matching_timeframes': 0,
            'details': {}
        }
        
        for tf in self.timeframes:
            df = timeframe_data.get(tf)
            if df is None or len(df) < self.lookback_window:
                continue
                
            confirmation['timeframes_analyzed'] += 1
            analysis = self._analyze_timeframe(symbol, df)
            confirmation['details'][tf] = analysis
            
            if analysis['signal_present']:
                confirmation['matching_timeframes'] += 1
        
        # Require confirmation from at least 2 timeframes
        confirmation['confirmed'] = confirmation['matching_timeframes'] >= 2
        return confirmation

    def _analyze_timeframe(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Analyze a single timeframe for mean reversion conditions"""
        prices = df['close'].values
        if len(prices) < self.lookback_window:
            return {'signal_present': False}
        
        # Calculate indicators
        sma = calculate_sma(prices, self.sma_period)[-1]
        atr = calculate_atr(df['high'], df['low'], df['close'], self.atr_period)[-1]
        zscore = calculate_zscore(prices, self.sma_period)[-1]
        volatility = calculate_rolling_volatility(prices, self.volatility_lookback)[-1]
        
        # DTW analysis
        recent_pattern = prices[-self.lookback_window:]
        historical_mean = np.mean(prices[:-self.lookback_window])
        reference_pattern = [historical_mean] * self.lookback_window
        distance, _ = fastdtw(recent_pattern, reference_pattern, dist=euclidean)
        normalized_distance = distance / self.lookback_window
        
        # Signal conditions
        signal_present = (
            abs(zscore) > self.zscore_threshold and 
            normalized_distance > volatility * 0.7 and
            abs(prices[-1] - sma) > atr * 0.5
        )
        
        return {
            'signal_present': signal_present,
            'zscore': zscore,
            'sma': sma,
            'atr': atr,
            'volatility': volatility,
            'dtw_distance': normalized_distance,
            'price': prices[-1]
        }

    def _generate_signal(self, symbol: str, df: pd.DataFrame, 
                        confirmation: Dict) -> Optional[TradingSignal]:
        """Generate a validated trading signal"""
        analysis = confirmation['details']['1h']  # Use primary timeframe for signal generation
        current_price = analysis['price']
        zscore = analysis['zscore']
        
        direction = SignalDirection.LONG if zscore < 0 else SignalDirection.SHORT
        confidence = self._calculate_confidence(analysis, confirmation)
        
        # Risk management parameters
        stop_loss, take_profit = self._calculate_exit_levels(
            direction, current_price, analysis['atr'], analysis['volatility'])
        
        position_size = self._calculate_position_size(
            current_price, stop_loss, analysis['volatility'])
        
        # Create signal
        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            size=position_size,
            rationale=json.dumps({
                'analysis': analysis,
                'confirmation': confirmation,
                'strategy': 'DTW Mean Reversion'
            }),
            metadata={
                'timeframes_analyzed': confirmation['timeframes_analyzed'],
                'matching_timeframes': confirmation['matching_timeframes']
            }
        )
        
        # Update performance tracking
        self._update_signal_history(signal)
        return signal

    def _calculate_confidence(self, analysis: Dict, confirmation: Dict) -> float:
        """Calculate dynamic signal confidence score"""
        base_confidence = min(1.0, abs(analysis['zscore']) / self.zscore_threshold)
        timeframe_factor = confirmation['matching_timeframes'] / len(self.timeframes)
        volatility_factor = 1.0 - min(1.0, analysis['volatility'] / 0.1)  # Normalize volatility
        
        return base_confidence * 0.5 + timeframe_factor * 0.3 + volatility_factor * 0.2

    def _calculate_exit_levels(self, direction: SignalDirection, entry_price: float, 
                              atr: float, volatility: float) -> Tuple[float, float]:
        """Calculate volatility-adjusted exit levels"""
        # Base levels
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - atr * 2.5
            take_profit = entry_price + atr * 4
        else:
            stop_loss = entry_price + atr * 2.5
            take_profit = entry_price - atr * 4
        
        # Adjust for volatility regime
        if volatility > 0.15:  # High volatility
            stop_loss *= 1.3
            take_profit *= 0.8
        elif volatility < 0.05:  # Low volatility
            stop_loss *= 0.8
            take_profit *= 1.2
            
        return stop_loss, take_profit

    def _calculate_position_size(self, entry_price: float, stop_loss: float, 
                               volatility: float) -> float:
        """Calculate volatility-adjusted position size"""
        risk_amount = self.config.capital * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            return 0.0
            
        base_size = risk_amount / risk_per_unit
        
        # Adjust for volatility
        size = base_size * (0.1 / max(0.01, volatility))  # Normalize to 10% volatility
        
        # Apply maximum position size
        max_size = (self.config.capital * self.max_position_size) / entry_price
        return min(size, max_size)

    def _update_signal_history(self, signal: TradingSignal) -> None:
        """Update signal history and performance metrics"""
        self._signal_history.append(signal)
        self._performance_metrics['total_signals'] += 1
        
        self.logger.info(
            f"Generated {signal.direction.name} signal for {signal.symbol} at {signal.entry_price:.2f} "
            f"(Confidence: {signal.confidence:.2%}, Size: {signal.size:.2f})"
        )

    def save_state(self) -> Dict:
        """Save current strategy state for persistence"""
        return {
            'price_history': self._price_history,
            'performance_metrics': self._performance_metrics,
            'last_signals': [asdict(s) for s in self._signal_history[-100:]],  # Save last 100 signals
            'config': {
                'lookback_window': self.lookback_window,
                'zscore_threshold': self.zscore_threshold,
                'atr_period': self.atr_period,
                'sma_period': self.sma_period,
                'max_position_size': self.max_position_size,
                'risk_per_trade': self.risk_per_trade
            }
        }

    def load_state(self, state: Dict) -> None:
        """Load strategy state from saved data"""
        self._price_history = state.get('price_history', {})
        self._performance_metrics = state.get('performance_metrics', {})
        
        # Reconstruct signal history
        self._signal_history = [
            TradingSignal(**s) for s in state.get('last_signals', [])
        ]
        
        # Update configuration
        config = state.get('config', {})
        self.lookback_window = config.get('lookback_window', self.lookback_window)
        self.zscore_threshold = config.get('zscore_threshold', self.zscore_threshold)
        self.atr_period = config.get('atr_period', self.atr_period)
        self.sma_period = config.get('sma_period', self.sma_period)
        self.max_position_size = config.get('max_position_size', self.max_position_size)
        self.risk_per_trade = config.get('risk_per_trade', self.risk_per_trade)

