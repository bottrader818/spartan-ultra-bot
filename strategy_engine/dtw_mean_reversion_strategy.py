import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from .base_strategy import BaseStrategy, TradingSignal, SignalDirection
from .utils import calculate_atr, calculate_sma, calculate_zscore

class DTWMeanReversionStrategy(BaseStrategy):
    """
    Dynamic Time Warping (DTW) Mean Reversion Strategy
    
    A sophisticated mean reversion strategy that uses DTW to identify similar price patterns
    and statistical methods to detect mean reversion opportunities with:
    - DTW-based pattern matching
    - Volatility-adjusted position sizing
    - Z-score based entry/exit signals
    - Multi-timeframe confirmation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DTW Mean Reversion Strategy
        
        Args:
            config: Configuration dictionary containing:
                - lookback_window: DTW comparison window size
                - zscore_threshold: Z-score threshold for signals
                - atr_period: Period for ATR calculation
                - sma_period: Period for SMA calculation
                - max_position_size: Maximum position size as % of capital
                - risk_per_trade: Risk per trade as % of capital
        """
        super().__init__(config)
        
        # Strategy parameters with defaults
        self.lookback_window = config.get('lookback_window', 14)
        self.zscore_threshold = config.get('zscore_threshold', 2.0)
        self.atr_period = config.get('atr_period', 14)
        self.sma_period = config.get('sma_period', 20)
        self.max_position_size = config.get('max_position_size', 0.1)
        self.risk_per_trade = config.get('risk_per_trade', 0.01)
        
        # State variables
        self._price_history = {}
        self._signal_history = []
        self._last_signal = None
        
        # Initialize indicators
        self._indicators = {
            'sma': {},
            'atr': {},
            'zscore': {}
        }
        
        self.logger.info(f"Initialized DTWMeanReversionStrategy with config: {config}")

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """
        Generate trading signals based on DTW pattern matching and mean reversion
        
        Args:
            data: Dictionary of market data DataFrames (keyed by symbol)
                  Each DataFrame should contain 'close', 'high', 'low' columns
            
        Returns:
            List of TradingSignal objects
        """
        signals = []
        
        for symbol, df in data.items():
            # Update price history
            self._update_price_history(symbol, df['close'])
            
            # Calculate indicators if we have enough data
            if len(self._price_history.get(symbol, [])) >= max(self.lookback_window, self.sma_period):
                try:
                    signals.extend(self._analyze_symbol(symbol, df))
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {str(e)}", exc_info=True)
        
        return signals

    def _update_price_history(self, symbol: str, prices: pd.Series) -> None:
        """Maintain rolling window of prices for each symbol"""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        
        # Append new prices and maintain lookback window
        self._price_history[symbol].extend(prices.tolist())
        if len(self._price_history[symbol]) > self.lookback_window * 2:
            self._price_history[symbol] = self._price_history[symbol][-self.lookback_window * 2:]

    def _analyze_symbol(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Analyze a single symbol for trading opportunities"""
        signals = []
        prices = self._price_history[symbol]
        current_price = prices[-1]
        
        # Calculate indicators
        sma = calculate_sma(prices, self.sma_period)[-1]
        atr = calculate_atr(data['high'], data['low'], data['close'], self.atr_period)[-1]
        zscore = calculate_zscore(prices, self.sma_period)[-1]
        
        # Store indicators for reference
        self._indicators['sma'][symbol] = sma
        self._indicators['atr'][symbol] = atr
        self._indicators['zscore'][symbol] = zscore
        
        # Calculate DTW distance to recent price pattern
        recent_pattern = prices[-self.lookback_window:]
        historical_mean = np.mean(prices[:-self.lookback_window])
        reference_pattern = [historical_mean] * self.lookback_window
        
        distance, _ = fastdtw(recent_pattern, reference_pattern, dist=euclidean)
        normalized_distance = distance / self.lookback_window
        
        # Generate signals based on conditions
        if (abs(zscore) > self.zscore_threshold and 
            normalized_distance > np.std(prices) * 0.5):
            
            direction = SignalDirection.LONG if zscore < 0 else SignalDirection.SHORT
            confidence = min(1.0, abs(zscore) / self.zscore_threshold)
            
            # Calculate stop loss and take profit
            if direction == SignalDirection.LONG:
                stop_loss = current_price - atr * 2
                take_profit = current_price + atr * 3
            else:
                stop_loss = current_price + atr * 2
                take_profit = current_price - atr * 3
            
            # Calculate position size
            position_size = self.calculate_position_size(
                entry_price=current_price,
                stop_loss=stop_loss,
                risk_pct=self.risk_per_trade
            )
            
            # Cap position size
            max_size = (self.config.capital * self.max_position_size) / current_price
            position_size = min(position_size, max_size)
            
            signal = TradingSignal(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                size=position_size,
                rationale=f"DTW Mean Reversion: zscore={zscore:.2f}, distance={normalized_distance:.2f}"
            )
            
            signals.append(signal)
            self._signal_history.append(signal)
            self._last_signal = signal
            
            self.logger.info(
                f"Generated {direction.name} signal for {symbol} at {current_price:.2f} "
                f"(zscore: {zscore:.2f}, confidence: {confidence:.2%})"
            )
        
        return signals

    def get_current_indicators(self, symbol: str) -> Dict[str, float]:
        """Get current indicator values for a symbol"""
        return {
            'sma': self._indicators['sma'].get(symbol),
            'atr': self._indicators['atr'].get(symbol),
            'zscore': self._indicators['zscore'].get(symbol)
        }

    def get_signal_history(self) -> List[TradingSignal]:
        """Get history of all generated signals"""
        return self._signal_history.copy()

    def get_last_signal(self) -> Optional[TradingSignal]:
        """Get the most recent trading signal"""
        return self._last_signal
