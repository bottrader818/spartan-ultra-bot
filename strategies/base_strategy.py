from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging
import pandas as pd
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

class SignalDirection(Enum):
    """Direction of trading signals"""
    LONG = auto()
    SHORT = auto()
    EXIT = auto()
    HOLD = auto()

class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING = auto()
    MEAN_REVERTING = auto()
    VOLATILE = auto()
    RANGING = auto()
    BREAKOUT = auto()

@dataclass
class TradingSignal:
    """Structured trading signal with metadata"""
    symbol: str
    direction: SignalDirection
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    size: float = 1.0  # Position size as fraction of capital
    validity: Optional[timedelta] = None  # How long the signal is valid
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyConfig:
    """Configuration for strategy execution"""
    capital: float = 100000.0
    risk_per_trade: float = 0.01  # 1% of capital
    max_position_size: float = 0.1  # 10% of capital
    slippage: float = 0.0005  # 0.05%
    commission: float = 0.0002  # 0.02%
    data_resolution: str = '1h'  # Default data resolution
    warmup_period: int = 100  # Bars needed for initialization
    enabled: bool = True

class BaseStrategy(ABC):
    """
    Professional-Grade Base Strategy Class with:
    - Standardized signal generation interface
    - Risk management integration
    - Performance tracking
    - Market regime awareness
    - Configuration management
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration
        
        Args:
            config: Dictionary containing strategy configuration
        """
        self.config = StrategyConfig(**config)
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"strategy.{self.name}")
        
        # State tracking
        self._signals_generated = 0
        self._positions = {}
        self._market_regime = None
        self._last_data_timestamp = None
        
        # Performance metrics
        self._performance_metrics = {
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'last_signal_time': None
        }
        
        # Initialize indicators and state
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize strategy state and indicators"""
        self._indicators = {}
        self._signals = []
        self._historical_signals = []
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, TradingSignal]:
        """
        Generate trading signals from market data
        
        Args:
            data: Dictionary of market data DataFrames (keyed by symbol)
            
        Returns:
            Dictionary of trading signals (keyed by symbol)
        """
        pass
    
    def analyze_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Analyze current market regime
        
        Args:
            data: Market data for analysis
            
        Returns:
            MarketRegime enum value
        """
        # Default implementation - can be overridden by subclasses
        volatility = data['close'].pct_change().std()
        trend_strength = self._calculate_trend_strength(data)
        
        if volatility > 0.02:  # 2% standard deviation
            return MarketRegime.VOLATILE
        elif trend_strength > 0.5:
            return MarketRegime.TRENDING
        else:
            return MarketRegime.MEAN_REVERTING
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength (0-1 scale)"""
        short_ma = data['close'].rolling(20).mean()
        long_ma = data['close'].rolling(50).mean()
        trend_diff = (short_ma - long_ma).abs() / data['close']
        return trend_diff.mean()
    
    def calculate_position_size(self, 
                              entry_price: float, 
                              stop_loss: float, 
                              risk_pct: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            entry_price: Intended entry price
            stop_loss: Stop loss price
            risk_pct: Percentage of capital to risk (overrides config)
            
        Returns:
            Position size in units
        """
        risk_amount = self.config.capital * (risk_pct or self.config.risk_per_trade)
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            return 0.0
            
        size = risk_amount / risk_per_unit
        max_size = (self.config.capital * self.config.max_position_size) / entry_price
        
        return min(size, max_size)
    
    def update_performance(self, 
                         symbol: str, 
                         pnl: float, 
                         exit_reason: Optional[str] = None) -> None:
        """
        Update strategy performance metrics
        
        Args:
            symbol: Symbol for the position
            pnl: Realized PnL for the position
            exit_reason: Reason for position exit
        """
        with self._lock:
            self._performance_metrics['total_pnl'] += pnl
            
            if pnl > 0:
                self._performance_metrics['winning_signals'] += 1
            else:
                self._performance_metrics['losing_signals'] += 1
                
            self._performance_metrics['win_rate'] = (
                self._performance_metrics['winning_signals'] / 
                max(1, self._performance_metrics['winning_signals'] + 
                    self._performance_metrics['losing_signals'])
            )
            
            # Update max drawdown
            if pnl < 0 and abs(pnl) > self._performance_metrics['max_drawdown']:
                self._performance_metrics['max_drawdown'] = abs(pnl)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self._performance_metrics.copy()
    
    def get_historical_signals(self, 
                             lookback: Optional[timedelta] = None) -> List[TradingSignal]:
        """
        Get historical trading signals
        
        Args:
            lookback: Optional time period to look back
            
        Returns:
            List of trading signals within lookback period
        """
        if lookback is None:
            return self._historical_signals.copy()
            
        cutoff = datetime.utcnow() - lookback
        return [s for s in self._historical_signals if s.timestamp >= cutoff]
    
    def save_state(self) -> Dict[str, Any]:
        """Save current strategy state for persistence"""
        return {
            'performance_metrics': self._performance_metrics,
            'positions': self._positions,
            'market_regime': self._market_regime.name if self._market_regime else None,
            'last_data_timestamp': self._last_data_timestamp.isoformat() if self._last_data_timestamp else None
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load strategy state from saved data"""
        self._performance_metrics = state.get('performance_metrics', {})
        self._positions = state.get('positions', {})
        
        regime = state.get('market_regime')
        if regime:
            self._market_regime = MarketRegime[regime]
        
        timestamp = state.get('last_data_timestamp')
        if timestamp:
            self._last_data_timestamp = datetime.fromisoformat(timestamp)
    
    def __str__(self) -> str:
        """String representation of strategy"""
        return f"{self.name} (Capital: {self.config.capital:,.2f}, Win Rate: {self._performance_metrics['win_rate']:.2%})"
