import random
from typing import Literal, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum, auto
import logging
import numpy as np
from scipy.stats import norm
import pandas as pd

class SlippageModelType(Enum):
    """Types of slippage models available"""
    CONSTANT = auto()
    PROPORTIONAL = auto()
    VOLATILITY_ADJUSTED = auto()
    LIQUIDITY_ADJUSTED = auto()
    MARKET_IMPACT = auto()

class OrderType(Enum):
    """Supported order types"""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()

@dataclass
class SlippageResult:
    """Detailed slippage calculation result"""
    original_price: float
    adjusted_price: float
    slippage_amount: float
    slippage_pct: float
    model_used: str
    confidence_interval: Optional[tuple[float, float]] = None
    liquidity_factor: Optional[float] = None
    volatility_factor: Optional[float] = None

class SlippageEngine:
    """
    Professional-Grade Slippage Engine with:
    - Multiple slippage models
    - Volatility-adjusted calculations
    - Liquidity awareness
    - Market impact modeling
    - Confidence intervals
    - Comprehensive statistics
    """
    
    def __init__(self,
                 base_slippage_rate: float = 0.001,
                 model_type: SlippageModelType = SlippageModelType.PROPORTIONAL,
                 volatility_window: int = 20,
                 liquidity_threshold: float = 1.0):
        """
        Initialize slippage engine
        
        Args:
            base_slippage_rate: Base slippage rate (0.001 = 0.1%)
            model_type: Type of slippage model to use
            volatility_window: Window for volatility calculation (in periods)
            liquidity_threshold: Threshold for liquidity adjustment factor
        """
        self.base_slippage_rate = max(0.0, base_slippage_rate)
        self.model_type = model_type
        self.volatility_window = max(5, volatility_window)
        self.liquidity_threshold = max(0.1, liquidity_threshold)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # State tracking
        self.price_history = []
        self.volume_history = []
        self._model_parameters = self._initialize_model_parameters()
        
        # Statistics
        self.slippage_stats = {
            'total_calculations': 0,
            'total_slippage': 0.0,
            'max_slippage': 0.0,
            'slippage_by_side': {'buy': 0.0, 'sell': 0.0},
            'slippage_by_order_type': {ot: 0.0 for ot in OrderType}
        }
    
    def _initialize_model_parameters(self) -> Dict:
        """Initialize parameters for different slippage models"""
        return {
            SlippageModelType.CONSTANT: {
                'fixed_adjustment': self.base_slippage_rate
            },
            SlippageModelType.PROPORTIONAL: {
                'rate': self.base_slippage_rate,
                'min_slippage': 0.0001,  # Minimum 0.01%
                'max_slippage': 0.05     # Maximum 5%
            },
            SlippageModelType.VOLATILITY_ADJUSTED: {
                'base_rate': self.base_slippage_rate,
                'volatility_multiplier': 1.5,
                'min_volatility': 0.005,  # 0.5%
                'max_volatility': 0.20    # 20%
            },
            SlippageModelType.LIQUIDITY_ADJUSTED: {
                'base_rate': self.base_slippage_rate,
                'liquidity_power': 0.5,  # Square root relationship
                'max_liquidity_impact': 3.0
            },
            SlippageModelType.MARKET_IMPACT: {
                'base_rate': self.base_slippage_rate,
                'impact_factor': 0.1,
                'depth_factor': 0.01
            }
        }
    
    def update_market_data(self, price: float, volume: float) -> None:
        """Update engine with latest market data for calculations"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Maintain windowed history
        if len(self.price_history) > self.volatility_window * 2:
            self.price_history = self.price_history[-self.volatility_window * 2:]
            self.volume_history = self.volume_history[-self.volatility_window * 2:]
    
    def _calculate_volatility(self) -> float:
        """Calculate current volatility based on price history"""
        if len(self.price_history) < self.volatility_window:
            return self._model_parameters[SlippageModelType.VOLATILITY_ADJUSTED]['min_volatility']
        
        returns = np.diff(np.log(self.price_history[-self.volatility_window:]))
        return np.std(returns)
    
    def _calculate_liquidity_factor(self) -> float:
        """Calculate liquidity adjustment factor"""
        if len(self.volume_history) < self.volatility_window:
            return 1.0
        
        avg_volume = np.mean(self.volume_history[-self.volatility_window:])
        params = self._model_parameters[SlippageModelType.LIQUIDITY_ADJUSTED]
        
        # Normalize volume against threshold
        normalized_volume = avg_volume / self.liquidity_threshold
        liquidity_factor = min(
            params['max_liquidity_impact'],
            max(1.0, normalized_volume ** params['liquidity_power'])
        )
        
        return 1.0 / liquidity_factor
    
    def _calculate_market_impact(self, order_size: float, side: str) -> float:
        """Estimate market impact of an order"""
        if not self.price_history or not self.volume_history:
            return self.base_slippage_rate
        
        params = self._model_parameters[SlippageModelType.MARKET_IMPACT]
        current_price = self.price_history[-1]
        avg_volume = np.mean(self.volume_history[-self.volatility_window:])
        
        # Calculate size ratio (order size / average volume)
        size_ratio = order_size / avg_volume
        
        # Market impact formula
        impact = (params['impact_factor'] * size_ratio ** params['depth_factor'])
        
        # Adjust for order side (buys typically have more impact)
        if side == 'buy':
            impact *= 1.2
        
        return impact
    
    def apply_slippage(self,
                      price: float,
                      side: Literal['buy', 'sell'],
                      order_type: OrderType = OrderType.MARKET,
                      order_size: Optional[float] = None,
                      confidence_level: float = 0.95) -> SlippageResult:
        """
        Apply slippage to a price based on current market conditions
        
        Args:
            price: Original price
            side: 'buy' or 'sell'
            order_type: Type of order
            order_size: Size of order (for market impact models)
            confidence_level: Confidence interval for slippage range
            
        Returns:
            SlippageResult with detailed slippage information
        """
        if price <= 0:
            raise ValueError("Price must be positive")
            
        self.slippage_stats['total_calculations'] += 1
        
        # Select model and calculate slippage
        if self.model_type == SlippageModelType.CONSTANT:
            slippage_pct = self._model_parameters[SlippageModelType.CONSTANT]['fixed_adjustment']
        elif self.model_type == SlippageModelType.PROPORTIONAL:
            params = self._model_parameters[SlippageModelType.PROPORTIONAL]
            slippage_pct = random.uniform(
                params['min_slippage'],
                min(params['max_slippage'], params['rate'])
            )
        elif self.model_type == SlippageModelType.VOLATILITY_ADJUSTED:
            vol = self._calculate_volatility()
            params = self._model_parameters[SlippageModelType.VOLATILITY_ADJUSTED]
            normalized_vol = max(
                params['min_volatility'],
                min(params['max_volatility'], vol)
            )
            slippage_pct = params['base_rate'] * (normalized_vol * params['volatility_multiplier'])
        elif self.model_type == SlippageModelType.LIQUIDITY_ADJUSTED:
            liquidity_factor = self._calculate_liquidity_factor()
            slippage_pct = self.base_slippage_rate * liquidity_factor
        elif self.model_type == SlippageModelType.MARKET_IMPACT:
            if order_size is None:
                order_size = 1.0  # Default to nominal size
            slippage_pct = self._calculate_market_impact(order_size, side)
        else:
            slippage_pct = self.base_slippage_rate
        
        # Apply slippage based on side
        slippage_amount = price * slippage_pct
        adjusted_price = price + slippage_amount if side == 'buy' else price - slippage_amount
        
        # Calculate confidence interval
        if self.model_type in [SlippageModelType.VOLATILITY_ADJUSTED, SlippageModelType.MARKET_IMPACT]:
            std_dev = slippage_pct * 0.5  # Estimate standard deviation
            z_score = norm.ppf(1 - (1 - confidence_level) / 2)
            ci_low = adjusted_price - (price * std_dev * z_score)
            ci_high = adjusted_price + (price * std_dev * z_score)
            confidence_interval = (ci_low, ci_high)
        else:
            confidence_interval = None
        
        # Update statistics
        self._update_stats(slippage_amount, side, order_type)
        
        return SlippageResult(
            original_price=price,
            adjusted_price=adjusted_price,
            slippage_amount=slippage_amount,
            slippage_pct=slippage_pct,
            model_used=self.model_type.name,
            confidence_interval=confidence_interval,
            liquidity_factor=(
                self._calculate_liquidity_factor() 
                if self.model_type == SlippageModelType.LIQUIDITY_ADJUSTED 
                else None
            ),
            volatility_factor=(
                self._calculate_volatility()
                if self.model_type == SlippageModelType.VOLATILITY_ADJUSTED
                else None
            )
        )
    
    def _update_stats(self, slippage_amount: float, side: str, order_type: OrderType) -> None:
        """Update slippage statistics"""
        abs_slippage = abs(slippage_amount)
        
        self.slippage_stats['total_slippage'] += abs_slippage
        self.slippage_stats['max_slippage'] = max(
            self.slippage_stats['max_slippage'],
            abs_slippage
        )
        self.slippage_stats['slippage_by_side'][side] += abs_slippage
        self.slippage_stats['slippage_by_order_type'][order_type] += abs_slippage
    
    def get_slippage_stats(self) -> pd.DataFrame:
        """Get slippage statistics as a DataFrame"""
        stats = self.slippage_stats.copy()
        
        # Calculate averages
        if stats['total_calculations'] > 0:
            stats['avg_slippage'] = stats['total_slippage'] / stats['total_calculations']
        else:
            stats['avg_slippage'] = 0.0
            
        return pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    
    def reset_stats(self) -> None:
        """Reset all statistics counters"""
        self.slippage_stats = {
            'total_calculations': 0,
            'total_slippage': 0.0,
            'max_slippage': 0.0,
            'slippage_by_side': {'buy': 0.0, 'sell': 0.0},
            'slippage_by_order_type': {ot: 0.0 for ot in OrderType}
        }
