from typing import Tuple, Optional, Dict, List
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
import logging
from scipy.stats import norm
from collections import deque
import pandas as pd

class KellyOptimizationMode(Enum):
    """Kelly optimization approaches with dynamic adaptation"""
    FULL_KELLY = auto()
    HALF_KELLY = auto()
    QUARTER_KELLY = auto()
    FRACTIONAL = auto()
    DYNAMIC = auto()  # Automatically adjusts based on market conditions
    VOLATILITY_ADJUSTED = auto()  # Adjusts based on volatility regime

class PortfolioConstraint(Enum):
    """Types of portfolio constraints for position sizing"""
    LEVERAGE_LIMIT = auto()
    DRAWDOWN_TARGET = auto()
    VOLATILITY_TARGET = auto()
    CORRELATION_ADJUSTMENT = auto()
    LIQUIDITY_CONSTRAINT = auto()

@dataclass
class KellyOptimizationParameters:
    """Dynamic parameters for Kelly optimization"""
    win_rate_decay_factor: float = 0.99  # Exponential decay for older wins
    min_win_rate: float = 0.4  # Minimum credible win rate
    max_win_rate: float = 0.8  # Maximum credible win rate
    reward_risk_floor: float = 1.0  # Minimum reward/risk ratio
    reward_risk_cap: float = 5.0  # Maximum reward/risk ratio
    volatility_adjustment_factor: float = 0.5  # How much to reduce sizing in high vol
    drawdown_sensitivity: float = 2.0  # How aggressively to reduce on drawdowns

@dataclass
class KellyResult:
    """Comprehensive Kelly Criterion calculation result with advanced metrics"""
    fraction: float  # Recommended fraction of capital to bet
    optimal_fraction: float  # Pure Kelly fraction before adjustments
    mode: KellyOptimizationMode
    win_rate: float
    reward_risk_ratio: float
    expected_growth_rate: float
    risk_adjusted_fraction: float
    max_drawdown_estimate: float
    volatility_adjusted_fraction: float
    regime_adjusted_fraction: float
    confidence_interval: Tuple[float, float]
    portfolio_allocation: float
    component_scores: Dict[str, float]
    constraints_applied: List[PortfolioConstraint]

class KellyOptimizer:
    """
    Professional-Grade Kelly Criterion Optimizer with:
    - Dynamic parameter adaptation
    - Market regime awareness
    - Advanced risk constraints
    - Portfolio-level optimization
    - Confidence interval estimation
    - Performance decay adjustments
    - Comprehensive metrics tracking
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.parameters = KellyOptimizationParameters(**self.config.get('kelly_params', {}))
        self._setup_metrics()
        self.trade_history = deque(maxlen=1000)  # Circular buffer for performance tracking
        
    def _setup_metrics(self) -> None:
        """Initialize advanced performance tracking"""
        self.metrics = {
            'calculations_performed': 0,
            'avg_fraction_used': 0.0,
            'win_rate_trend': 0.0,
            'reward_risk_trend': 0.0,
            'growth_rate_achieved': 0.0,
            'drawdown_experienced': 0.0,
            'regime_performance': {},
            'constraint_usage': {c: 0 for c in PortfolioConstraint}
        }

    def calculate_kelly(
        self,
        win_rate: float,
        reward_risk_ratio: float,
        mode: KellyOptimizationMode = KellyOptimizationMode.DYNAMIC,
        custom_fraction: Optional[float] = None,
        portfolio_context: Optional[Dict] = None,
        market_regime: Optional[str] = None,
        volatility: Optional[float] = None,
        current_drawdown: Optional[float] = None,
        confidence_level: float = 0.9
    ) -> KellyResult:
        """
        Enhanced Kelly Criterion calculator with professional features
        
        Args:
            win_rate: Probability of winning (0 < win_rate < 1)
            reward_risk_ratio: Ratio of average win to average loss (>0)
            mode: Kelly optimization approach
            custom_fraction: Custom fraction for FRACTIONAL mode
            portfolio_context: Current portfolio state (positions, correlations, etc.)
            market_regime: Current market regime ('high_vol', 'low_vol', 'trending', etc.)
            volatility: Current volatility measure (annualized)
            current_drawdown: Current portfolio drawdown (0-1)
            confidence_level: Statistical confidence level for estimates (0-1)
            
        Returns:
            KellyResult with comprehensive calculation metrics
        """
        # Input validation and adjustment
        win_rate = self._adjust_win_rate(win_rate)
        reward_risk_ratio = self._adjust_reward_risk(reward_risk_ratio)
        
        # Calculate pure Kelly fraction
        optimal_fraction = max(0.0, win_rate - (1 - win_rate) / reward_risk_ratio)
        
        # Apply optimization mode
        base_fraction = self._apply_optimization_mode(
            optimal_fraction, mode, custom_fraction
        )
        
        # Apply market regime adjustments
        regime_adjusted = self._apply_regime_adjustments(
            base_fraction, market_regime, volatility
        )
        
        # Apply risk management constraints
        constrained_fraction, constraints = self._apply_risk_constraints(
            regime_adjusted, portfolio_context, current_drawdown, volatility
        )
        
        # Calculate advanced metrics
        metrics = self._calculate_advanced_metrics(
            constrained_fraction,
            win_rate,
            reward_risk_ratio,
            confidence_level
        )
        
        # Update performance tracking
        self._update_metrics(constrained_fraction, win_rate, reward_risk_ratio)
        
        return KellyResult(
            fraction=constrained_fraction,
            optimal_fraction=optimal_fraction,
            mode=mode,
            win_rate=win_rate,
            reward_risk_ratio=reward_risk_ratio,
            expected_growth_rate=metrics['expected_growth'],
            risk_adjusted_fraction=metrics['risk_adjusted'],
            max_drawdown_estimate=metrics['max_drawdown'],
            volatility_adjusted_fraction=metrics['vol_adjusted'],
            regime_adjusted_fraction=regime_adjusted,
            confidence_interval=metrics['confidence_interval'],
            portfolio_allocation=metrics['portfolio_allocation'],
            component_scores=metrics['component_scores'],
            constraints_applied=constraints
        )

    def _adjust_win_rate(self, win_rate: float) -> float:
        """Apply credibility adjustments to win rate"""
        decayed = (win_rate * self.parameters.win_rate_decay_factor + 
                  self.parameters.min_win_rate * (1 - self.parameters.win_rate_decay_factor))
        return min(max(decayed, self.parameters.min_win_rate), self.parameters.max_win_rate)

    def _adjust_reward_risk(self, rr: float) -> float:
        """Apply reasonable bounds to reward/risk ratio"""
        return min(max(rr, self.parameters.reward_risk_floor), self.parameters.reward_risk_cap)

    def _apply_optimization_mode(
        self,
        optimal_fraction: float,
        mode: KellyOptimizationMode,
        custom_fraction: Optional[float]
    ) -> float:
        """Apply the selected optimization mode"""
        if mode == KellyOptimizationMode.FULL_KELLY:
            return optimal_fraction
        elif mode == KellyOptimizationMode.HALF_KELLY:
            return optimal_fraction * 0.5
        elif mode == KellyOptimizationMode.QUARTER_KELLY:
            return optimal_fraction * 0.25
        elif mode == KellyOptimizationMode.FRACTIONAL:
            if custom_fraction is None:
                raise ValueError("Custom fraction required for FRACTIONAL mode")
            return optimal_fraction * custom_fraction
        elif mode == KellyOptimizationMode.DYNAMIC:
            # Dynamic sizing based on recent performance
            recent_perf = self._get_recent_performance()
            if recent_perf['drawdown'] > 0.1:
                return optimal_fraction * 0.25
            elif recent_perf['win_rate'] > 0.6:
                return optimal_fraction * 0.75
            else:
                return optimal_fraction * 0.5
        elif mode == KellyOptimizationMode.VOLATILITY_ADJUSTED:
            # Will be adjusted later based on actual volatility
            return optimal_fraction
        else:
            raise ValueError("Invalid optimization mode")

    def _apply_regime_adjustments(
        self,
        fraction: float,
        regime: Optional[str],
        volatility: Optional[float]
    ) -> float:
        """Adjust position sizing based on market conditions"""
        if regime == 'high_vol' or (volatility and volatility > 0.2):
            return fraction * (1 - self.parameters.volatility_adjustment_factor)
        elif regime == 'low_vol' or (volatility and volatility < 0.1):
            return min(fraction * 1.2, 1.0)
        elif regime == 'trending':
            return fraction * 0.8  # Reduce mean-reversion sizing in trends
        return fraction

    def _apply_risk_constraints(
        self,
        fraction: float,
        portfolio_context: Optional[Dict],
        current_drawdown: Optional[float],
        volatility: Optional[float]
    ) -> Tuple[float, List[PortfolioConstraint]]:
        """Apply portfolio-level risk constraints"""
        constraints = []
        constrained_fraction = fraction
        
        # Drawdown constraint
        if current_drawdown and current_drawdown > 0:
            reduction = min(1.0, current_drawdown * self.parameters.drawdown_sensitivity)
            constrained_fraction *= (1 - reduction)
            constraints.append(PortfolioConstraint.DRAWDOWN_TARGET)
        
        # Volatility constraint
        if volatility and volatility > 0.15:
            constrained_fraction *= 0.7
            constraints.append(PortfolioConstraint.VOLATILITY_TARGET)
        
        # Portfolio leverage constraint
        if portfolio_context and 'current_leverage' in portfolio_context:
            max_leverage = portfolio_context.get('max_leverage', 1.0)
            available_leverage = max(0, max_leverage - portfolio_context['current_leverage'])
            constrained_fraction = min(constrained_fraction, available_leverage)
            constraints.append(PortfolioConstraint.LEVERAGE_LIMIT)
        
        # Correlation adjustment
        if portfolio_context and 'correlations' in portfolio_context:
            avg_correlation = np.mean(list(portfolio_context['correlations'].values()))
            if avg_correlation > 0.5:
                constrained_fraction *= (1 - avg_correlation)
                constraints.append(PortfolioConstraint.CORRELATION_ADJUSTMENT)
        
        return max(0, constrained_fraction), constraints

    def _calculate_advanced_metrics(
        self,
        fraction: float,
        win_rate: float,
        reward_risk_ratio: float,
        confidence_level: float
    ) -> Dict:
        """Calculate comprehensive performance metrics"""
        # Expected logarithmic growth
        growth_rate = (win_rate * np.log(1 + fraction * reward_risk_ratio) +
                      (1 - win_rate) * np.log(1 - fraction))
        
        # Risk-adjusted fraction
        z_score = norm.ppf(confidence_level)
        win_rate_std = np.sqrt(win_rate * (1 - win_rate) / 100)  # Assuming 100 sample size
        adj_win_rate = max(0, win_rate - z_score * win_rate_std)
        risk_adjusted = max(0.0, adj_win_rate - (1 - adj_win_rate) / reward_risk_ratio)
        
        # Drawdown estimation
        max_drawdown = self._estimate_drawdown_monte_carlo(
            fraction, win_rate, reward_risk_ratio
        )
        
        # Portfolio allocation score
        portfolio_score = min(1.0, fraction / (risk_adjusted + 1e-6))
        
        return {
            'expected_growth': growth_rate,
            'risk_adjusted': risk_adjusted,
            'max_drawdown': max_drawdown,
            'vol_adjusted': fraction * (1 - self.parameters.volatility_adjustment_factor),
            'confidence_interval': self._calculate_confidence_interval(
                fraction, win_rate, reward_risk_ratio, confidence_level
            ),
            'portfolio_allocation': portfolio_score,
            'component_scores': {
                'growth_potential': growth_rate * 10,
                'risk_efficiency': risk_adjusted / (fraction + 1e-6),
                'drawdown_risk': 1 - max_drawdown,
                'regime_adaptation': portfolio_score
            }
        }

    def _estimate_drawdown_monte_carlo(
        self,
        fraction: float,
        win_rate: float,
        reward_risk_ratio: float,
        n_simulations: int = 1000,
        n_trades: int = 100
    ) -> float:
        """Monte Carlo simulation for drawdown estimation"""
        outcomes = np.random.choice(
            [reward_risk_ratio, -1],
            size=(n_simulations, n_trades),
            p=[win_rate, 1 - win_rate]
        )
        cumulative = np.cumprod(1 + fraction * outcomes, axis=1)
        max_drawdowns = 1 - np.min(cumulative / np.maximum.accumulate(cumulative, axis=1), axis=1)
        return np.percentile(max_drawdowns, 95)  # 95th percentile drawdown

    def _calculate_confidence_interval(
        self,
        fraction: float,
        win_rate: float,
        reward_risk_ratio: float,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for expected returns"""
        returns = fraction * np.array([reward_risk_ratio, -1])
        mean = win_rate * reward_risk_ratio - (1 - win_rate)
        std = np.sqrt(win_rate * (1 - win_rate) * (reward_risk_ratio + 1)**2)
        
        z = norm.ppf(1 - (1 - confidence_level) / 2)
        lower = mean - z * std / np.sqrt(100)  # Assuming 100 trades
        upper = mean + z * std / np.sqrt(100)
        
        return (lower, upper)

    def _get_recent_performance(self) -> Dict:
        """Calculate recent performance metrics from trade history"""
        if not self.trade_history:
            return {
                'win_rate': 0.5,
                'reward_risk': 1.0,
                'drawdown': 0.0
            }
            
        df = pd.DataFrame(self.trade_history)
        wins = df['return'] > 0
        return {
            'win_rate': wins.mean(),
            'reward_risk': df[wins]['return'].mean() / df[~wins]['return'].abs().mean(),
            'drawdown': df['drawdown'].max()
        }

    def _update_metrics(
        self,
        fraction_used: float,
        win_rate: float,
        reward_risk_ratio: float
    ) -> None:
        """Update performance tracking metrics"""
        self.metrics['calculations_performed'] += 1
        self.metrics['avg_fraction_used'] = (
            self.metrics['avg_fraction_used'] * (self.metrics['calculations_performed'] - 1) +
            fraction_used
        ) / self.metrics['calculations_performed']
        
        # Update trend metrics with exponential smoothing
        alpha = 0.1
        self.metrics['win_rate_trend'] = (
            alpha * win_rate + (1 - alpha) * self.metrics['win_rate_trend']
            if self.metrics['win_rate_trend'] else win_rate
        )
        self.metrics['reward_risk_trend'] = (
            alpha * reward_risk_ratio + (1 - alpha) * self.metrics['reward_risk_trend']
            if self.metrics['reward_risk_trend'] else reward_risk_ratio
        )

    def record_trade_outcome(
        self,
        fraction_used: float,
        return_amount: float,
        drawdown_impact: float
    ) -> None:
        """Record actual trade outcomes for performance tracking"""
        self.trade_history.append({
            'fraction': fraction_used,
            'return': return_amount,
            'drawdown': drawdown_impact,
            'timestamp': pd.Timestamp.now()
        })

# Example usage
if __name__ == "__main__":
    optimizer = KellyOptimizer({
        'kelly_params': {
            'win_rate_decay_factor': 0.95,
            'volatility_adjustment_factor': 0.3,
            'drawdown_sensitivity': 1.5
        }
    })
    
    result = optimizer.calculate_kelly(
        win_rate=0.65,
        reward_risk_ratio=1.8,
        mode=KellyOptimizationMode.DYNAMIC,
        market_regime='high_vol',
        volatility=0.25,
        current_drawdown=0.05,
        confidence_level=0.95,
        portfolio_context={
            'current_leverage': 0.5,
            'max_leverage': 2.0,
            'correlations': {'SPX': 0.6, 'NDX': 0.7}
        }
    )
    
    print(f"Recommended position size: {result.fraction:.2%}")
    print(f"Risk-adjusted fraction: {result.risk_adjusted_fraction:.2%}")
    print(f"Expected growth rate: {result.expected_growth_rate:.4f}")
    print(f"95% CI for returns: [{result.confidence_interval[0]:.2%}, {result.confidence_interval[1]:.2%}]")
    print(f"Applied constraints: {result.constraints_applied}")
