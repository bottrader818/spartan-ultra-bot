from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

class RiskViolationType(Enum):
    """Types of risk violations"""
    POSITION_SIZE = auto()
    LEVERAGE = auto()
    CONCENTRATION = auto()
    LIQUIDITY = auto()
    VOLATILITY = auto()
    BLACKLIST = auto()
    MAX_DRAWDOWN = auto()
    HOUR_LIMIT = auto()

class RiskLevel(Enum):
    """Risk severity levels"""
    WARNING = auto()
    MINOR = auto()
    MAJOR = auto()
    CRITICAL = auto()

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result"""
    is_valid: bool
    violations: List[Tuple[RiskViolationType, RiskLevel, str]]
    adjusted_trade: Optional[Dict] = None
    max_position_size: Optional[float] = None
    required_adjustments: Optional[List[str]] = None
    risk_score: float = 0.0
    timestamp: datetime = datetime.utcnow()

class BaseRiskEngine:
    """
    Professional-Grade Base Risk Engine with:
    - Comprehensive risk checks
    - Violation severity assessment
    - Trade adjustment suggestions
    - Risk scoring
    - Portfolio-aware risk management
    - Liquidity considerations
    """

    def __init__(self, config: Dict):
        """
        Initialize risk engine with configuration
        
        Args:
            config: Dictionary containing:
                - max_position_size: Maximum position size as % of capital
                - max_leverage: Maximum allowed leverage
                - sector_limits: Dict of {sector: max_allocation}
                - blacklist: List of prohibited instruments
                - max_daily_loss: Maximum daily loss % before shutdown
                - liquidity_threshold: Minimum liquidity score
                - volatility_limits: Dict of volatility thresholds
                - trading_hours: Allowed trading hours
        """
        self.config = self._validate_config(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # State tracking
        self._current_positions = defaultdict(float)
        self._daily_pnl = 0.0
        self._violation_history = []
        self._risk_metrics = {
            'total_checks': 0,
            'total_violations': 0,
            'violations_by_type': defaultdict(int),
            'violations_by_level': defaultdict(int),
            'max_risk_score': 0.0,
            'last_reset': datetime.utcnow()
        }
        
        # Initialize risk models
        self._liquidity_model = self._init_liquidity_model()
        self._volatility_model = self._init_volatility_model()
        
        self.logger.info("Risk engine initialized with config: %s", self.config)

    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set default risk parameters"""
        defaults = {
            'max_position_size': 0.1,  # 10% of capital
            'max_leverage': 3.0,
            'sector_limits': {},
            'blacklist': [],
            'max_daily_loss': 0.05,  # 5%
            'liquidity_threshold': 0.7,
            'volatility_limits': {
                'high': 0.2,
                'medium': 0.15,
                'low': 0.1
            },
            'trading_hours': {
                'start': '09:30',
                'end': '16:00'
            },
            'max_drawdown': 0.2  # 20%
        }
        
        validated = {**defaults, **config}
        
        # Additional validation
        validated['max_position_size'] = min(max(0.01, validated['max_position_size']), 0.5)
        validated['max_leverage'] = min(max(1.0, validated['max_leverage']), 10.0)
        
        return validated

    def validate_trade(self, trade: Dict) -> RiskAssessment:
        """
        Comprehensive trade validation with risk assessment
        
        Args:
            trade: Dictionary containing:
                - symbol: Instrument symbol
                - quantity: Trade quantity
                - price: Execution price
                - direction: LONG/SHORT
                - order_type: Market/Limit/etc.
                - sector: Instrument sector
                - portfolio: Portfolio reference
                - account: Account details
                
        Returns:
            RiskAssessment object with detailed validation results
        """
        violations = []
        adjustments = []
        risk_score = 0.0
        
        # 1. Basic parameter validation
        if not self._validate_trade_params(trade):
            violations.append((RiskViolationType.POSITION_SIZE, RiskLevel.CRITICAL, "Invalid trade parameters"))
            risk_score += 1.0
            return RiskAssessment(False, violations, risk_score=risk_score)
        
        # 2. Position size check
        position_check = self._check_position_size(trade)
        if not position_check.is_valid:
            violations.extend(position_check.violations)
            risk_score += 0.3 * len(position_check.violations)
            adjustments.append('position_size')
        
        # 3. Leverage check
        leverage_check = self._check_leverage(trade)
        if not leverage_check.is_valid:
            violations.extend(leverage_check.violations)
            risk_score += 0.4 * len(leverage_check.violations)
            adjustments.append('leverage')
        
        # 4. Concentration risk
        concentration_check = self._check_concentration(trade)
        if not concentration_check.is_valid:
            violations.extend(concentration_check.violations)
            risk_score += 0.2 * len(concentration_check.violations)
            adjustments.append('sector_allocation')
        
        # 5. Liquidity check
        liquidity_check = self._check_liquidity(trade)
        if not liquidity_check.is_valid:
            violations.extend(liquidity_check.violations)
            risk_score += 0.5 * len(liquidity_check.violations)
            adjustments.append('liquidity')
        
        # 6. Volatility check
        volatility_check = self._check_volatility(trade)
        if not volatility_check.is_valid:
            violations.extend(volatility_check.violations)
            risk_score += 0.4 * len(volatility_check.violations)
            adjustments.append('volatility')
        
        # 7. Blacklist check
        blacklist_check = self._check_blacklist(trade)
        if not blacklist_check.is_valid:
            violations.extend(blacklist_check.violations)
            risk_score += 1.0 * len(blacklist_check.violations)
        
        # 8. Drawdown protection
        drawdown_check = self._check_drawdown(trade)
        if not drawdown_check.is_valid:
            violations.extend(drawdown_check.violations)
            risk_score += 0.8 * len(drawdown_check.violations)
        
        # 9. Trading hours check
        hours_check = self._check_trading_hours()
        if not hours_check.is_valid:
            violations.extend(hours_check.violations)
            risk_score += 0.2 * len(hours_check.violations)
        
        # Update risk metrics
        self._update_risk_metrics(violations, risk_score)
        
        # Prepare adjusted trade if possible
        adjusted_trade = self._suggest_adjustments(trade, adjustments) if adjustments else None
        
        return RiskAssessment(
            is_valid=len(violations) == 0,
            violations=violations,
            adjusted_trade=adjusted_trade,
            max_position_size=self._calculate_max_position(trade),
            required_adjustments=adjustments if adjustments else None,
            risk_score=min(1.0, risk_score)
        )

    def _validate_trade_params(self, trade: Dict) -> bool:
        """Validate basic trade parameters"""
        required = ['symbol', 'quantity', 'price', 'direction']
        return all(k in trade for k in required) and trade['quantity'] > 0 and trade['price'] > 0

    def _check_position_size(self, trade: Dict) -> RiskAssessment:
        """Check position size against limits"""
        position_value = trade['quantity'] * trade['price']
        max_position = self.config['max_position_size'] * trade.get('portfolio_value', 1e6)
        
        violations = []
        if position_value > max_position:
            msg = f"Position size {position_value:.2f} exceeds max {max_position:.2f}"
            violations.append((RiskViolationType.POSITION_SIZE, RiskLevel.MAJOR, msg))
        
        return RiskAssessment(len(violations) == 0, violations)

    def _check_leverage(self, trade: Dict) -> RiskAssessment:
        """Check leverage against limits"""
        current_leverage = trade.get('current_leverage', 1.0)
        new_leverage = trade.get('new_leverage', current_leverage)
        max_leverage = self.config['max_leverage']
        
        violations = []
        if new_leverage > max_leverage:
            msg = f"Leverage {new_leverage:.1f}x exceeds max {max_leverage:.1f}x"
            violations.append((RiskViolationType.LEVERAGE, RiskLevel.CRITICAL, msg))
        
        return RiskAssessment(len(violations) == 0, violations)

    def _check_concentration(self, trade: Dict) -> RiskAssessment:
        """Check sector concentration limits"""
        sector = trade.get('sector')
        if not sector or sector not in self.config['sector_limits']:
            return RiskAssessment(True, [])
        
        current_allocation = trade.get('sector_allocation', 0.0)
        max_allocation = self.config['sector_limits'][sector]
        new_allocation = current_allocation + (trade['quantity'] * trade['price'] / trade.get('portfolio_value', 1e6))
        
        violations = []
        if new_allocation > max_allocation:
            msg = f"Sector {sector} allocation {new_allocation:.2%} exceeds max {max_allocation:.2%}"
            violations.append((RiskViolationType.CONCENTRATION, RiskLevel.MAJOR, msg))
        
        return RiskAssessment(len(violations) == 0, violations)

    def _check_liquidity(self, trade: Dict) -> RiskAssessment:
        """Check instrument liquidity"""
        liquidity_score = trade.get('liquidity_score', self._liquidity_model.get(trade['symbol'], 1.0))
        min_liquidity = self.config['liquidity_threshold']
        
        violations = []
        if liquidity_score < min_liquidity:
            msg = f"Liquidity score {liquidity_score:.2f} below threshold {min_liquidity:.2f}"
            violations.append((RiskViolationType.LIQUIDITY, RiskLevel.MINOR, msg))
        
        return RiskAssessment(len(violations) == 0, violations)

    def _check_volatility(self, trade: Dict) -> RiskAssessment:
        """Check instrument volatility"""
        volatility = trade.get('volatility', self._volatility_model.get(trade['symbol'], 0.1))
        limits = self.config['volatility_limits']
        
        violations = []
        if volatility > limits['high']:
            level = RiskLevel.CRITICAL
        elif volatility > limits['medium']:
            level = RiskLevel.MAJOR
        elif volatility > limits['low']:
            level = RiskLevel.MINOR
        else:
            level = None
            
        if level:
            msg = f"Volatility {volatility:.2%} exceeds {level.name.lower()} threshold"
            violations.append((RiskViolationType.VOLATILITY, level, msg))
        
        return RiskAssessment(len(violations) == 0, violations)

    def _check_blacklist(self, trade: Dict) -> RiskAssessment:
        """Check if instrument is blacklisted"""
        violations = []
        if trade['symbol'] in self.config['blacklist']:
            msg = f"Instrument {trade['symbol']} is blacklisted"
            violations.append((RiskViolationType.BLACKLIST, RiskLevel.CRITICAL, msg))
        
        return RiskAssessment(len(violations) == 0, violations)

    def _check_drawdown(self, trade: Dict) -> RiskAssessment:
        """Check portfolio drawdown limits"""
        daily_pnl = trade.get('daily_pnl_pct', 0.0)
        max_daily_loss = self.config['max_daily_loss']
        max_drawdown = self.config['max_drawdown']
        current_drawdown = trade.get('current_drawdown', 0.0)
        
        violations = []
        if daily_pnl < -max_daily_loss:
            msg = f"Daily PnL {daily_pnl:.2%} exceeds loss limit {-max_daily_loss:.2%}"
            violations.append((RiskViolationType.MAX_DRAWDOWN, RiskLevel.CRITICAL, msg))
        
        if current_drawdown > max_drawdown:
            msg = f"Drawdown {current_drawdown:.2%} exceeds max {max_drawdown:.2%}"
            violations.append((RiskViolationType.MAX_DRAWDOWN, RiskLevel.CRITICAL, msg))
        
        return RiskAssessment(len(violations) == 0, violations)

    def _check_trading_hours(self) -> RiskAssessment:
        """Check if current time is within allowed trading hours"""
        # Simplified implementation - would use proper timezone handling in production
        violations = []
        current_time = datetime.now().time()
        start = datetime.strptime(self.config['trading_hours']['start'], '%H:%M').time()
        end = datetime.strptime(self.config['trading_hours']['end'], '%H:%M').time()
        
        if not (start <= current_time <= end):
            msg = f"Trading outside allowed hours {start}-{end}"
            violations.append((RiskViolationType.HOUR_LIMIT, RiskLevel.MINOR, msg))
        
        return RiskAssessment(len(violations) == 0, violations)

    def _suggest_adjustments(self, trade: Dict, adjustments: List[str]) -> Dict:
        """Suggest trade adjustments to resolve violations"""
        adjusted = trade.copy()
        
        if 'position_size' in adjustments:
            max_size = self._calculate_max_position(trade)
            adjusted['quantity'] = min(trade['quantity'], max_size / trade['price'])
        
        if 'leverage' in adjustments:
            adjusted['leverage'] = min(trade.get('leverage', 1.0), self.config['max_leverage'])
        
        return adjusted

    def _calculate_max_position(self, trade: Dict) -> float:
        """Calculate maximum allowed position size considering all constraints"""
        portfolio_value = trade.get('portfolio_value', 1e6)
        
        # Base position limit
        max_pos = self.config['max_position_size'] * portfolio_value
        
        # Apply sector limit if applicable
        sector = trade.get('sector')
        if sector in self.config['sector_limits']:
            current_sector = trade.get('current_sector_value', 0.0)
            sector_limit = self.config['sector_limits'][sector] * portfolio_value
            max_pos = min(max_pos, sector_limit - current_sector)
        
        return max(0, max_pos)

    def _update_risk_metrics(self, violations: List, risk_score: float):
        """Update risk metrics and violation history"""
        self._risk_metrics['total_checks'] += 1
        self._risk_metrics['total_violations'] += len(violations)
        self._risk_metrics['max_risk_score'] = max(self._risk_metrics['max_risk_score'], risk_score)
        
        for violation in violations:
            self._risk_metrics['violations_by_type'][violation[0]] += 1
            self._risk_metrics['violations_by_level'][violation[1]] += 1
        
        self._violation_history.extend(violations)
        if len(self._violation_history) > 1000:
            self._violation_history = self._violation_history[-1000:]

    def _init_liquidity_model(self) -> Dict[str, float]:
        """Initialize liquidity scoring model"""
        # In production, this would load from a real data source
        return defaultdict(lambda: 1.0)

    def _init_volatility_model(self) -> Dict[str, float]:
        """Initialize volatility scoring model"""
        # In production, this would load from a real data source
        return defaultdict(lambda: 0.1)

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return self._risk_metrics.copy()

    def reset_metrics(self) -> None:
        """Reset risk metrics"""
        self._risk_metrics = {
            'total_checks': 0,
            'total_violations': 0,
            'violations_by_type': defaultdict(int),
            'violations_by_level': defaultdict(int),
            'max_risk_score': 0.0,
            'last_reset': datetime.utcnow()
        }
