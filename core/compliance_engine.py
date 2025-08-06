from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

class ComplianceLevel(Enum):
    WARNING = auto()
    MINOR_VIOLATION = auto()
    MAJOR_VIOLATION = auto()
    CRITICAL_VIOLATION = auto()

class RuleType(Enum):
    QUANTITATIVE = auto()
    QUALITATIVE = auto()
    TEMPORAL = auto()
    RISK_BASED = auto()
    REGULATORY = auto()

@dataclass
class ComplianceResult:
    is_compliant: bool
    violated_rules: List[Tuple[str, ComplianceLevel]]
    passed_rules: List[str]
    override_status: bool = False
    override_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Trade:
    symbol: str
    quantity: float
    price: float
    direction: str  # 'BUY' or 'SELL'
    account_id: str
    trader_id: str
    timestamp: datetime
    instrument_type: str
    exchange: str
    order_type: str
    metadata: Dict[str, Any] = None

class ComplianceEngine:
    def __init__(self, rules: Optional[Dict[str, Dict]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._rules = self._initialize_rules(rules or {})
        self._violation_history = defaultdict(list)
        self._rule_performance = {}
        self._rule_dependencies = {}

        self.metrics = {
            'total_checks': 0,
            'total_violations': 0,
            'violations_by_level': {level: 0 for level in ComplianceLevel},
            'violations_by_rule': defaultdict(int),
            'average_check_time': 0.0,
            'last_check': None
        }

    def _initialize_rules(self, rule_configs: Dict) -> Dict:
        validated_rules = {}
        for name, config in rule_configs.items():
            if not all(k in config for k in ['function', 'level', 'type']):
                self.logger.warning(f"Rule {name} missing required configuration, skipping")
                continue

            validated_rules[name] = {
                'function': config['function'],
                'level': ComplianceLevel(config['level']),
                'type': RuleType(config['type']),
                'active': config.get('active', True),
                'description': config.get('description', ''),
                'weight': config.get('weight', 1.0)
            }
        return validated_rules

    def add_rule(self, name: str, rule_func: Callable,
                 level: ComplianceLevel, rule_type: RuleType,
                 description: str = '', active: bool = True,
                 weight: float = 1.0) -> None:
        self._rules[name] = {
            'function': rule_func,
            'level': level,
            'type': rule_type,
            'active': active,
            'description': description,
            'weight': weight
        }

    def enable_rule(self, name: str) -> None:
        if name in self._rules:
            self._rules[name]['active'] = True

    def disable_rule(self, name: str) -> None:
        if name in self._rules:
            self._rules[name]['active'] = False

    def add_dependency(self, rule_name: str, depends_on: List[str]) -> None:
        self._rule_dependencies[rule_name] = depends_on

    def check_trade(self, trade: Trade, allow_override: bool = False,
                    override_reason: Optional[str] = None) -> ComplianceResult:
        start_time = datetime.now()
        self.metrics['total_checks'] += 1

        violated_rules = []
        passed_rules = []
        override_used = False

        # ✅ FIXED: Filter active rules BEFORE sorting
        active_rules = [
            (name, rule) for name, rule in sorted(
                ((n, r) for n, r in self._rules.items() if r['active']),
                key=lambda x: (x[1]['level'].value, -x[1]['weight'])
            )
        ]

        for rule_name, rule in active_rules:
            if rule_name in self._rule_dependencies:
                deps = self._rule_dependencies[rule_name]
                if any(dep in [v[0] for v in violated_rules] for dep in deps):
                    self.logger.debug(f"Skipping {rule_name} due to failed dependencies")
                    continue

            try:
                rule_start = datetime.now()
                is_compliant = rule['function'](trade)
                rule_time = (datetime.now() - rule_start).total_seconds()

                self._update_rule_performance(rule_name, rule_time, is_compliant)

                if is_compliant:
                    passed_rules.append(rule_name)
                else:
                    violated_rules.append((rule_name, rule['level']))
                    self._log_violation(trade, rule_name, rule['level'])

            except Exception as e:
                self.logger.error(f"Rule {rule_name} failed during execution: {e}", exc_info=True)
                violated_rules.append((rule_name, ComplianceLevel.CRITICAL_VIOLATION))

        if violated_rules:
            max_violation_level = max(v[1] for v in violated_rules)

            if allow_override and max_violation_level.value < ComplianceLevel.CRITICAL_VIOLATION.value:
                override_used = True
                self.logger.warning(f"Compliance override used for trade {trade}: {override_reason}")
            else:
                self.metrics['total_violations'] += 1
                for rule_name, level in violated_rules:
                    self.metrics['violations_by_rule'][rule_name] += 1
                    self.metrics['violations_by_level'][level] += 1

        check_time = (datetime.now() - start_time).total_seconds()
        self.metrics['average_check_time'] = (
            self.metrics['average_check_time'] * (self.metrics['total_checks'] - 1) + check_time
        ) / self.metrics['total_checks']
        self.metrics['last_check'] = start_time

        return ComplianceResult(
            is_compliant=not bool(violated_rules) or override_used,
            violated_rules=violated_rules,
            passed_rules=passed_rules,
            override_status=override_used,
            override_reason=override_reason
        )

    def _update_rule_performance(self, rule_name: str, execution_time: float, passed: bool) -> None:
        if rule_name not in self._rule_performance:
            self._rule_performance[rule_name] = {
                'total_executions': 0,
                'total_failures': 0,
                'total_time': 0.0,
                'average_time': 0.0
            }

        stats = self._rule_performance[rule_name]
        stats['total_executions'] += 1
        stats['total_time'] += execution_time
        stats['average_time'] = stats['total_time'] / stats['total_executions']

        if not passed:
            stats['total_failures'] += 1

    def _log_violation(self, trade: Trade, rule_name: str, level: ComplianceLevel) -> None:
        violation_entry = {
            'timestamp': datetime.now(),
            'trade': trade,
            'rule': rule_name,
            'level': level,
            'rule_description': self._rules[rule_name]['description']
        }
        self._violation_history[rule_name].append(violation_entry)
        self.logger.warning(
            f"Compliance violation - Rule: {rule_name} ({level.name}), "
            f"Trade: {trade.symbol} {trade.quantity}@{trade.price}, "
            f"Trader: {trade.trader_id}, Account: {trade.account_id}"
        )

    def get_violation_history(self,
                              rule_name: Optional[str] = None,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> List[Dict]:
        if rule_name:
            violations = self._violation_history.get(rule_name, [])
        else:
            violations = [v for rule_vios in self._violation_history.values() for v in rule_vios]

        if start_date:
            violations = [v for v in violations if v['timestamp'] >= start_date]
        if end_date:
            violations = [v for v in violations if v['timestamp'] <= end_date]

        return sorted(violations, key=lambda x: x['timestamp'], reverse=True)

    def get_rule_performance(self) -> Dict:
        return self._rule_performance

    def get_compliance_report(self) -> pd.DataFrame:
        report_data = []
        for rule_name, stats in self._rule_performance.items():
            rule = self._rules.get(rule_name, {})
            report_data.append({
                'Rule': rule_name,
                'Type': rule.get('type', '').name if rule else '',
                'Level': rule.get('level', '').name if rule else '',
                'Active': rule.get('active', False),
                'Executions': stats['total_executions'],
                'Failures': stats['total_failures'],
                'Failure Rate': stats['total_failures'] / max(1, stats['total_executions']),
                'Avg Time (ms)': stats['average_time'] * 1000,
                'Description': rule.get('description', '')
            })
        return pd.DataFrame(report_data)

    def get_metrics(self) -> Dict:
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        self.metrics = {
            'total_checks': 0,
            'total_violations': 0,
            'violations_by_level': {level: 0 for level in ComplianceLevel},
            'violations_by_rule': defaultdict(int),
            'average_check_time': 0.0,
            'last_check': None
        }
