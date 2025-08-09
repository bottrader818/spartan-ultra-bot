#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Professional Compliance Engine v1.0"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import json
import uuid
from enum import Enum, auto
from dataclasses import dataclass
import yaml  # Requires pyyaml

# ────────────────────────────────────────────────
# ENUMS & EXCEPTIONS
# ────────────────────────────────────────────────

class ComplianceViolation(Exception):
    def __init__(self, message: str, rule: str, severity: str = "HIGH"):
        self.rule = rule
        self.severity = severity
        super().__init__(f"[{severity}] {rule} - {message}")

class RuleSeverity(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

@dataclass
class ComplianceRule:
    name: str
    description: str
    severity: RuleSeverity
    enabled: bool = True
    custom_check: Optional[Callable[[Dict, Dict], Tuple[bool, str]]] = None

# ────────────────────────────────────────────────
# MAIN ENGINE
# ────────────────────────────────────────────────

class ComplianceEngine:
    def __init__(
        self, 
        rules_config: Optional[str] = None, 
        log_dir: str = "logs/compliance",
        alert_threshold: RuleSeverity = RuleSeverity.HIGH
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.alert_threshold = alert_threshold
        self.logger = self._setup_logging()
        self.rules = self._load_default_rules()
        self.custom_rules: Dict[str, ComplianceRule] = {}

        if rules_config:
            self.load_rules_from_config(rules_config)

        self.violation_counts: Dict[str, int] = {}
        self.last_alert_time: Optional[datetime] = None
        self.alert_cooldown = timedelta(minutes=30)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(self.log_dir / "compliance_actions.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        eh = logging.FileHandler(self.log_dir / "compliance_errors.log")
        eh.setLevel(logging.ERROR)
        eh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(eh)

        return logger

    def _load_default_rules(self) -> Dict[str, ComplianceRule]:
        return {
            "allow_short_selling": ComplianceRule(
                name="Short Selling Restriction",
                description="Prohibits short selling activities",
                severity=RuleSeverity.HIGH,
                enabled=True
            ),
            "restricted_symbols": ComplianceRule(
                name="Restricted Securities",
                description="Blocks trading of designated symbols",
                severity=RuleSeverity.CRITICAL,
                enabled=True
            ),
            "max_trade_size": ComplianceRule(
                name="Position Size Limit",
                description="Limits trade size as % of portfolio",
                severity=RuleSeverity.MEDIUM,
                enabled=True
            ),
            "account_verification": ComplianceRule(
                name="Account Verification",
                description="Requires KYC verification for trading",
                severity=RuleSeverity.CRITICAL,
                enabled=True
            ),
            "daily_trade_limit": ComplianceRule(
                name="Day Trading Limit",
                description="Limits number of daily trades",
                severity=RuleSeverity.MEDIUM,
                enabled=True
            )
        }

    def load_rules_from_config(self, config_path: str) -> None:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            for rule_name, rule_config in config.get("rules", {}).items():
                if rule_name in self.rules:
                    if "enabled" in rule_config:
                        self.rules[rule_name].enabled = rule_config["enabled"]
                    if "severity" in rule_config:
                        self.rules[rule_name].severity = RuleSeverity[rule_config["severity"]]
                else:
                    self.add_custom_rule(
                        name=rule_name,
                        description=rule_config.get("description", ""),
                        severity=RuleSeverity[rule_config.get("severity", "MEDIUM")],
                        enabled=rule_config.get("enabled", True)
                    )

            if "restricted_symbols" in config:
                self.set_parameter("restricted_symbols", config["restricted_symbols"])
            if "max_trade_size_pct" in config:
                self.set_parameter("max_trade_size_pct", config["max_trade_size_pct"])

        except Exception as e:
            self.logger.error(f"Failed to load rules config: {e}", exc_info=True)
            raise

    def add_custom_rule(
        self,
        name: str,
        description: str,
        severity: RuleSeverity,
        enabled: bool = True,
        custom_check: Optional[Callable[[Dict, Dict], Tuple[bool, str]]] = None
    ) -> None:
        self.custom_rules[name] = ComplianceRule(
            name=name,
            description=description,
            severity=severity,
            enabled=enabled,
            custom_check=custom_check
        )
        self.logger.info(f"Added custom rule: {name}")

    def set_parameter(self, param_name: str, value: Any) -> None:
        setattr(self, param_name, value)
        self.logger.info(f"Updated parameter {param_name} = {value}")

    def enforce_rules(self, trade: Dict[str, Any], account: Dict[str, Any]) -> None:
        violations = self.check_rules(trade, account)

        if violations:
            for rule_name, _, _ in violations:
                self.violation_counts[rule_name] = self.violation_counts.get(rule_name, 0) + 1

            highest_severity = max(v[2] for v in violations)
            messages = [f"{v[1]} (Rule: {v[0]})" for v in violations]

            if highest_severity.value >= self.alert_threshold.value:
                self.trigger_alert(violations, trade, account)

            raise ComplianceViolation(
                message="; ".join(messages),
                rule="multiple",
                severity=highest_severity.name
            )

        self._log_action("APPROVED", trade, account)

    def check_rules(self, trade: Dict[str, Any], account: Dict[str, Any]) -> List[Tuple[str, str, RuleSeverity]]:
        violations = []

        if self.rules["allow_short_selling"].enabled:
            if not getattr(self, "allow_short_selling", False) and trade.get("direction", "").upper() == "SHORT":
                violations.append(("allow_short_selling", "Short selling is disallowed", self.rules["allow_short_selling"].severity))

        if self.rules["restricted_symbols"].enabled:
            restricted_symbols = getattr(self, "restricted_symbols", [])
            if trade.get("symbol") in restricted_symbols:
                violations.append(("restricted_symbols", f"Trading restricted symbol: {trade['symbol']}", self.rules["restricted_symbols"].severity))

        for rule_name, rule in self.custom_rules.items():
            if rule.enabled and rule.custom_check:
                passed, message = rule.custom_check(trade, account)
                if not passed:
                    violations.append((rule_name, message, rule.severity))

        return violations

    def trigger_alert(self, violations: List[Tuple[str, str, RuleSeverity]], trade: Dict[str, Any], account: Dict[str, Any]) -> None:
        if self.last_alert_time and datetime.now() - self.last_alert_time < self.alert_cooldown:
            return

        alert_details = {
            "time": datetime.utcnow().isoformat(),
            "trade": trade,
            "account": account.get("account_id"),
            "violations": [
                {"rule": v[0], "message": v[1], "severity": v[2].name} for v in violations
            ]
        }

        alert_path = self.log_dir / "alerts" / f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        alert_path.parent.mkdir(exist_ok=True)

        try:
            with alert_path.open("w") as f:
                json.dump(alert_details, f, indent=2)
            self.logger.critical(f"ALERT: {len(violations)} high-severity violations detected")
            self.last_alert_time = datetime.now()
        except Exception as e:
            self.logger.error(f"Failed to write alert: {e}", exc_info=True)

    def _log_action(self, action: str, trade: Dict[str, Any], account: Dict[str, Any], violations: Optional[List[Tuple[str, str, RuleSeverity]]] = None) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_id": str(uuid.uuid4()),
            "action": action,
            "symbol": trade.get("symbol"),
            "side": trade.get("direction"),
            "size": trade.get("size"),
            "account_id": account.get("account_id"),
            "violations": [
                {"rule": v[0], "message": v[1], "severity": v[2].name} for v in violations
            ] if violations else [],
            "metadata": {
                "ip_address": account.get("ip_address"),
                "device_id": account.get("device_id")
            }
        }

        try:
            log_path = self.log_dir / "compliance_audit.log"
            with log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            self.logger.info(f"{action} - {trade.get('symbol')} - {entry['event_id']}")
        except Exception as e:
            self.logger.error(f"Failed to write compliance log: {e}", exc_info=True)

    def get_violation_stats(self, time_window: timedelta = timedelta(days=1)) -> Dict[str, int]:
        return {k: v for k, v in self.violation_counts.items()}
    
    def verify_account(self, account: Dict[str, Any]) -> bool:
        return all([
            account.get("verified", False),
            not account.get("restricted", False),
            account.get("status") == "ACTIVE"
        ])
