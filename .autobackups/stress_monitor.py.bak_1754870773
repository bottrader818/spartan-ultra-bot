from enum import Enum

class StressState(str, Enum):
    NORMAL = "normal"
    STRESS = "stress"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    EXTREME = "extreme"

class MarketStressMonitor:
    def __init__(self, threshold_stress=0.4, threshold_elevated=0.6, threshold_critical=0.8, threshold_extreme=0.9):
        self.th_s = threshold_stress
        self.th_e = threshold_elevated
        self.th_c = threshold_critical
        self.th_x = threshold_extreme

    def score(self, metrics: dict) -> float:
        keys = [k for k in ("volatility", "spread", "drawdown", "correlation") if k in metrics]
        if not keys:
            return 0.0
        return float(sum(metrics[k] for k in keys) / len(keys))

    def state(self, metrics: dict) -> "StressState":
        s = self.score(metrics)
        if s >= self.th_x: return StressState.EXTREME
        if s >= self.th_c: return StressState.CRITICAL
        if s >= self.th_e: return StressState.ELEVATED
        if s >= self.th_s: return StressState.STRESS
        return StressState.NORMAL
