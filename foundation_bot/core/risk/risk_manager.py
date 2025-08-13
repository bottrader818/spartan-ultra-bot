from dataclasses import dataclass

@dataclass
class RiskManager:
    max_drawdown: float = 0.25  # 25% hard stop

    def breach(self, peak_equity: float, equity: float) -> bool:
        if peak_equity <= 0: return False
        dd = 1.0 - (equity / peak_equity)
        return dd >= self.max_drawdown
