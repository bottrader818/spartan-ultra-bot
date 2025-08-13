from dataclasses import dataclass
from typing import Optional

@dataclass
class ExecutionReport:
    filled_qty: float
    avg_price: float
    venue: Optional[str] = "SIM"

class PaperBroker:
    def execute(self, target_weight: float, price: float, equity: float) -> ExecutionReport:
        # simple fill assumption
        qty = target_weight * equity / max(price, 1e-9)
        return ExecutionReport(filled_qty=qty, avg_price=price, venue="PAPER")
