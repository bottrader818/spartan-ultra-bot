from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class AllocationSlice:
    name: str
    venue: Any
    quantity: float

@dataclass
class AllocationPlan:
    symbol: str
    total_qty: float
    allocations: List[AllocationSlice]
    primary_reason: Optional[str] = None
    fallback_reason: Optional[str] = None

    @property
    def primary(self) -> AllocationSlice:
        return self.allocations[0] if self.allocations else AllocationSlice("NONE", None, 0.0)

class ExecutionVenue:
    """Minimal base; tests subclass this and provide attributes and submit()."""
    name: str
    fill_rate: float
    fee: float
    latency_ms: int = 0

    def submit(self, order: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
