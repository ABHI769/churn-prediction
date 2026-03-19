from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"


def get_paths() -> Paths:
    # src/churn/config.py -> src/churn -> src -> project root
    root = Path(__file__).resolve().parents[2]
    return Paths(root=root)


@dataclass(frozen=True)
class ProfitParams:
    retention_gain: float = 120.0
    retention_cost: float = 20.0

