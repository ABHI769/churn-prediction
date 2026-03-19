from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn.config import ProfitParams, get_paths
from churn.data.synthetic import SyntheticChurnConfig, make_synthetic_churn_df
from churn.model.train import save_artifacts, train_xgboost
from churn.utils.io import dump_json, ensure_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--churn-rate", type=float, default=0.27)
    p.add_argument("--no-smote", action="store_true")
    p.add_argument("--retention-gain", type=float, default=120.0)
    p.add_argument("--retention-cost", type=float, default=20.0)
    args = p.parse_args()

    paths = get_paths()
    ensure_dir(paths.data_dir)

    df = make_synthetic_churn_df(
        SyntheticChurnConfig(n_rows=args.n_rows, seed=args.seed, churn_rate_target=args.churn_rate)
    )
    data_path = paths.data_dir / "synthetic_churn.csv"
    df.to_csv(data_path, index=False)

    profit = ProfitParams(retention_gain=args.retention_gain, retention_cost=args.retention_cost)
    bundle = train_xgboost(df, profit=profit, use_smote=(not args.no_smote))
    artifacts = save_artifacts(bundle)

    metrics_path = paths.artifacts_dir / "training_metrics.json"
    dump_json(bundle["metadata"]["metrics"], metrics_path)

    print("Saved dataset:", data_path)
    print("Saved artifacts:", artifacts)
    print("Key metrics:", bundle["metadata"]["metrics"])


if __name__ == "__main__":
    main()

