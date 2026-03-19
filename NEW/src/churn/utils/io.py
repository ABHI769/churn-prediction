from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def dump_joblib(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def dump_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

