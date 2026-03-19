from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """
    Portable training entrypoint:
    - generates synthetic data
    - trains model and writes artifacts/
    """
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from scripts.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()

