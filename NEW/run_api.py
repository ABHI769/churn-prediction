from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """
    Portable entrypoint:
    - no need to set PYTHONPATH
    - works on any Windows machine with Python + deps installed
    """
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import uvicorn

    uvicorn.run("churn.api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()

