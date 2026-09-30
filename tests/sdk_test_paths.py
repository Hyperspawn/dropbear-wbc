"""Path setup shared by the SDK tests (imported by each test module; no conftest collision)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "source", ROOT / "third_party" / "pydeps"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
