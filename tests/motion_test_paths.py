"""Path setup + shared helpers for the motion-pipeline tests (imported by each test module; no conftest)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "source", ROOT / "third_party" / "pydeps"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MOCK_CAL = ROOT / "tests" / "fixtures" / "mock_semantic_calibration.json"
REAL_CAL = ROOT / "data" / "calibration" / "dropbear_semantic_calibration.json"
