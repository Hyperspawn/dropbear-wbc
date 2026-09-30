"""CPU parity test of the most recent tracking-policy export (skips when there is none).

Needs torch + onnxruntime (system python). The export is produced by ``scripts/play.py --export``
under ``logs/rsl_rl/dropbear_tracking/<run>/exported``. Override with ``DROPBEAR_EXPORT_DIR``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "source"), str(REPO / "tools")]


def _latest_export() -> Path | None:
    env = os.environ.get("DROPBEAR_EXPORT_DIR")
    if env:
        return Path(env)
    candidates = sorted(REPO.glob("logs/rsl_rl/dropbear_tracking/*/exported/policy.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1].parent if candidates else None


def test_export_parity():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("torch")
    export_dir = _latest_export()
    if export_dir is None or not (export_dir / "policy.json").is_file():
        pytest.skip("no tracking-policy export found")
    import check_export_parity

    assert check_export_parity.main(export_dir) == 0
