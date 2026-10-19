"""Head and foot height rewards for the tracker (docs/RESULTS.md 2026-10-16): wiring checks, Isaac-free (AST)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source" / "dropbear_wbc"


def _funcs(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_reward_terms_use_heights_only():
    f = _funcs(SRC / "tasks/tracking/mdp/rewards.py")
    for name in ("motion_anchor_height_error_exp", "motion_feet_height_error_exp"):
        assert name in f
        src = ast.unparse(f[name])
        assert "[:, 2]" in src or ", 2]" in src, f"{name} must compare z only (x/y drift is not a height error)"
        assert "torch.exp" in src


def test_cfg_and_scripts_expose_the_weights():
    cfg = (SRC / "tasks/tracking/config/dropbear/flat_env_cfg.py").read_text(encoding="utf-8")
    assert "def enable_height_tracking" in cfg and "FOOT_EE_BODIES" in cfg
    train = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert "--anchor_height_weight" in train and "--feet_height_weight" in train and "enable_height_tracking" in train
    drv = (ROOT / "tools/run_chunked_training.py").read_text(encoding="utf-8")
    assert "--anchor_height_weight=" in drv and "--feet_height_weight=" in drv


def test_strictness_switches_are_wired():
    f = _funcs(SRC / "tasks/tracking/mdp/rewards.py")
    assert "motion_joint_pos_error_exp" in f and "robot_joint_pos" in ast.unparse(f["motion_joint_pos_error_exp"])
    cfg = (SRC / "tasks/tracking/config/dropbear/flat_env_cfg.py").read_text(encoding="utf-8")
    assert "def set_tracking_strictness" in cfg and "terminations.ee_body_pos" in cfg
    train = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    for flag in ("--body_pos_std", "--joint_pos_weight", "--end_threshold_m"):
        assert flag in train and flag + "=" in (ROOT / "tools/run_chunked_training.py").read_text(encoding="utf-8")
