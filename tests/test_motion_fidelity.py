"""tools/motion_fidelity.py: the visible-feature comparison between a reference motion and a rollout. CPU, Isaac-free."""
import importlib.util
from pathlib import Path

import pytest

import sdk_test_paths  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("motion_fidelity", ROOT / "tools" / "motion_fidelity.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

SQUAT = ROOT / "data/motions_v6ts/kimodo_gen/squat_ts1.19_v6.npz"
JUMP = ROOT / "data/motions_v6ts/asap_g1/motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass_ts1.19_v6.npz"


def test_compare_rules():
    ref = {"flight_frac": 0.2, "lift_cm": 20.0, "torso_drop_cm": 3.0}
    sim = {"flight_frac": 0.0, "lift_cm": 15.0, "torso_drop_cm": 3.0}
    c = mf.compare(ref, sim)
    assert c["present"] == ["flight_frac", "lift_cm"]  # torso drop under the 8 cm threshold: not present
    assert c["performed"] is False and c["score"] == pytest.approx((0.0 + 0.75) / 2, abs=0.01)
    gesture = mf.compare({"flight_frac": 0.0, "lift_cm": 1.0, "torso_drop_cm": 2.0}, {"flight_frac": 0, "lift_cm": 0, "torso_drop_cm": 0})
    assert gesture["performed"] is True and gesture["score"] == 1.0


@pytest.mark.skipif(not SQUAT.is_file() or not JUMP.is_file(), reason="motion data not present")
def test_reference_features_see_the_squat_and_the_jump():
    squat, jump = mf.features(SQUAT), mf.features(JUMP)
    assert squat["torso_drop_cm"] > 10 and squat["flight_frac"] == 0
    assert jump["flight_frac"] > 0.1 and jump["lift_cm"] > 15
    assert mf.compare(squat, squat)["performed"] and mf.compare(jump, jump)["score"] == 1.0
