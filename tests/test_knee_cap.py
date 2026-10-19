"""RetargetOptions.knee_max_deg: the knee-capped SemanticMap (squat references kept off the knee's flexion stop)."""
import numpy as np

import sdk_test_paths  # noqa: F401
from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES, SemanticMap
from dropbear_wbc.motion.g1_to_dropbear import cap_knee_flexion
from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES

CAL = sdk_test_paths.ROOT / "data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json"


def test_knee_cap_limits_semantic_and_motor_side_alike():
    sm = SemanticMap.load(CAL)
    capped = cap_knee_flexion(sm, 44.0)
    for side, motor in (("left", "LL_knee_actuator_joint"), ("right", "RL_knee_actuator_joint")):
        si, mi = SEMANTIC_NAMES.index(f"{side}_knee"), MOTOR_NAMES.index(motor)
        assert np.isclose(np.degrees(capped.semantic_limits[si, 1]), 44.0)
        # a request past the cap maps to the cap, and the capped crank limit maps back to it
        q = np.zeros(len(SEMANTIC_NAMES))
        q[si] = np.radians(60.0)
        m = capped.semantic_to_motor(q)
        m = np.asarray(m[0] if isinstance(m, tuple) else m)
        back = np.degrees(capped.motor_to_semantic(m[None])[0, si])
        assert abs(back - 44.0) < 0.5, back
        assert capped.motor_limits[mi, 1] < sm.motor_limits[mi, 1]
    # the original map is untouched (the foot-contact stage's cached map must not change)
    assert np.degrees(sm.semantic_limits[SEMANTIC_NAMES.index("left_knee"), 1]) > 46.0


def test_cap_above_the_plant_range_changes_nothing():
    sm = SemanticMap.load(CAL)
    capped = cap_knee_flexion(sm, 80.0)
    assert np.array_equal(capped.semantic_limits, sm.semantic_limits)
    assert np.array_equal(capped.motor_limits, sm.motor_limits)
