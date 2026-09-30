"""xr_teleoperate-style arm teleoperation for Dropbear (simulation).

Modules
-------
* :mod:`.arm_ik`    5-DoF-per-arm IK in the semantic (G1-named) joint space, built from the semantic calibration.
* :mod:`.frames`    frame conventions (OpenXR <-> robot basis, Unitree arm initial-pose convention, torso frame).
* :mod:`.devices`   input sources producing left/right wrist target poses in the robot torso frame
                    (scripted, keyboard, WebXR via Vuer).
* :mod:`.gravity`   gravity feed-forward torques for the arm motors (xr_teleoperate's ``pin.rnea`` equivalent).
* :mod:`.recorder`  LeRobot-v2-like session recorder (parquet + meta JSON + GR00T ``modality.json`` draft).

See ``docs/TELEOP.md``.
"""
