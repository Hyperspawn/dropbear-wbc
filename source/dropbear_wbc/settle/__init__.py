"""CPU-side pieces of the motion settle pipeline (``tools/settle_motion.py``).

* :mod:`.motion_io`  -- read/write ``dropbear-motion-csv-v1`` (+ sidecar) and resample to 50 Hz.
* :mod:`.ground`     -- foot sole geometry and the per-frame ground (root z) correction.
* :mod:`.kinematics` -- finite-difference velocities (BeyondMimic ``csv_to_npz`` conventions) and pose composition.

Pure numpy; importable from the system python and from Isaac kit python.
"""
