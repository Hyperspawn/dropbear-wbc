"""GR00T-ready tabletop task for Dropbear (fixed base, push the block into the target zone).

Pure-Python modules (importable without Isaac): :mod:`.kinematics` (hand-tool IK on the teleop arm IK),
:mod:`.layout` (table / block / zone geometry, placement sampling with a held-out split, success rule),
:mod:`.scripted` (IK-based scripted push policy), :mod:`.episode_io` (raw episode writer).
Isaac-only: :mod:`.tabletop_env_cfg`, :mod:`.mdp`, :mod:`.config` (gym registration). See docs/GROOT.md.
"""
