"""H1-style velocity-command locomotion for Dropbear (contract ``dropbear-velocity-v1``, docs/CONTRACTS.md section 7).

Port of Isaac Lab's ``Isaac-Velocity-Flat-H1-v0`` (``isaaclab_tasks/manager_based/locomotion/velocity``, BSD-3) with
the command-range curriculum of unitree_rl_lab's ``Unitree-H1-Velocity`` (Apache-2.0), adapted to Dropbear's
closed-chain plant. Everything except :mod:`.stand` needs a running Isaac Sim app.
"""
