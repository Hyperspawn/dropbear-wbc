"""Gym registration of the Dropbear tabletop task (import after the Isaac Sim app has started)."""
import gymnasium as gym

PUSH_ID = "Dropbear-Tabletop-Push-v0"
TASK_IDS = (PUSH_ID,)

if PUSH_ID not in gym.registry:
    gym.register(
        id=PUSH_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "dropbear_wbc.tasks.tabletop.tabletop_env_cfg:DropbearTabletopEnvCfg"},
    )
