"""Gym registration of the Dropbear velocity tasks (import after the Isaac Sim app has started)."""
import gymnasium as gym

TRAIN_ID = "Dropbear-Velocity-Flat-v0"
PLAY_ID = "Dropbear-Velocity-Flat-Play-v0"
TASK_IDS = (TRAIN_ID, PLAY_ID)

_AGENT = f"{__name__}.agents.rsl_rl_ppo_cfg:DropbearVelocityFlatPPORunnerCfg"

if TRAIN_ID not in gym.registry:
    gym.register(
        id=TRAIN_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DropbearVelocityFlatEnvCfg",
            "rsl_rl_cfg_entry_point": _AGENT,
        },
    )
if PLAY_ID not in gym.registry:
    gym.register(
        id=PLAY_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DropbearVelocityFlatPlayEnvCfg",
            "rsl_rl_cfg_entry_point": _AGENT,
        },
    )
