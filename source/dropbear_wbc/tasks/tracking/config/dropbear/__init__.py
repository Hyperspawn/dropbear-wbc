"""Gym registration of the Dropbear tracking tasks (import after the Isaac Sim app has started)."""
import gymnasium as gym

TRAIN_ID = "Dropbear-Tracking-Flat-v0"
PLAY_ID = "Dropbear-Tracking-Flat-Play-v0"
TASK_IDS = (TRAIN_ID, PLAY_ID)

_AGENT = f"{__name__}.agents.rsl_rl_ppo_cfg:DropbearFlatPPORunnerCfg"

if TRAIN_ID not in gym.registry:
    gym.register(
        id=TRAIN_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DropbearFlatEnvCfg",
            "rsl_rl_cfg_entry_point": _AGENT,
        },
    )
if PLAY_ID not in gym.registry:
    gym.register(
        id=PLAY_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DropbearFlatPlayEnvCfg",
            "rsl_rl_cfg_entry_point": _AGENT,
        },
    )

# Motion-library (multi-clip) tracking, docs/CONTRACTS.md 5.3 (added by the multiclip track, 2026-09-24). Additive:
# the single-clip ids above are unchanged. Entry points are strings, so nothing is imported until a task is made.
LIBRARY_TRAIN_ID = "Dropbear-Tracking-Library-v0"
LIBRARY_PLAY_ID = "Dropbear-Tracking-Library-Play-v0"
LIBRARY_FUTURE_TRAIN_ID = "Dropbear-Tracking-Library-Future-v0"
LIBRARY_FUTURE_PLAY_ID = "Dropbear-Tracking-Library-Future-Play-v0"
LIBRARY_TASK_IDS = (LIBRARY_TRAIN_ID, LIBRARY_PLAY_ID, LIBRARY_FUTURE_TRAIN_ID, LIBRARY_FUTURE_PLAY_ID)
_LIBRARY_AGENT = f"{__name__}.agents.rsl_rl_ppo_library_cfg:DropbearLibraryPPORunnerCfg"
for _task_id, _cfg_name in zip(LIBRARY_TASK_IDS, ("DropbearLibraryEnvCfg", "DropbearLibraryPlayEnvCfg",
                                                  "DropbearLibraryFutureEnvCfg", "DropbearLibraryFuturePlayEnvCfg")):
    if _task_id not in gym.registry:
        gym.register(
            id=_task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.library_env_cfg:{_cfg_name}",
                "rsl_rl_cfg_entry_point": _LIBRARY_AGENT,
            },
        )
TASK_IDS = TASK_IDS + LIBRARY_TASK_IDS

# No-state-estimation library variant (added 2026-09-25, lead): the policy drops motion_anchor_pos_b and base_lin_vel
# (simulator truth a real robot cannot measure); the critic keeps them. BeyondMimic's Wo-State-Estimation recipe.
LIBRARY_NOSTATE_TRAIN_ID = "Dropbear-Tracking-Library-NoState-v0"
LIBRARY_NOSTATE_PLAY_ID = "Dropbear-Tracking-Library-NoState-Play-v0"
for _task_id, _cfg_name in ((LIBRARY_NOSTATE_TRAIN_ID, "DropbearLibraryNoStateEnvCfg"),
                            (LIBRARY_NOSTATE_PLAY_ID, "DropbearLibraryNoStatePlayEnvCfg")):
    if _task_id not in gym.registry:
        gym.register(
            id=_task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.library_env_cfg:{_cfg_name}",
                "rsl_rl_cfg_entry_point": _LIBRARY_AGENT,
            },
        )
TASK_IDS = TASK_IDS + (LIBRARY_NOSTATE_TRAIN_ID, LIBRARY_NOSTATE_PLAY_ID)

# Single-clip no-state variant (added 2026-09-25 evening): the deployable counterpart of Dropbear-Tracking-Flat-v0.
FLAT_NOSTATE_TRAIN_ID = "Dropbear-Tracking-Flat-NoState-v0"
FLAT_NOSTATE_PLAY_ID = "Dropbear-Tracking-Flat-NoState-Play-v0"
for _task_id, _cfg_name in ((FLAT_NOSTATE_TRAIN_ID, "DropbearFlatNoStateEnvCfg"),
                            (FLAT_NOSTATE_PLAY_ID, "DropbearFlatNoStatePlayEnvCfg")):
    if _task_id not in gym.registry:
        gym.register(
            id=_task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.library_env_cfg:{_cfg_name}",
                "rsl_rl_cfg_entry_point": _AGENT,
            },
        )
TASK_IDS = TASK_IDS + (FLAT_NOSTATE_TRAIN_ID, FLAT_NOSTATE_PLAY_ID)
