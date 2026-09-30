"""PPO runner config for the Dropbear motion-library tracking tasks: the single-clip BeyondMimic hyper-parameters
(``rsl_rl_ppo_cfg``) with its own experiment directory ``logs/rsl_rl/dropbear_tracking_library`` (so "latest run"
look-ups of the single-clip tracks never pick a library run and vice versa)."""
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import DropbearFlatPPORunnerCfg


@configclass
class DropbearLibraryPPORunnerCfg(DropbearFlatPPORunnerCfg):
    experiment_name = "dropbear_tracking_library"
