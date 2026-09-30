"""PPO runner config for Dropbear velocity locomotion (Isaac Lab H1 hyper-parameters, rsl-rl-lib 2.3.3).

Differences from ``H1FlatPPORunnerCfg``: the rough-terrain network size [512, 256, 128] (Dropbear's closed-chain plant
and 78-dim observation; H1 flat uses [128]*3), empirical observation normalization on (joint velocities and
commands have very different scales; same as the Dropbear tracking task), 3000 iterations, save every 100.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DropbearVelocityFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "dropbear_velocity"
    empirical_normalization = True
    logger = "tensorboard"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
