# Isaac-GR00T N1.7 NEW_EMBODIMENT modality config for the Dropbear tabletop datasets
# (data/groot/<dataset>/, built by tools/build_groot_dataset.py; contract: docs/GROOT.md, docs/CONTRACTS.md section 7).
#
# Modelled on Isaac-GR00T examples/SO100/so100_config.py and the pre-registered
# "unitree_g1_full_body_with_waist_height_nav_cmd" config (gr00t/configs/data/embodiment_configs.py): ego_view video,
# left_arm / right_arm joint-space state and RELATIVE NON_EEF arm actions. Dropbear has no hand, so there are no
# hand / gripper keys.
#
# Usage (Isaac-GR00T repo root, e.g. on Brev):
#   uv run python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B \
#       --dataset-path <dataset> --embodiment-tag NEW_EMBODIMENT \
#       --modality-config-path <dataset>/dropbear_tabletop_config.py ...
#
# State / action units: radians, semantic (G1-named) arm joints: shoulder pitch / roll / yaw, elbow (G1 convention),
# wrist roll; left arm [0:5], right arm [5:10]. Recording rate 20 Hz -> a 16-step chunk is 0.8 s.
# Wrist views are optional: remove them here if the dataset was built without them (meta/modality.json lists the
# views that exist).

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

VIDEO_KEYS = ["ego_view", "left_wrist_view", "right_wrist_view"]

dropbear_tabletop_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["left_arm", "right_arm"]),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["left_arm", "right_arm"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(dropbear_tabletop_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
