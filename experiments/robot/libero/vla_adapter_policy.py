"""
vla_adapter_policy.py

Wraps VLA-Adapter inference as an openpi-compatible BasePolicy.
"""

import numpy as np
from openpi_client.base_policy import BasePolicy

from experiments.robot.robot_utils import (
    get_action,
    normalize_gripper_action,
    invert_gripper_action,
)


class VLAAdapterPolicy(BasePolicy):
    """Wraps VLA-Adapter model components into the openpi BasePolicy interface."""

    def __init__(self, cfg, model, processor, action_head, proprio_projector, mvae_decoder=None):
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.mvae_decoder = mvae_decoder

    def infer(self, obs: dict) -> dict:
        # Map openpi observation keys to VLA-Adapter format
        vla_obs = {
            "full_image": obs["observation/image"],
            "wrist_image": obs["observation/wrist_image"],
            "state": obs["observation/state"],
        }
        prompt = obs["prompt"]

        # Run VLA-Adapter inference → list of np.ndarray actions
        actions = get_action(
            self.cfg,
            self.model,
            vla_obs,
            prompt,
            processor=self.processor,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
            use_minivlm=self.cfg.use_minivlm,
            mvae_decoder=self.mvae_decoder,
        )

        # Post-process each action: normalize gripper [0,1]→[-1,+1] and invert sign
        processed = []
        for a in actions:
            a = normalize_gripper_action(a, binarize=True)
            a = invert_gripper_action(a)
            processed.append(a)

        return {"actions": np.stack(processed)}
