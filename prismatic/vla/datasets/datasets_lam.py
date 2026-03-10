"""
datasets_lam.py

Dataset with LAM index loading for TwoHead architecture.
Extends base RLDS dataset to include LAM ground truth indices.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, IGNORE_INDEX, NUM_ACTIONS_CHUNK, NUM_TOKENS
from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


@dataclass
class RLDSBatchTransformWithLAM:
    """
    Batch transform that includes LAM index computation for TwoHead architecture.
    Returns both robot actions (for L1RegressionActionHead) and LAM indices (for LAMCodeHead).
    """
    action_tokenizer: ActionTokenizer  # Action tokenizer for creating action tokens (needed for model masking)
    lam_encoder: torch.nn.Module  # LAM encoder for computing latent action indices
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform  # Transform for LAM encoder input (just ToTensor)
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the TwoHead model."""
        dataset_name = rlds_batch["dataset_name"]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]
        current_action = actions[0]
        future_actions = actions[1:]

        # Get first and last frame for LAM encoding
        img_first = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_last = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])

        # Compute LAM indices from image pair
        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(img_first)
            target_pixel_values = self.image_transform_lam(img_last)

            # Get device from LAM encoder
            if hasattr(self.lam_encoder, "device") and not callable(getattr(self.lam_encoder, "device", None)):
                device = self.lam_encoder.device
            elif callable(getattr(self.lam_encoder, "device", None)):
                device = self.lam_encoder.device()
            else:
                device = next(self.lam_encoder.parameters()).device

            video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(device)

            # Use vq_encode to get LAM indices
            latent_action_idx = self.lam_encoder.vq_encode(video)['indices'].squeeze()

        # Construct prompt with action tokens (needed for model's action mask logic)
        if self.use_minivlm:
            prompt_builder = QwenPromptBuilder("openvla")

            # Tokenize actions for input_ids/labels (model needs action tokens for masking)
            future_actions_string = self.action_tokenizer(future_actions, self.use_minivlm)
            current_action_string = self.action_tokenizer(current_action, self.use_minivlm)

            action_chunk_string = [current_action_string] + future_actions_string
            flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ""},
            ]

            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

            # Remove trailing tokens for minivlm
            if len(input_ids) >= 3:
                input_ids = input_ids[:-3]

            # Add action tokens to input_ids (needed for model's mask logic)
            if NUM_TOKENS < len(flattened_action_chunk_string):
                input_ids = input_ids + flattened_action_chunk_string[:NUM_TOKENS]
            else:
                remaining_length = NUM_TOKENS - len(flattened_action_chunk_string)
                extended_array = random.choices(flattened_action_chunk_string, k=remaining_length)
                input_ids = input_ids + flattened_action_chunk_string + extended_array

            # Labels should have action tokens (for model mask logic) but we don't use CE loss on them
            labels = list(input_ids)
            action_chunk_len = NUM_TOKENS

        else:
            prompt_builder = self.prompt_builder_fn("openvla")
            future_actions_string = ''.join(self.action_tokenizer(future_actions, use_minivlm=False))
            current_action_string = self.action_tokenizer(current_action, use_minivlm=False)
            action_chunk_string = current_action_string + future_actions_string

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string[0]},
            ]

            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(input_ids)
            action_chunk_len = 1

        # Tensorize
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # [CRITICAL] Mask all labels except action tokens - this prevents special tokens from being
        # detected as action tokens by the mask logic (special tokens have high IDs > ACTION_TOKEN_BEGIN_IDX)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        pixel_values = self.image_transform(img)

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,  # Robot actions for L1RegressionActionHead
            latent_action_idx=latent_action_idx,  # LAM indices for LAMCodeHead (tensor of 4 ints, 0-15)
        )

        # Add wrist image if requested
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            if all_wrist_pixels:
                return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)

        # Add proprio if requested (only first timestep, not full trajectory)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"][0]  # Take first timestep only
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDatasetWithLAM(IterableDataset):
    """RLDS Dataset that includes LAM index computation for TwoHead architecture."""
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransformWithLAM,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]

        if "aloha" in self.data_mix or "basket" in self.data_mix or "lift_pot" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=NUM_ACTIONS_CHUNK,  # Need all 8 frames for LAM encoding (first + last)
                future_action_window_size=0,  # No additional future actions needed
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs": dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )})

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
