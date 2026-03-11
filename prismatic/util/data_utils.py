"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

import re
import string
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)


        if self.padding_side == "left":
            def left_pad_sequence(sequences, padding_value):
                max_len = max(seq.size(0) for seq in sequences)
                padded = []
                for seq in sequences:
                    pad_len = max_len - seq.size(0)
                    pad = torch.full((pad_len,), padding_value, dtype=seq.dtype)
                    padded_seq = torch.cat([pad, seq], dim=0)
                    padded.append(padded_seq)
                return torch.stack(padded)

            input_ids = left_pad_sequence(input_ids, self.pad_token_id)
            labels = left_pad_sequence(labels, IGNORE_INDEX)
        else:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)


        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        # Stack latent_action_idx for LAM head (TwoHead architecture)
        if "latent_action_idx" in instances[0]:
            latent_action_idx = [instance["latent_action_idx"] for instance in instances]
            latent_action_idx = torch.stack(latent_action_idx)
        else:
            latent_action_idx = None

        # Stack villax_latent for villa-x continuous latent head (TwoHead architecture)
        if "villax_latent" in instances[0]:
            villax_latent = [instance["villax_latent"] for instance in instances]
            villax_latent = torch.stack(villax_latent)
        else:
            villax_latent = None

        # Stack u_target for MVAE u_t head
        if "u_target" in instances[0]:
            u_target = [instance["u_target"] for instance in instances]
            u_target = torch.stack(u_target)
        else:
            u_target = None

        # Stack u_target_valid mask (True if hash lookup succeeded)
        if "u_target_valid" in instances[0]:
            u_target_valid = torch.stack([instance["u_target_valid"] for instance in instances])
        else:
            u_target_valid = None

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if latent_action_idx is not None:
            output["latent_action_idx"] = latent_action_idx
        if villax_latent is not None:
            output["villax_latent"] = villax_latent
        if u_target is not None:
            output["u_target"] = u_target
        if u_target_valid is not None:
            output["u_target_valid"] = u_target_valid
        return output


@dataclass
class CollatorForLatentAction:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values = torch.stack(initial_pixel_values)

        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        target_pixel_values = torch.stack(target_pixel_values)
        pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)

        action = [torch.from_numpy(instance["action"]) for instance in instances]
        action = torch.stack(action)

        task_instruction = [re.sub('[{}]'.format(string.punctuation), "", instance["task_instruction"]) for instance in instances]

        output = dict(
            videos=pixel_values,
            task_instruction=task_instruction,
            action=action,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        return output


@dataclass
class CollatorForMultiViewVideo:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values = torch.stack(initial_pixel_values)

        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        target_pixel_values = torch.stack(target_pixel_values)
        pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)

        initial_pixel_values_view2 = [instance["initial_pixel_values_view2"] for instance in instances]
        initial_pixel_values_view2 = torch.stack(initial_pixel_values_view2)

        target_pixel_values_view2 = [instance["target_pixel_values_view2"] for instance in instances]
        target_pixel_values_view2 = torch.stack(target_pixel_values_view2)
        pixel_values_view2 = torch.stack([initial_pixel_values_view2, target_pixel_values_view2], dim=1)

        action = [torch.from_numpy(instance["action"]) for instance in instances]
        action = torch.stack(action)

        task_instruction = [re.sub('[{}]'.format(string.punctuation), "", instance["task_instruction"]) for instance in instances]

        output = dict(
            videos=pixel_values,
            videos_view2=pixel_values_view2,
            task_instruction=task_instruction,
            action=action,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        return output
