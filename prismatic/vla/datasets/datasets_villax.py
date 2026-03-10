"""
datasets_villax.py

Dataset with precomputed villa-x continuous latent loading for TwoHead architecture.
Extends base RLDS dataset to include precomputed villa-x latent targets.

Unlike datasets_lam.py which computes LAM indices on-the-fly, this module loads
precomputed continuous latent vectors from disk for efficiency and to avoid
environment conflicts (villa-x has incompatible dependencies).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
import json
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


def compute_image_hash(img: np.ndarray) -> str:
    """
    Compute a fast hash from an image for lookup purposes.
    Uses a sample of pixels to create a unique identifier.
    """
    # Sample pixels at fixed positions for speed
    h, w = img.shape[:2]
    samples = []
    for i in range(0, h, h // 4):
        for j in range(0, w, w // 4):
            samples.extend(img[i, j].flatten()[:3].tolist())
    return "_".join(str(int(x)) for x in samples[:20])


class VillaXLatentLoader:
    """
    Loads precomputed villa-x latents from disk.

    Latents are organized by dataset and can be indexed by:
    1. (episode_idx, step_start) keys - for sequential preprocessing
    2. Content-based hash keys - for robust matching with shuffled data
    """

    def __init__(self, latents_root: Path):
        """
        Args:
            latents_root: Root directory containing precomputed latents.
                         Should have structure:
                         latents_root/
                             metadata.json
                             <dataset_name>/
                                 latents.npy
                                 key_to_idx.json
                                 hash_to_idx.json (optional, for content-based lookup)
        """
        self.latents_root = Path(latents_root)

        # Load metadata
        with open(self.latents_root / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.latent_dim = self.metadata["latent_dim"]
        self.window_size = self.metadata["window_size"]

        # Load latents and mappings for each dataset
        self.latents = {}  # dataset_name -> numpy array
        self.key_to_idx = {}  # dataset_name -> {key: idx}
        self.hash_to_idx = {}  # dataset_name -> {hash: idx} (optional)

        for dataset_name in self.metadata["datasets"]:
            dataset_dir = self.latents_root / dataset_name
            self.latents[dataset_name] = np.load(dataset_dir / "latents.npy")
            with open(dataset_dir / "key_to_idx.json", "r") as f:
                self.key_to_idx[dataset_name] = json.load(f)
            # Try to load hash-based index if available
            hash_file = dataset_dir / "hash_to_idx.json"
            if hash_file.exists():
                with open(hash_file, "r") as f:
                    self.hash_to_idx[dataset_name] = json.load(f)

        self._miss_count = 0
        self._hit_count = 0

    def get_latent(self, dataset_name: str, episode_idx: int, step_start: int) -> Optional[np.ndarray]:
        """
        Get the precomputed latent for a sample using episode/step indices.

        Args:
            dataset_name: Name of the dataset
            episode_idx: Episode index within the dataset
            step_start: Starting step index of the window

        Returns:
            Latent vector of shape (latent_dim,) or None if not found
        """
        key = f"{dataset_name}_{episode_idx}_{step_start}"

        if dataset_name not in self.key_to_idx:
            return None

        idx = self.key_to_idx[dataset_name].get(key)
        if idx is None:
            return None

        return self.latents[dataset_name][idx]

    def get_latent_by_hash(
        self,
        dataset_name: str,
        img_first: np.ndarray,
        img_last: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Get the precomputed latent using content-based hash.
        This is more robust when training data order doesn't match preprocessing.

        Args:
            dataset_name: Name of the dataset
            img_first: First frame of the window
            img_last: Last frame of the window

        Returns:
            Latent vector of shape (latent_dim,) or None if not found
        """
        if dataset_name not in self.hash_to_idx:
            self._miss_count += 1
            return None

        # Compute hash from image pair
        hash_key = f"{compute_image_hash(img_first)}_{compute_image_hash(img_last)}"

        idx = self.hash_to_idx[dataset_name].get(hash_key)
        if idx is None:
            self._miss_count += 1
            return None

        self._hit_count += 1
        return self.latents[dataset_name][idx]

    def get_stats(self) -> dict:
        """Get cache hit/miss statistics."""
        total = self._hit_count + self._miss_count
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": self._hit_count / total if total > 0 else 0.0
        }


@dataclass
class RLDSBatchTransformWithVillaX:
    """
    Batch transform that loads precomputed villa-x latents for TwoHead architecture.
    Returns both robot actions (for L1RegressionActionHead) and villa-x latents (for continuous LAM head).

    Unlike RLDSBatchTransformWithLAM, this does NOT compute latents on-the-fly.
    Instead, it loads precomputed latents from disk using episode/step indices.
    """
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False
    villax_loader: Optional[VillaXLatentLoader] = None  # Loader for precomputed villa-x latents (fallback)

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the TwoHead model with villa-x latents."""
        dataset_name = rlds_batch["dataset_name"]
        dataset_name_str = dataset_name.decode() if isinstance(dataset_name, bytes) else dataset_name
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]
        current_action = actions[0]
        future_actions = actions[1:]

        # Read villax_latent directly from the RLDS observation
        # (embedded in the tfrecord by embed_villax_in_rlds.py)
        assert "villax_latent" in rlds_batch["observation"], (
            f"'villax_latent' not found in RLDS observation for dataset '{dataset_name_str}'. "
            f"Available keys: {list(rlds_batch['observation'].keys())}. "
            f"Make sure you are using --data_root_dir pointing to the embedded RLDS "
            f"(e.g. modified_libero_rlds_villax), NOT the original modified_libero_rlds. "
            f"Hash-based fallback is disabled because image augmentation corrupts pixel hashes."
        )
        # The observation is windowed: shape (window_size, latent_dim)
        # Use the first step's latent (corresponds to window starting at this step)
        villax_latent = rlds_batch["observation"]["villax_latent"][0]

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

        # [CRITICAL] Mask all labels except action tokens
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        pixel_values = self.image_transform(img)

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,  # Robot actions for L1RegressionActionHead
            villax_latent=torch.from_numpy(villax_latent).float(),  # villa-x latent for continuous head (512,)
        )

        # Add wrist image if requested
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if k.startswith("image_") and "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            if all_wrist_pixels:
                return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)

        # Add proprio if requested
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"][0]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDatasetWithVillaX(IterableDataset):
    """RLDS Dataset that loads precomputed villa-x latents for TwoHead architecture."""

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransformWithVillaX,
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

        if "aloha" in self.data_mix or "basket" in self.data_mix or "lift_pot" in self.data_mix or "plate_handover" in self.data_mix or "close_marker" in self.data_mix:
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
                window_size=NUM_ACTIONS_CHUNK,
                future_action_window_size=0,
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
