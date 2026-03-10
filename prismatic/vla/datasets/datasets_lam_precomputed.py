"""
datasets_lam_precomputed.py

Dataset with precomputed LAM index loading for TwoHead architecture.
Extends base RLDS dataset to include precomputed LAM discrete indices.

Unlike datasets_lam.py which computes LAM indices on-the-fly using the LAM encoder,
this module loads precomputed discrete indices from disk for efficiency.
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
    """MD5 hash of full image bytes. Must match precompute_lam_indices.py."""
    import hashlib
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


class LAMIndexLoader:
    """
    Loads precomputed LAM discrete indices from disk.

    Indices are organized by dataset and can be looked up by:
    1. (episode_idx, step_start) keys - for sequential preprocessing
    2. Content-based hash keys - for robust matching with shuffled data
    """

    def __init__(self, indices_root: Path):
        """
        Args:
            indices_root: Root directory containing precomputed LAM indices.
                         Should have structure:
                         indices_root/
                             metadata.json
                             <dataset_name>/
                                 indices.npy       (N, 4) int16
                                 key_to_idx.json
                                 hash_to_idx.json
        """
        self.indices_root = Path(indices_root)

        # Load metadata
        with open(self.indices_root / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.num_codes = self.metadata["num_codes"]  # 4
        self.num_categories = self.metadata["num_categories"]  # 16

        # Load indices and mappings for each dataset
        self.indices = {}  # dataset_name -> numpy array (N, 4)
        self.key_to_idx = {}  # dataset_name -> {key: idx}
        self.hash_to_idx = {}  # dataset_name -> {hash: idx}

        for dataset_name in self.metadata["datasets"]:
            dataset_dir = self.indices_root / dataset_name
            self.indices[dataset_name] = np.load(dataset_dir / "indices.npy")
            with open(dataset_dir / "key_to_idx.json", "r") as f:
                self.key_to_idx[dataset_name] = json.load(f)
            hash_file = dataset_dir / "hash_to_idx.json"
            if hash_file.exists():
                with open(hash_file, "r") as f:
                    self.hash_to_idx[dataset_name] = json.load(f)

        # Load u_vectors for each dataset if available (from MVAE precompute_u.py)
        self.u_vectors = {}
        for dataset_name in self.metadata["datasets"]:
            dataset_dir = self.indices_root / dataset_name
            u_path = dataset_dir / "u_vectors.npy"
            if u_path.exists():
                self.u_vectors[dataset_name] = np.load(u_path)
                print(f"  Loaded u_vectors for {dataset_name}: shape={self.u_vectors[dataset_name].shape}")

        self._miss_count = 0
        self._hit_count = 0

    def get_indices_by_hash(
        self,
        dataset_name: str,
        img_first: np.ndarray,
        img_last: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Get precomputed LAM indices using content-based hash.

        Args:
            dataset_name: Name of the dataset
            img_first: First frame of the window
            img_last: Last frame of the window

        Returns:
            Array of shape (4,) with LAM code indices (0-15), or None if not found
        """
        if dataset_name not in self.hash_to_idx:
            self._miss_count += 1
            return None

        hash_key = f"{compute_image_hash(img_first)}_{compute_image_hash(img_last)}"
        idx = self.hash_to_idx[dataset_name].get(hash_key)
        if idx is None:
            self._miss_count += 1
            return None

        self._hit_count += 1
        return self.indices[dataset_name][idx]

    def get_u_vector_by_hash(
        self,
        dataset_name: str,
        img_first: np.ndarray,
        img_last: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Get precomputed u_t vector using content-based hash (same hash as LAM indices)."""
        if dataset_name not in self.hash_to_idx or dataset_name not in self.u_vectors:
            return None
        hash_key = f"{compute_image_hash(img_first)}_{compute_image_hash(img_last)}"
        idx = self.hash_to_idx[dataset_name].get(hash_key)
        if idx is None:
            return None
        return self.u_vectors[dataset_name][idx]

    def get_stats(self) -> dict:
        total = self._hit_count + self._miss_count
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": self._hit_count / total if total > 0 else 0.0,
        }


@dataclass
class RLDSBatchTransformWithPrecomputedLAM:
    """
    Batch transform that loads precomputed LAM indices for TwoHead architecture.
    Returns both robot actions (for L1RegressionActionHead) and LAM indices (for LAMCodeHead).

    Unlike RLDSBatchTransformWithLAM, this does NOT run the LAM encoder on-the-fly.
    Instead, it loads precomputed discrete indices from disk using image hash lookup.
    """
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False
    lam_loader: Optional[LAMIndexLoader] = None

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the TwoHead model with precomputed LAM indices."""
        dataset_name = rlds_batch["dataset_name"]
        dataset_name_str = dataset_name.decode() if isinstance(dataset_name, bytes) else dataset_name
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]
        current_action = actions[0]
        future_actions = actions[1:]

        # Look up precomputed LAM indices by image hash
        # Use non-augmented images if available (noaug_image_primary), else fall back to image_primary
        latent_action_idx = None
        if self.lam_loader is not None:
            obs = rlds_batch["observation"]
            if "noaug_image_primary" in obs:
                img_first = obs["noaug_image_primary"][0]
                img_last = obs["noaug_image_primary"][-1]
            else:
                img_first = obs["image_primary"][0]
                img_last = obs["image_primary"][-1]
            latent_action_idx = self.lam_loader.get_indices_by_hash(
                dataset_name_str, img_first, img_last
            )
        if latent_action_idx is None:
            # Fallback: zeros (will produce random gradients but training can still proceed)
            latent_action_idx = np.zeros(4, dtype=np.int64)

        # Look up u_target (MVAE fused latent) if available
        u_target = None
        u_target_valid = False
        if self.lam_loader is not None and hasattr(self.lam_loader, 'u_vectors') and self.lam_loader.u_vectors:
            u_target = self.lam_loader.get_u_vector_by_hash(
                dataset_name_str, img_first, img_last
            )
            if u_target is not None:
                u_target_valid = True
            else:
                # Fallback: zero vector (for ~6% hash misses at episode-start padded windows)
                u_dim = next(iter(self.lam_loader.u_vectors.values())).shape[1]
                u_target = np.zeros(u_dim, dtype=np.float32)

        # Construct prompt with action tokens (needed for model's action mask logic)
        if self.use_minivlm:
            prompt_builder = QwenPromptBuilder("openvla")

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

            if len(input_ids) >= 3:
                input_ids = input_ids[:-3]

            if NUM_TOKENS < len(flattened_action_chunk_string):
                input_ids = input_ids + flattened_action_chunk_string[:NUM_TOKENS]
            else:
                remaining_length = NUM_TOKENS - len(flattened_action_chunk_string)
                extended_array = random.choices(flattened_action_chunk_string, k=remaining_length)
                input_ids = input_ids + flattened_action_chunk_string + extended_array

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
            latent_action_idx=torch.from_numpy(latent_action_idx.astype(np.int64)),  # LAM indices (4,) with values 0-15
        )

        # Add u_target (MVAE fused latent) — always present when u_vectors loaded (zero fallback on miss)
        if u_target is not None:
            return_dict["u_target"] = torch.from_numpy(u_target.astype(np.float32))
            return_dict["u_target_valid"] = torch.tensor(u_target_valid, dtype=torch.bool)

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


class RLDSDatasetWithPrecomputedLAM(IterableDataset):
    """RLDS Dataset that loads precomputed LAM indices for TwoHead architecture."""

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransformWithPrecomputedLAM,
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
