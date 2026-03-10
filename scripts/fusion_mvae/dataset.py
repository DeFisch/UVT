"""
Data extraction and FusionDataset for Precision-Aware PoE MVAE.

Extracts (action_chunk, lam_code) pairs from RLDS data and precomputed LAM indices.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fusion_mvae.configs import FusionMVAEConfig, Q01, Q99


def parse_key(key: str):
    """Parse key like 'libero_spatial_no_noops_s0_42_7' -> (shard_idx, local_ep_idx, step_start)."""
    # Match the shard/episode/step suffix: _s{shard}_{ep}_{step}
    m = re.search(r'_s(\d+)_(\d+)_(\d+)$', key)
    if m is None:
        raise ValueError(f"Cannot parse key: {key}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def extract_fusion_data(cfg: FusionMVAEConfig, output_path: Path = None):
    """Extract all (action_chunk, lam_code) pairs and save as .npz.

    Steps:
    1. Load precomputed keys, key_to_idx, indices
    2. Group keys by (shard_idx, local_ep_idx) so we load each episode once
    3. For each shard, load RLDS split and extract action chunks
    4. Apply gripper transform + BOUNDS_Q99 normalization
    5. Save as .npz
    """
    import tensorflow_datasets as tfds

    if output_path is None:
        output_path = cfg.output_dir / f"fusion_data_{cfg.dataset_name}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load precomputed LAM indices
    print("Loading precomputed LAM indices...")
    with open(cfg.precomputed_dir / "keys.json") as f:
        keys = json.load(f)
    with open(cfg.precomputed_dir / "key_to_idx.json") as f:
        key_to_idx = json.load(f)
    indices = np.load(cfg.precomputed_dir / "indices.npy")  # (N, 4)
    print(f"  Loaded {len(keys)} keys, indices shape {indices.shape}")

    # Group keys by (shard_idx, local_ep_idx) for efficient episode-level loading
    # key -> (shard_idx, local_ep_idx, step_start)
    episode_windows = defaultdict(list)  # (shard, local_ep) -> [(step_start, key), ...]
    for key in keys:
        shard_idx, local_ep_idx, step_start = parse_key(key)
        episode_windows[(shard_idx, local_ep_idx)].append((step_start, key))

    # Sort by step_start within each episode
    for k in episode_windows:
        episode_windows[k].sort()

    # Find unique shards needed
    shards_needed = sorted(set(s for s, _ in episode_windows.keys()))
    print(f"  Need shards: {shards_needed}")

    # RLDS dataset path
    dataset_path = cfg.data_root / cfg.dataset_name
    version_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if version_dirs:
        dataset_path = version_dirs[0]
    builder = tfds.builder_from_directory(str(dataset_path))

    q01 = np.array(Q01, dtype=np.float32)
    q99 = np.array(Q99, dtype=np.float32)
    num_shards_total = 5  # libero_spatial_no_noops uses 5 shards

    all_action_chunks = []
    all_lam_codes = []
    all_keys_ordered = []

    for shard_idx in shards_needed:
        pct_start = (shard_idx * 100) // num_shards_total
        pct_end = ((shard_idx + 1) * 100) // num_shards_total
        split_str = f"train[{pct_start}%:{pct_end}%]"
        print(f"\nProcessing shard {shard_idx}: split '{split_str}'")

        # CRITICAL: Must match precompute_lam_indices.py ReadConfig exactly,
        # otherwise episode ordering differs and local_idx maps to wrong episodes.
        read_config = tfds.ReadConfig(
            interleave_cycle_length=16,
            interleave_block_length=1,
            num_parallel_calls_for_decode=16,
            num_parallel_calls_for_interleave_files=16,
        )
        ds = builder.as_dataset(split=split_str, read_config=read_config)
        ds = ds.prefetch(buffer_size=8)

        # Which local_ep_idxs do we need from this shard?
        eps_needed = {ep: wins for (s, ep), wins in episode_windows.items() if s == shard_idx}

        for local_idx, episode in enumerate(tqdm(ds, desc=f"Shard {shard_idx}")):
            if local_idx not in eps_needed:
                continue

            steps = list(episode['steps'])
            num_steps = len(steps)

            # Extract raw actions (7D)
            raw_actions = np.stack([step['action'].numpy() for step in steps], axis=0)  # (T, 7)

            # Apply gripper transform: clip to [0,1], then invert (1 - x)
            raw_actions[:, -1] = 1.0 - np.clip(raw_actions[:, -1], 0.0, 1.0)

            # Apply BOUNDS_Q99 normalization
            actions_norm = np.clip(
                2.0 * (raw_actions - q01) / (q99 - q01 + 1e-8) - 1.0,
                -1.0, 1.0
            ).astype(np.float32)

            # Extract action chunks for each window
            for step_start, key in eps_needed[local_idx]:
                if step_start + cfg.chunk_size > num_steps:
                    print(f"  WARNING: skipping {key}, step_start={step_start} + chunk_size={cfg.chunk_size} > num_steps={num_steps}")
                    continue

                action_chunk = actions_norm[step_start : step_start + cfg.chunk_size]  # (8, 7)
                lam_code = indices[key_to_idx[key]]  # (4,)

                all_action_chunks.append(action_chunk)
                all_lam_codes.append(lam_code)
                all_keys_ordered.append(key)

            # Remove from eps_needed to track progress
            del eps_needed[local_idx]

        if eps_needed:
            print(f"  WARNING: {len(eps_needed)} episodes not found in shard {shard_idx}: {list(eps_needed.keys())[:5]}")

    action_chunks = np.stack(all_action_chunks, axis=0)  # (N, 8, 7)
    lam_codes = np.stack(all_lam_codes, axis=0)  # (N, 4)

    print(f"\nExtracted {len(all_keys_ordered)} pairs")
    print(f"  action_chunks: {action_chunks.shape}, dtype={action_chunks.dtype}")
    print(f"  lam_codes: {lam_codes.shape}, dtype={lam_codes.dtype}")
    print(f"  action range: [{action_chunks.min():.3f}, {action_chunks.max():.3f}]")
    print(f"  lam code range: [{lam_codes.min()}, {lam_codes.max()}]")

    # Spot-check: per-dim action stats
    for d in range(cfg.action_dim):
        vals = action_chunks[:, :, d]
        print(f"  dim {d}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

    np.savez_compressed(
        output_path,
        action_chunks=action_chunks.astype(np.float32),
        lam_codes=lam_codes.astype(np.int16),
        keys=np.array(all_keys_ordered, dtype=str),
    )
    print(f"Saved to {output_path}")
    return output_path


class FusionDataset(Dataset):
    """Simple map-style dataset for (action_chunk, lam_code) pairs."""

    def __init__(self, action_chunks: np.ndarray, lam_codes: np.ndarray):
        """
        Args:
            action_chunks: (N, 8, 7) float32, normalized to [-1, 1]
            lam_codes: (N, 4) int16, values 0-15
        """
        self.action_chunks = torch.from_numpy(action_chunks).float()  # (N, 8, 7)
        self.lam_codes = torch.from_numpy(lam_codes).long()  # (N, 4)

    def __len__(self):
        return len(self.action_chunks)

    def __getitem__(self, idx):
        return self.action_chunks[idx], self.lam_codes[idx]


def load_fusion_data(npz_path: Path, cfg: FusionMVAEConfig):
    """Load .npz and return train/val DataLoaders."""
    data = np.load(npz_path, allow_pickle=True)
    action_chunks = data['action_chunks']  # (N, 8, 7)
    lam_codes = data['lam_codes']  # (N, 4)

    print(f"Loaded {len(action_chunks)} pairs from {npz_path}")

    full_dataset = FusionDataset(action_chunks, lam_codes)

    # 90/10 train/val split
    n_val = int(len(full_dataset) * cfg.val_fraction)
    n_train = len(full_dataset) - n_val
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    print(f"Train: {n_train} samples, {len(train_loader)} batches")
    print(f"Val:   {n_val} samples, {len(val_loader)} batches")
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract fusion data from RLDS + precomputed LAM indices")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path")
    args = parser.parse_args()

    cfg = FusionMVAEConfig()
    output_path = Path(args.output) if args.output else None
    extract_fusion_data(cfg, output_path)
