"""
Extract (action_chunk, lam_code) pairs from any RLDS dataset + precomputed LAM indices.

Unlike fusion_mvae/dataset.py which hardcodes num_shards_total=5 and libero-specific
gripper transform, this script is generic for any dataset.

Usage:
    python scripts/extract_fusion_data_generic.py \
        --dataset_name lift_pot \
        --data_root /srv/nvme0/ucinlp/daniel \
        --precomputed_dir precomputed_lam_indices/lift_pot \
        --output_npz scripts/fusion_mvae/outputs_lift_pot/fusion_data_lift_pot.npz \
        --precompute_num_shards 1 \
        --no_gripper_transform
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_key(key: str):
    m = re.search(r'_s(\d+)_(\d+)_(\d+)$', key)
    if m is None:
        raise ValueError(f"Cannot parse key: {key}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_dataset_q01_q99(data_root: Path, dataset_name: str):
    """Load Q01/Q99 from dataset_statistics JSON."""
    dataset_path = data_root / dataset_name
    version_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if version_dirs:
        dataset_path = version_dirs[0]

    # Find dataset_statistics file
    stats_files = list(dataset_path.glob("dataset_statistics_*.json"))
    if not stats_files:
        raise FileNotFoundError(f"No dataset_statistics_*.json in {dataset_path}")
    # Use most recent
    stats_file = max(stats_files, key=lambda p: p.stat().st_mtime)
    with open(stats_file) as f:
        stats = json.load(f)

    q01 = np.array(stats["action"]["q01"], dtype=np.float32)
    q99 = np.array(stats["action"]["q99"], dtype=np.float32)
    print(f"  Loaded Q01/Q99 from {stats_file.name} (action_dim={len(q01)})")
    return q01, q99


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--precomputed_dir", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--precompute_num_shards", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--no_gripper_transform", action="store_true",
                        help="Skip gripper inversion (for non-LIBERO datasets like ALOHA/lift_pot)")
    args = parser.parse_args()

    import tensorflow_datasets as tfds

    data_root = Path(args.data_root)
    precomputed_dir = Path(args.precomputed_dir)
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load Q01/Q99 from dataset statistics
    q01, q99 = load_dataset_q01_q99(data_root, args.dataset_name)
    action_dim = len(q01)

    # Load precomputed LAM indices
    print("Loading precomputed LAM indices...")
    with open(precomputed_dir / "keys.json") as f:
        keys = json.load(f)
    with open(precomputed_dir / "key_to_idx.json") as f:
        key_to_idx = json.load(f)
    indices = np.load(precomputed_dir / "indices.npy")
    print(f"  {len(keys)} keys, indices shape {indices.shape}")

    # Group keys by (shard_idx, local_ep_idx)
    episode_windows = defaultdict(list)
    for key in keys:
        shard_idx, local_ep_idx, step_start = parse_key(key)
        episode_windows[(shard_idx, local_ep_idx)].append((step_start, key))
    for k in episode_windows:
        episode_windows[k].sort()

    shards_needed = sorted(set(s for s, _ in episode_windows.keys()))
    print(f"  Shards needed: {shards_needed}")

    # Load RLDS dataset
    dataset_path = data_root / args.dataset_name
    version_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if version_dirs:
        dataset_path = version_dirs[0]
    builder = tfds.builder_from_directory(str(dataset_path))

    num_shards_total = args.precompute_num_shards

    all_action_chunks = []
    all_lam_codes = []
    all_keys_ordered = []

    for shard_idx in shards_needed:
        pct_start = (shard_idx * 100) // num_shards_total
        pct_end = ((shard_idx + 1) * 100) // num_shards_total
        split_str = f"train[{pct_start}%:{pct_end}%]"
        print(f"\nProcessing shard {shard_idx}: split '{split_str}'")

        read_config = tfds.ReadConfig(
            interleave_cycle_length=16,
            interleave_block_length=1,
            num_parallel_calls_for_decode=16,
            num_parallel_calls_for_interleave_files=16,
        )
        ds = builder.as_dataset(split=split_str, read_config=read_config)
        ds = ds.prefetch(buffer_size=8)

        eps_needed = {ep: wins for (s, ep), wins in episode_windows.items() if s == shard_idx}

        for local_idx, episode in enumerate(tqdm(ds, desc=f"Shard {shard_idx}")):
            if local_idx not in eps_needed:
                continue

            steps = list(episode['steps'])
            num_steps = len(steps)
            raw_actions = np.stack([step['action'].numpy() for step in steps], axis=0)

            # Apply gripper transform only for LIBERO-style datasets
            if not args.no_gripper_transform:
                raw_actions[:, -1] = 1.0 - np.clip(raw_actions[:, -1], 0.0, 1.0)

            # Normalize to [-1, 1]
            actions_norm = np.clip(
                2.0 * (raw_actions - q01) / (q99 - q01 + 1e-8) - 1.0,
                -1.0, 1.0
            ).astype(np.float32)

            for step_start, key in eps_needed[local_idx]:
                if step_start + args.chunk_size > num_steps:
                    continue
                action_chunk = actions_norm[step_start:step_start + args.chunk_size]
                lam_code = indices[key_to_idx[key]]

                all_action_chunks.append(action_chunk)
                all_lam_codes.append(lam_code)
                all_keys_ordered.append(key)

    action_chunks = np.stack(all_action_chunks, axis=0)
    lam_codes = np.stack(all_lam_codes, axis=0)

    print(f"\nExtracted {len(all_keys_ordered)} pairs")
    print(f"  action_chunks: {action_chunks.shape}, dtype={action_chunks.dtype}")
    print(f"  lam_codes: {lam_codes.shape}")
    print(f"  action range: [{action_chunks.min():.3f}, {action_chunks.max():.3f}]")

    # Save Q01/Q99 alongside data for downstream use
    np.savez_compressed(
        output_path,
        action_chunks=action_chunks.astype(np.float32),
        lam_codes=lam_codes.astype(np.int16),
        keys=np.array(all_keys_ordered, dtype=str),
        q01=q01,
        q99=q99,
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
