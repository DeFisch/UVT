"""
Compute u_vectors and decoder_weights for any dataset using an already-trained MVAE.

Reads precomputed LAM indices and RLDS actions, encodes through MVAE, saves outputs.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/compute_u_for_subset.py \
        --dataset_name libero_spatial_no_noops_10pct \
        --mvae_checkpoint scripts/fusion_mvae/outputs/checkpoints_scalar_alpha/best_model.pt \
        --precompute_num_shards 1

    # For lift_pot (14-dim actions, no gripper inversion):
    CUDA_VISIBLE_DEVICES=3 python scripts/compute_u_for_subset.py \
        --dataset_name lift_pot \
        --data_root /srv/nvme0/ucinlp/daniel \
        --mvae_checkpoint scripts/fusion_mvae/outputs_lift_pot/checkpoints/best_model.pt \
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
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fusion_mvae.configs import FusionMVAEConfig
from scripts.fusion_mvae.model import PrecisionAwarePoEMVAE


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
    stats_files = list(dataset_path.glob("dataset_statistics_*.json"))
    if not stats_files:
        raise FileNotFoundError(f"No dataset_statistics_*.json in {dataset_path}")
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
    parser.add_argument("--mvae_checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="modified_libero_rlds")
    parser.add_argument("--precomputed_dir", type=str, default="precomputed_lam_indices")
    parser.add_argument("--precompute_num_shards", type=int, default=1)
    parser.add_argument("--no_gripper_transform", action="store_true",
                        help="Skip gripper inversion (for non-LIBERO datasets)")
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    precomputed_dir = Path(args.precomputed_dir) / args.dataset_name
    output_dir = precomputed_dir

    # Load Q01/Q99 from dataset statistics
    q01, q99 = load_dataset_q01_q99(data_root, args.dataset_name)

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

    # Load RLDS dataset and extract action chunks
    import tensorflow_datasets as tfds

    dataset_path = data_root / args.dataset_name
    version_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if version_dirs:
        dataset_path = version_dirs[0]
    builder = tfds.builder_from_directory(str(dataset_path))

    num_shards_total = args.precompute_num_shards
    chunk_size = args.chunk_size

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

            actions_norm = np.clip(
                2.0 * (raw_actions - q01) / (q99 - q01 + 1e-8) - 1.0,
                -1.0, 1.0
            ).astype(np.float32)

            for step_start, key in eps_needed[local_idx]:
                if step_start + chunk_size > num_steps:
                    continue
                action_chunk = actions_norm[step_start:step_start + chunk_size]
                lam_code = indices[key_to_idx[key]]

                all_action_chunks.append(action_chunk)
                all_lam_codes.append(lam_code)
                all_keys_ordered.append(key)

    action_chunks = np.stack(all_action_chunks, axis=0)
    lam_codes = np.stack(all_lam_codes, axis=0)
    N = len(all_keys_ordered)
    action_dim = action_chunks.shape[-1]
    print(f"\nExtracted {N} pairs, action_chunks: {action_chunks.shape}, lam_codes: {lam_codes.shape}")

    # Load MVAE model
    print(f"\nLoading MVAE checkpoint from {args.mvae_checkpoint}")
    ckpt = torch.load(args.mvae_checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt['config']
    mvae_cfg = FusionMVAEConfig()
    for k, v in cfg_dict.items():
        if hasattr(mvae_cfg, k) and not isinstance(v, Path):
            try:
                setattr(mvae_cfg, k, v)
            except (TypeError, AttributeError):
                pass

    model = PrecisionAwarePoEMVAE(mvae_cfg).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded (epoch {ckpt.get('epoch', '?')}, val L1={ckpt.get('val_l1_action', '?')})")

    # Encode all pairs to u_vectors
    ac_tensor = torch.from_numpy(action_chunks).float()
    lc_tensor = torch.from_numpy(lam_codes).long()
    dataset = TensorDataset(ac_tensor, lc_tensor)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    all_u = []
    all_recon = []
    all_lam_logits = []

    print("Computing u_vectors...")
    with torch.no_grad():
        for ac_batch, lc_batch in tqdm(loader, desc="Encoding"):
            ac_batch = ac_batch.to(args.device)
            lc_batch = lc_batch.to(args.device)

            u = model.encode_to_u(ac_batch, lc_batch, deterministic=True)
            all_u.append(u.cpu())

            recon = model.action_decoder(u)
            all_recon.append(recon.cpu())

            logits = model.lam_decoder(u)
            all_lam_logits.append(logits.cpu())

    u_vectors = torch.cat(all_u, dim=0).numpy()
    recon_chunks = torch.cat(all_recon, dim=0).numpy()
    lam_logits = torch.cat(all_lam_logits, dim=0).numpy()

    # Quality diagnostics
    l1_per_sample = np.abs(recon_chunks - action_chunks).mean(axis=(1, 2))
    l1_overall = l1_per_sample.mean()
    lam_preds = lam_logits.argmax(axis=-1)
    lam_acc = (lam_preds == lam_codes).mean()
    per_code_acc = [(lam_preds[:, c] == lam_codes[:, c]).mean() for c in range(4)]

    print(f"\nu_vectors: shape={u_vectors.shape}")
    print(f"  mean={u_vectors.mean():.4f}, std={u_vectors.std():.4f}")
    print(f"Roundtrip L1: {l1_overall:.5f}, 95th pct: {np.percentile(l1_per_sample, 95):.5f}")
    print(f"LAM accuracy: {lam_acc:.4f}")

    # Save
    np.save(output_dir / "u_vectors.npy", u_vectors.astype(np.float32))
    print(f"\nSaved u_vectors.npy ({u_vectors.shape})")

    torch.save({
        'action_decoder': model.action_decoder.state_dict(),
        'lam_decoder': model.lam_decoder.state_dict(),
        'config': cfg_dict,
    }, output_dir / "decoder_weights.pt")
    print("Saved decoder_weights.pt")

    metadata = {
        'num_samples': N,
        'latent_dim': mvae_cfg.latent_dim,
        'chunk_size': mvae_cfg.chunk_size,
        'action_dim': mvae_cfg.action_dim,
        'u_shape': list(u_vectors.shape),
        'u_mean': float(u_vectors.mean()),
        'u_std': float(u_vectors.std()),
        'u_min': float(u_vectors.min()),
        'u_max': float(u_vectors.max()),
        'roundtrip_l1': float(l1_overall),
        'roundtrip_l1_95pct': float(np.percentile(l1_per_sample, 95)),
        'lam_accuracy': float(lam_acc),
        'per_code_accuracy': [float(a) for a in per_code_acc],
        'checkpoint_epoch': ckpt.get('epoch'),
        'checkpoint_val_l1': ckpt.get('val_l1_action'),
        'q01': q01.tolist(),
        'q99': q99.tolist(),
    }
    with open(output_dir / "u_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved u_metadata.json")


if __name__ == "__main__":
    main()
