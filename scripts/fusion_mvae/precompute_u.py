"""
Precompute u_t vectors from trained MVAE for all (action_chunk, lam_code) pairs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.fusion_mvae.precompute_u \
        --data_npz scripts/fusion_mvae/outputs/fusion_data_libero_spatial_no_noops.npz \
        --checkpoint scripts/fusion_mvae/outputs/checkpoints/best_model.pt

Outputs (saved to precomputed_lam_indices/libero_spatial_no_noops/):
    u_vectors.npy       - (49946, 32) float32
    decoder_weights.pt  - frozen ActionChunkDecoder state dict
    u_metadata.json     - statistics, config, roundtrip quality
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fusion_mvae.configs import FusionMVAEConfig, Q01, Q99
from scripts.fusion_mvae.dataset import FusionDataset
from scripts.fusion_mvae.model import PrecisionAwarePoEMVAE


def precompute_u(data_npz: Path, checkpoint: Path, device: str = "cuda", output_dir: Path = None):
    # Load checkpoint and config
    print(f"Loading checkpoint from {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt['config']
    cfg = FusionMVAEConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k) and not isinstance(v, Path):
            try:
                setattr(cfg, k, v)
            except (TypeError, AttributeError):
                pass

    if output_dir is None:
        output_dir = cfg.precomputed_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = PrecisionAwarePoEMVAE(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, val L1={ckpt.get('val_l1_action', '?')})")

    # Load data
    data = np.load(data_npz, allow_pickle=True)
    action_chunks = data['action_chunks']  # (N, 8, 7)
    lam_codes = data['lam_codes']  # (N, 4)
    keys = data['keys'].tolist() if 'keys' in data else None
    N = len(action_chunks)
    print(f"Loaded {N} pairs")

    dataset = FusionDataset(action_chunks, lam_codes)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # Encode all pairs
    all_u = []
    all_recon = []
    all_lam_logits = []

    with torch.no_grad():
        for ac_batch, lc_batch in loader:
            ac_batch = ac_batch.to(device)
            lc_batch = lc_batch.to(device)

            u = model.encode_to_u(ac_batch, lc_batch, deterministic=True)
            all_u.append(u.cpu())

            # Also compute reconstruction for quality diagnostics
            recon = model.action_decoder(u)
            all_recon.append(recon.cpu())

            logits = model.lam_decoder(u)
            all_lam_logits.append(logits.cpu())

    u_vectors = torch.cat(all_u, dim=0).numpy()  # (N, 32)
    recon_chunks = torch.cat(all_recon, dim=0).numpy()  # (N, 8, 7)
    lam_logits = torch.cat(all_lam_logits, dim=0).numpy()  # (N, 4, 16)

    print(f"\nu_vectors: shape={u_vectors.shape}, dtype={u_vectors.dtype}")
    print(f"  mean={u_vectors.mean():.4f}, std={u_vectors.std():.4f}")
    print(f"  min={u_vectors.min():.4f}, max={u_vectors.max():.4f}")

    # Per-dimension statistics
    print("\nPer-dimension u_t statistics:")
    for d in range(min(cfg.latent_dim, 8)):
        col = u_vectors[:, d]
        print(f"  dim {d}: mean={col.mean():.4f}, std={col.std():.4f}, range=[{col.min():.4f}, {col.max():.4f}]")
    if cfg.latent_dim > 8:
        print(f"  ... ({cfg.latent_dim - 8} more dimensions)")

    # Reconstruction quality
    l1_per_sample = np.abs(recon_chunks - action_chunks).mean(axis=(1, 2))  # (N,)
    l1_overall = l1_per_sample.mean()
    print(f"\nRoundtrip reconstruction quality (encode -> decode):")
    print(f"  Overall L1: {l1_overall:.5f}")
    print(f"  L1 median: {np.median(l1_per_sample):.5f}, 95th percentile: {np.percentile(l1_per_sample, 95):.5f}")

    # Per-step L1
    print("\n  Per-step L1:")
    for t in range(cfg.chunk_size):
        step_l1 = np.abs(recon_chunks[:, t, :] - action_chunks[:, t, :]).mean()
        print(f"    step {t}: {step_l1:.5f}")

    # Per-dim L1
    print("\n  Per-action-dim L1:")
    dim_names_7 = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
    dim_names_14 = ['x_L', 'y_L', 'z_L', 'rx_L', 'ry_L', 'rz_L', 'grip_L',
                    'x_R', 'y_R', 'z_R', 'rx_R', 'ry_R', 'rz_R', 'grip_R']
    dim_names = dim_names_14 if cfg.action_dim == 14 else dim_names_7 if cfg.action_dim == 7 else [f"d{i}" for i in range(cfg.action_dim)]
    for d in range(cfg.action_dim):
        dim_l1 = np.abs(recon_chunks[:, :, d] - action_chunks[:, :, d]).mean()
        print(f"    dim {d} ({dim_names[d]}): {dim_l1:.5f}")

    # LAM accuracy
    lam_preds = lam_logits.argmax(axis=-1)  # (N, 4)
    lam_acc = (lam_preds == lam_codes).mean()
    per_code_acc = [(lam_preds[:, c] == lam_codes[:, c]).mean() for c in range(cfg.num_lam_codes)]
    print(f"\n  LAM code accuracy: {lam_acc:.4f}")
    for c in range(cfg.num_lam_codes):
        print(f"    code {c}: {per_code_acc[c]:.4f}")

    # Save u_vectors
    u_path = output_dir / "u_vectors.npy"
    np.save(u_path, u_vectors.astype(np.float32))
    print(f"\nSaved u_vectors to {u_path}")

    # Save decoder weights
    decoder_path = output_dir / "decoder_weights.pt"
    torch.save({
        'action_decoder': model.action_decoder.state_dict(),
        'lam_decoder': model.lam_decoder.state_dict(),
        'config': cfg_dict,
    }, decoder_path)
    print(f"Saved decoder weights to {decoder_path}")

    # Save metadata
    metadata = {
        'num_samples': N,
        'latent_dim': cfg.latent_dim,
        'chunk_size': cfg.chunk_size,
        'action_dim': cfg.action_dim,
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
        'q01': Q01,
        'q99': Q99,
    }
    meta_path = output_dir / "u_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    return u_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute u_t vectors from trained MVAE")
    parser.add_argument("--data_npz", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    precompute_u(
        Path(args.data_npz),
        Path(args.checkpoint),
        args.device,
        Path(args.output_dir) if args.output_dir else None,
    )
