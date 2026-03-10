"""
Training loop for Precision-Aware PoE MVAE.

Usage:
    # First extract data (only needed once):
    python -m scripts.fusion_mvae.dataset

    # Then train:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.fusion_mvae.train \
        --data_npz scripts/fusion_mvae/outputs/fusion_data_libero_spatial_no_noops.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fusion_mvae.configs import FusionMVAEConfig
from scripts.fusion_mvae.dataset import load_fusion_data
from scripts.fusion_mvae.model import PrecisionAwarePoEMVAE


def compute_loss(model_out, action_chunk, lam_codes, beta, lambda_z):
    """Compute total loss with components.

    L = L1_action + lambda_z * CE_lam + beta * KL(q || N(0,I))
    """
    # L1 action reconstruction
    l1_action = F.l1_loss(model_out['action_recon'], action_chunk)

    # Cross-entropy for LAM code reconstruction
    # logits: (B, 4, 16), codes: (B, 4) -> reshape for CE
    ce_lam = F.cross_entropy(
        model_out['lam_logits'].reshape(-1, 16),
        lam_codes.reshape(-1),
    )

    # KL divergence: KL(N(mu, sigma^2) || N(0, I))
    pd_mu = model_out['pd_mu']
    pd_logvar = model_out['pd_logvar']
    kl = -0.5 * torch.mean(1.0 + pd_logvar - pd_mu.pow(2) - pd_logvar.exp())

    total = l1_action + lambda_z * ce_lam + beta * kl

    return {
        'total': total,
        'l1_action': l1_action,
        'ce_lam': ce_lam,
        'kl': kl,
    }


def get_beta(step, warmup_steps, beta_max):
    """Linear beta warmup from 0 to beta_max."""
    if step >= warmup_steps:
        return beta_max
    return beta_max * step / warmup_steps


@torch.no_grad()
def validate(model, val_loader, beta, lambda_z, device):
    model.eval()
    totals = {'l1_action': 0, 'ce_lam': 0, 'kl': 0, 'total': 0}
    correct = 0
    total_codes = 0
    n_batches = 0

    for action_chunk, lam_codes in val_loader:
        action_chunk = action_chunk.to(device)
        lam_codes = lam_codes.to(device)

        out = model(action_chunk, lam_codes)
        losses = compute_loss(out, action_chunk, lam_codes, beta, lambda_z)

        for k in totals:
            totals[k] += losses[k].item()

        # LAM accuracy
        preds = out['lam_logits'].argmax(dim=-1)  # (B, 4)
        correct += (preds == lam_codes).sum().item()
        total_codes += lam_codes.numel()
        n_batches += 1

    model.train()
    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avg['lam_accuracy'] = correct / max(total_codes, 1)
    return avg


def train(cfg: FusionMVAEConfig, data_npz: Path, device: str = "cuda"):
    # Setup wandb
    try:
        import wandb
        wandb_kwargs = {"project": cfg.wandb_project, "config": vars(cfg)}
        if cfg.wandb_entity:
            wandb_kwargs["entity"] = cfg.wandb_entity
        wandb.init(**wandb_kwargs)
        use_wandb = True
    except (ImportError, wandb.errors.UsageError):
        print("wandb not available, logging to stdout only")
        use_wandb = False

    # Data
    train_loader, val_loader = load_fusion_data(data_npz, cfg)

    # Model
    model = PrecisionAwarePoEMVAE(cfg).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr_min)

    # Checkpoint directory
    ckpt_dir = cfg.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_l1 = float('inf')
    global_step = 0
    t_start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = {'l1_action': 0, 'ce_lam': 0, 'kl': 0, 'total': 0}
        n_batches = 0

        for action_chunk, lam_codes in train_loader:
            action_chunk = action_chunk.to(device)
            lam_codes = lam_codes.to(device)

            beta = get_beta(global_step, cfg.beta_warmup_steps, cfg.beta_max)

            out = model(action_chunk, lam_codes)
            losses = compute_loss(out, action_chunk, lam_codes, beta, cfg.lambda_z)

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1
            global_step += 1

            # Step-level logging (every 50 steps)
            if global_step % 50 == 0 and use_wandb:
                # Precision diagnostics
                with torch.no_grad():
                    T_a_mean = out['T_a'].mean().item()
                    T_z_mean = out['T_z'].mean().item()
                    precision_ratio = T_a_mean / max(T_z_mean, 1e-8)

                wandb.log({
                    'train/l1_action': losses['l1_action'].item(),
                    'train/ce_lam': losses['ce_lam'].item(),
                    'train/kl': losses['kl'].item(),
                    'train/total': losses['total'].item(),
                    'train/beta': beta,
                    'train/lr': scheduler.get_last_lr()[0],
                    'precision/logvar_a_mean': out['logvar_a'].mean().item(),
                    'precision/logvar_z_mean': out['logvar_z'].mean().item(),
                    'precision/T_a_mean': T_a_mean,
                    'precision/T_z_mean': T_z_mean,
                    'precision/ratio_a_over_z': precision_ratio,
                    'precision/log_alpha_action_mean': model.log_alpha_action.mean().item(),
                    'precision/log_alpha_z_mean': model.log_alpha_z.mean().item(),
                    'step': global_step,
                })

        # Epoch averages
        epoch_avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # Validation
        beta_val = get_beta(global_step, cfg.beta_warmup_steps, cfg.beta_max)
        val_metrics = validate(model, val_loader, beta_val, cfg.lambda_z, device)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch+1:3d}/{cfg.epochs} ({elapsed:.0f}s) | "
              f"train L1={epoch_avg['l1_action']:.4f} CE={epoch_avg['ce_lam']:.4f} KL={epoch_avg['kl']:.4f} | "
              f"val L1={val_metrics['l1_action']:.4f} CE={val_metrics['ce_lam']:.4f} acc={val_metrics['lam_accuracy']:.3f}")

        if use_wandb:
            wandb.log({
                'val/l1_action': val_metrics['l1_action'],
                'val/ce_lam': val_metrics['ce_lam'],
                'val/kl': val_metrics['kl'],
                'val/total': val_metrics['total'],
                'val/lam_accuracy': val_metrics['lam_accuracy'],
                'epoch': epoch + 1,
                'step': global_step,
            })

        # Checkpoint best model by val L1
        if val_metrics['l1_action'] < best_val_l1:
            best_val_l1 = val_metrics['l1_action']
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_l1_action': best_val_l1,
                'val_lam_accuracy': val_metrics['lam_accuracy'],
                'config': vars(cfg),
            }, ckpt_path)
            print(f"  -> New best val L1: {best_val_l1:.4f}, saved to {ckpt_path}")

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'config': vars(cfg),
            }, ckpt_dir / f"epoch_{epoch+1}.pt")

    print(f"\nTraining complete. Best val L1: {best_val_l1:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}")

    if use_wandb:
        wandb.finish()

    return ckpt_dir / "best_model.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Precision-Aware PoE MVAE")
    parser.add_argument("--data_npz", type=str, required=True, help="Path to fusion_data .npz file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta_max", type=float, default=None)
    parser.add_argument("--lambda_z", type=float, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--logvar_a_init", type=float, default=None)
    parser.add_argument("--logvar_z_init", type=float, default=None)
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = FusionMVAEConfig()

    # Override config with CLI args
    for field_name in ['epochs', 'batch_size', 'lr', 'beta_max', 'lambda_z',
                       'latent_dim', 'logvar_a_init', 'logvar_z_init',
                       'action_dim', 'chunk_size', 'wandb_project', 'wandb_entity']:
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(cfg, field_name, val)
    if args.output_dir is not None:
        cfg.output_dir = Path(args.output_dir)

    train(cfg, Path(args.data_npz), args.device)
