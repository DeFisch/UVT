"""Hyperparameter configuration for Precision-Aware PoE MVAE."""

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# BOUNDS_Q99 from dataset_statistics JSON (after gripper transform: clip to [0,1], invert)
Q01 = [-0.7454732114076613, -0.6616071462631226, -0.9375, -0.1071428582072258,
       -0.20678570866584778, -0.1842857152223587, 0.0]
Q99 = [0.9375, 0.8758928775787354, 0.9321428537368774, 0.1039285734295845,
       0.17678570747375488, 0.14571428298950195, 1.0]


@dataclass
class FusionMVAEConfig:
    # Architecture
    latent_dim: int = 32
    hidden_dim: int = 256
    chunk_size: int = 8
    action_dim: int = 7
    num_lam_codes: int = 4
    num_lam_categories: int = 16

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    lr_min: float = 1e-5
    epochs: int = 200
    grad_clip: float = 1.0

    # Loss weights
    lambda_z: float = 0.5
    beta_max: float = 0.01
    beta_warmup_steps: int = 2000

    # Precision initialization
    logvar_a_init: float = -2.0  # var=0.135, precision=7.39 (HIGH)
    logvar_z_init: float = 1.0   # var=2.718, precision=0.368 (LOW)
    logvar_weight_gain: float = 0.1

    # Data
    val_fraction: float = 0.1
    seed: int = 42
    num_workers: int = 4

    # Paths (relative to REPO_ROOT)
    data_root: Path = field(default_factory=lambda: REPO_ROOT / "modified_libero_rlds")
    dataset_name: str = "libero_spatial_no_noops"
    precomputed_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "precomputed_lam_indices" / "libero_spatial_no_noops"
    )
    output_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "scripts" / "fusion_mvae" / "outputs"
    )

    # Wandb
    wandb_project: str = "fusion-mvae"
    wandb_entity: str = ""

    @property
    def input_action_dim(self) -> int:
        return self.chunk_size * self.action_dim  # 56

    @property
    def input_lam_dim(self) -> int:
        return self.num_lam_codes * self.num_lam_categories  # 64
