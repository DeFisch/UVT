"""
Precision-Aware Product-of-Experts MVAE for fusing action chunks + LAM codes.

Architecture:
    ActionChunkEncoder: (B, 8, 7) -> mu_a, logvar_a (B, 32)
    LAMCodeEncoder:     (B, 4) int -> mu_z, logvar_z (B, 32)
    PoEFusion:          (mu_a, logvar_a, mu_z, logvar_z) -> pd_mu, pd_logvar (B, 32)
    ActionChunkDecoder: (B, 32) -> (B, 8, 7)
    LAMCodeDecoder:     (B, 32) -> (B, 4, 16) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.fusion_mvae.configs import FusionMVAEConfig


class ActionChunkEncoder(nn.Module):
    """Encodes flattened action chunk (B, 56) -> (mu, logvar) each (B, latent_dim)."""

    def __init__(self, cfg: FusionMVAEConfig):
        super().__init__()
        input_dim = cfg.input_action_dim  # 56
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        # Initialize logvar head for HIGH precision (low variance)
        nn.init.xavier_uniform_(self.logvar_head.weight, gain=cfg.logvar_weight_gain)
        nn.init.constant_(self.logvar_head.bias, cfg.logvar_a_init)  # -2.0

    def forward(self, action_chunk: torch.Tensor) -> tuple:
        """
        Args:
            action_chunk: (B, 8, 7) normalized actions
        Returns:
            mu: (B, latent_dim), logvar: (B, latent_dim)
        """
        x = action_chunk.reshape(action_chunk.shape[0], -1)  # (B, 56)
        h = self.trunk(x)
        return self.mu_head(h), self.logvar_head(h)


class LAMCodeEncoder(nn.Module):
    """Encodes LAM codes (B, 4) int -> one-hot (B, 64) -> (mu, logvar) each (B, latent_dim)."""

    def __init__(self, cfg: FusionMVAEConfig):
        super().__init__()
        input_dim = cfg.input_lam_dim  # 64
        self.num_codes = cfg.num_lam_codes
        self.num_categories = cfg.num_lam_categories
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        # Initialize logvar head for LOW precision (high variance)
        nn.init.xavier_uniform_(self.logvar_head.weight, gain=cfg.logvar_weight_gain)
        nn.init.constant_(self.logvar_head.bias, cfg.logvar_z_init)  # +1.0

    def forward(self, lam_codes: torch.Tensor) -> tuple:
        """
        Args:
            lam_codes: (B, 4) long, values 0-15
        Returns:
            mu: (B, latent_dim), logvar: (B, latent_dim)
        """
        # One-hot encode: (B, 4) -> (B, 4, 16) -> (B, 64)
        one_hot = F.one_hot(lam_codes, self.num_categories).float()  # (B, 4, 16)
        x = one_hot.reshape(one_hot.shape[0], -1)  # (B, 64)
        h = self.trunk(x)
        return self.mu_head(h), self.logvar_head(h)


class ActionChunkDecoder(nn.Module):
    """Decodes latent (B, latent_dim) -> (B, 8, 7) action chunk."""

    def __init__(self, cfg: FusionMVAEConfig):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        self.action_dim = cfg.action_dim
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.input_action_dim),  # 56
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, latent_dim)
        Returns:
            action_chunk: (B, 8, 7)
        """
        return self.net(u).reshape(-1, self.chunk_size, self.action_dim)


class LAMCodeDecoder(nn.Module):
    """Decodes latent (B, latent_dim) -> (B, 4, 16) logits."""

    def __init__(self, cfg: FusionMVAEConfig):
        super().__init__()
        self.num_codes = cfg.num_lam_codes
        self.num_categories = cfg.num_lam_categories
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.num_lam_codes * cfg.num_lam_categories),  # 64
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, latent_dim)
        Returns:
            logits: (B, 4, 16)
        """
        return self.net(u).reshape(-1, self.num_codes, self.num_categories)


class PrecisionAwarePoEMVAE(nn.Module):
    """Full Precision-Aware Product-of-Experts MVAE.

    Fuses action chunk expert (high precision) with LAM code expert (low precision)
    using scalar learnable scaling factors (alpha) — shared across all latent dims
    to prevent the model from partitioning dims between experts.
    """

    def __init__(self, cfg: FusionMVAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoders
        self.action_encoder = ActionChunkEncoder(cfg)
        self.lam_encoder = LAMCodeEncoder(cfg)

        # Scalar learnable log-scaling factors for PoE (NOT per-dimension)
        # This forces both experts to compete on all dims equally
        self.log_alpha_action = nn.Parameter(torch.zeros(1))
        self.log_alpha_z = nn.Parameter(torch.zeros(1))

        # Decoders
        self.action_decoder = ActionChunkDecoder(cfg)
        self.lam_decoder = LAMCodeDecoder(cfg)

    def poe_fuse(self, mu_a, logvar_a, mu_z, logvar_z, eps=1e-8):
        """Precision-Aware Product-of-Experts fusion.

        Args:
            mu_a, logvar_a: Action expert (B, D)
            mu_z, logvar_z: LAM expert (B, D)
        Returns:
            pd_mu, pd_logvar: Fused posterior (B, D)
        """
        var_a = torch.exp(logvar_a) + eps
        var_z = torch.exp(logvar_z) + eps

        # Scalar scaled precisions — broadcasts across all dims equally
        alpha_a = torch.exp(self.log_alpha_action)  # scalar
        alpha_z = torch.exp(self.log_alpha_z)  # (D,)
        T_a = alpha_a / var_a  # (B, D)
        T_z = alpha_z / var_z  # (B, D)

        # Product of Experts
        pd_var = 1.0 / (T_a + T_z)
        pd_mu = pd_var * (T_a * mu_a + T_z * mu_z)
        pd_logvar = torch.log(pd_var)

        return pd_mu, pd_logvar, T_a, T_z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, action_chunk, lam_codes):
        """
        Args:
            action_chunk: (B, 8, 7)
            lam_codes: (B, 4) long
        Returns:
            dict with all outputs for loss computation
        """
        # Encode
        mu_a, logvar_a = self.action_encoder(action_chunk)
        mu_z, logvar_z = self.lam_encoder(lam_codes)

        # Fuse
        pd_mu, pd_logvar, T_a, T_z = self.poe_fuse(mu_a, logvar_a, mu_z, logvar_z)

        # Sample
        u = self.reparameterize(pd_mu, pd_logvar)

        # Decode
        action_recon = self.action_decoder(u)  # (B, 8, 7)
        lam_logits = self.lam_decoder(u)  # (B, 4, 16)

        return {
            'action_recon': action_recon,
            'lam_logits': lam_logits,
            'u': u,
            'pd_mu': pd_mu,
            'pd_logvar': pd_logvar,
            'mu_a': mu_a,
            'logvar_a': logvar_a,
            'mu_z': mu_z,
            'logvar_z': logvar_z,
            'T_a': T_a,
            'T_z': T_z,
        }

    @torch.no_grad()
    def encode_to_u(self, action_chunk, lam_codes, deterministic=True):
        """Encode (action_chunk, lam_codes) -> u_t for precomputation.

        Args:
            action_chunk: (B, 8, 7)
            lam_codes: (B, 4) long
            deterministic: if True, return pd_mu (no sampling)
        Returns:
            u: (B, latent_dim)
        """
        mu_a, logvar_a = self.action_encoder(action_chunk)
        mu_z, logvar_z = self.lam_encoder(lam_codes)
        pd_mu, pd_logvar, _, _ = self.poe_fuse(mu_a, logvar_a, mu_z, logvar_z)
        if deterministic:
            return pd_mu
        return self.reparameterize(pd_mu, pd_logvar)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
