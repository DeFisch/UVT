from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer

from latent_action_model.genie.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer


from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class UncontrolledDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(UncontrolledDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Load T5 text encoder model
        self.text_encoder = T5EncoderModel.from_pretrained('./t5-base')
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(768, model_dim)

        # Load T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('./t5-base')

    def encode_text(self, lang: List):
        # Tokenize the batch with padding to the longest sequence
        encoding = self.tokenizer(lang, return_tensors="pt", padding=True).to(self.device) 

        # Access the input IDs and attention masks
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Access the last hidden states
        last_hidden_states = encoder_outputs.last_hidden_state

        return last_hidden_states, attention_mask

    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.encoder(padded_patches, lang_embed, attention_mask) 
      
        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        lang_embed, attention_mask = self.encode_text(batch["task_instruction"])
        lang_embed = self.lang_proj(lang_embed)
        attention_mask = torch.cat([torch.ones((B, self.num_codes + (H // self.patch_size)**2)).to(self.device),
                                    attention_mask],
                                    dim = -1)

        outputs = self.vq_encode(batch["videos"], repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1)) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches, lang_embed.unsqueeze(1), attention_mask)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device




class ControllableDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(ControllableDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.action_up_uncontrol = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.vq_action = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
        self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        # we only optimize the new tack-centric codebook in stage-2
        self.vq.requires_grad_(False)


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)
        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        # Encode
        z = self.encoder(padded_patches) 
      
        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(z_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, self.latent_dim)

        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            "indices": indices,
            "indices_uncontrol": indices_uncontrol,
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        outputs = self.vq_encode(batch["videos"]) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])

        # Decode
        video_action_patches = torch.cat([self.action_up(outputs["z_q"]), 
                                          self.action_up_uncontrol(outputs["z_q_uncontrol"]), 
                                          video_patches],
                                          dim=2)
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, -video_patches.shape[2]:] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device

import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

# laq_model imports disabled for local testing - DanActionModel not needed
# import sys
# sys.path.append("/fs/scratch/PAS2099/danielf/geometry_grounded_latents/LAPA/laq")
# from laq_model.attention import Transformer
# from laq_model.nsvq import NSVQ
# from laq_model.mast3r_latents import Mast3rLatentExtractor

# Stub classes to prevent NameError when DanActionModel is parsed (not used for training)
class Transformer:
    def __init__(self, *args, **kwargs): raise NotImplementedError("laq_model not available")
class NSVQ:
    def __init__(self, *args, **kwargs): raise NotImplementedError("laq_model not available")
class Mast3rLatentExtractor:
    def __init__(self, *args, **kwargs): raise NotImplementedError("laq_model not available")

def exists(x): return x is not None

# Note: DanActionModel class below is disabled - requires laq_model
# Using ControllableDINOLatentActionModel instead

class DanActionModel(nn.Module):
    """
    Quantize concatenated MASt3R features (B, 392, Dm) into 4 discrete codes, then reconstruct (B, 392, Dm).

    Set:
      code_seq_len = 4
      codebook_size = 16     # -> indices in [0..15]
      quant_dim small (e.g., 64 or 128)
      dim = model token dim (can set = Dm to avoid adapters)
    """

    def __init__(
        self,
        device: torch.device,
        *,
        dim=768,                 # model token dim
        quant_dim=64,           # codebook embedding dim (internal)
        codebook_size=16,       # 16 for [0..15]
        num_input_tokens=392,# concatenated length (e.g., 2*14*14)
        mast3r_dim=768,     # Dm; if None, inferred on first call
        code_seq_len=4,      # N = 4
        heads=8,
        dim_head=64,
        attn_dropout=0.0,
        ff_dropout=0.0,
        predict_delta=False, # here we reconstruct absolute features
        cosine_loss_weight=0.5,
        bottleneck_depth=2,  # depth for summarizer & decoder xattn blocks
    ):
        super().__init__()
        assert code_seq_len == 4, "Set code_seq_len=4 for 4 codes"
        self.model_dim = dim
        self.mast3r_dim = mast3r_dim
        self.num_input_tokens = num_input_tokens
        self.code_seq_len = code_seq_len
        self.predict_delta = predict_delta
        self.cosine_loss_weight = cosine_loss_weight

        # adapters (lazy if mast3r_dim not given)
        self.feat_in  = nn.Linear(mast3r_dim, dim) if mast3r_dim is not None else nn.Identity()
        self.feat_out = nn.Linear(dim, mast3r_dim) if mast3r_dim is not None else nn.Identity()
        self.norm_in  = nn.LayerNorm(mast3r_dim) if mast3r_dim is not None else nn.Identity()
        self.norm_tok = nn.LayerNorm(dim)

        # -------- summarizer: 4 learned queries attend over 392 input tokens --------
        # implement as a Transformer with cross-attn: queries learnable, context=input
        xattn_kwargs = dict(
            dim=dim, dim_head=dim_head, heads=heads,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            peg=False, peg_causal=False, has_cross_attn=True, dim_context=dim,
        )
        self.summarizer = Transformer(depth=bottleneck_depth, **xattn_kwargs)
        self.latent_queries = nn.Parameter(torch.randn(1, code_seq_len, dim))  # (1, 4, dim)

        # -------- VQ over the 4 latent tokens --------
        self.vq = NSVQ(
            dim=dim,
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            device=device,
            code_seq_len=code_seq_len
        )

        # -------- decoder: expand from 4 quantized tokens -> 392 tokens --------
        self.decoder = Transformer(depth=bottleneck_depth, **xattn_kwargs)
        self.output_queries = nn.Parameter(torch.randn(1, num_input_tokens, dim))  # (1, 392, dim)

        # -------- Mast3R model --------
        self.mast3r_model = Mast3rLatentExtractor()

    # --- your extractor: MUST return concatenated features (B, 392, Dm) OR tuple you concat here ---
    def extract_mast3r_latent(self, img1, img2):
        with torch.no_grad():
            return self.mast3r_model.extract_mast3r_latent(img1, img2)

    # --- forward ---
    def forward(
        self,
        img1, img2,
        step=0,
        return_recons_only=False,
        return_only_codebook_ids=False
    ):
        # 1) MASt3R concatenated features (B, 392, Dm) or tuple -> concat
        lat = self.extract_mast3r_latent(img1, img2)
        if isinstance(lat, tuple):
            f1, f2 = lat      # (B,S,Dm) each
            feats = torch.cat([f1, f2], dim=1)  # (B, 2S, Dm) == (B, 392, Dm)
        else:
            feats = lat       # (B, 392, Dm)

        B, S, Dm = feats.shape
        assert S == self.num_input_tokens, f"Expected {self.num_input_tokens} tokens, got {S}"

        # 2) map to model dim
        x = self.feat_in(self.norm_in(feats))     # (B, 392, dim)

        # 3) summarizer: 4 latent queries attend over x -> (B, 4, dim)
        q = self.latent_queries.expand(B, -1, -1) # (B, 4, dim)
        # Transformer expects (seq, d) or (batch*len, d) depending on impl; here we pass (B, T, D) with context
        latents = self.summarizer(q, context=x)   # (B, 4, dim)

        # 4) VQ: quantize the 4 tokens
        # NSVQ API in original takes (first, last). We pass the same latents twice to trigger quantization.
        quant_tokens, perplexity, codebook_usage, indices = self.vq(latents, use_delta=False)
        # shapes: quant_tokens -> (B, 4, dim), indices -> (B, 4)
        num_unique_indices = indices.unique().size(0)

        if return_only_codebook_ids:
            return indices  # each ∈ [0..codebook_size-1]; with codebook_size=16 → [0..15]

        # 5) decoder: expand from 4 codes -> 392 tokens (model dim)
        out_q = self.output_queries.expand(B, -1, -1)   # (B, 392, dim)
        decoded = self.decoder(out_q, context=quant_tokens)  # (B, 392, dim)

        decoded = self.norm_tok(decoded)
        recon = self.feat_out(decoded)  # (B, 392, Dm)

        if return_recons_only:
            return recon  # (B, 392, Dm)

        # 6) feature-space loss
        mse = F.mse_loss(recon, feats)

        if self.cosine_loss_weight > 0:
            cos = 1 - F.cosine_similarity(recon.flatten(1), feats.flatten(1), dim=-1).mean()
            loss = mse + self.cosine_loss_weight * cos
        else:
            loss = mse

        # optional: recycle unused codes (as in original)
        if ((step % 10 == 0 and step < 100) or (step % 100 == 0 and step < 1000) or (step % 500 == 0 and step < 5000)) and step != 0:
            self.vq.replace_unused_codebooks(quant_tokens.shape[0])

        return loss, num_unique_indices

    @torch.no_grad()
    def inference(
        self,
        img1, img2,
    ):
        outputs = {}

        lat = self.extract_mast3r_latent(img1, img2)
        if isinstance(lat, tuple):
            f1, f2 = lat
            feats = torch.cat([f1, f2], dim=1)
        else:
            feats = lat

        B, S, Dm = feats.shape

        x = self.feat_in(self.norm_in(feats))          # (B, 392, dim)
        q = self.latent_queries.expand(B, -1, -1)      # (B, 4, dim)
        latents = self.summarizer(q, context=x)        # (B, 4, dim)

        quant_tokens, indices = self.vq.inference(latents, latents)  # (B, 4, dim), (B, 4)

        outputs["z_q"] = quant_tokens
        outputs["indices"] = indices
        return outputs
    
    @property
    def device(self):
        return next(self.parameters()).device
