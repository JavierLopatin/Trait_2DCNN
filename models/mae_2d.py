"""2D Masked Autoencoder (MAE) for spectral image pretraining.

Based on He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners",
adapted for single-channel spectral images (reshape transform).

Architecture matches Cherif et al. (2025) MAE-1D hyperparameters where applicable:
- Mask ratio: 75%
- Loss: Cosine similarity + MSE
- Encoder: ViT-Tiny (patch_size=16, embed_dim=192, depth=6, heads=3) ~5.4M params
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class MAE2D(nn.Module):
    """Masked Autoencoder with ViT encoder and lightweight decoder.

    During pretraining:
      - Split image into patches
      - Randomly mask 75% of patches
      - Encode visible patches with ViT encoder
      - Decode all patches (visible + mask tokens) with small decoder
      - Reconstruct pixel values of masked patches

    After pretraining:
      - Use encoder only for downstream tasks (+ regression head)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        # Encoder (ViT-Tiny-like)
        encoder_embed_dim: int = 192,
        encoder_depth: int = 6,
        encoder_num_heads: int = 3,
        # Decoder (lightweight)
        decoder_embed_dim: int = 96,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
        # MAE
        mask_ratio: float = 0.75,
        # Loss
        cosine_weight: float = 1.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.cosine_weight = cosine_weight
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(encoder_embed_dim, encoder_num_heads)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_chans)

        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings with sin-cos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(dec_pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x):
        """Random masking: keep (1-mask_ratio) of patches.

        Args:
            x: (B, N, D) patch embeddings

        Returns:
            x_masked: (B, N_visible, D)
            mask: (B, N) binary, 1=masked
            ids_restore: (B, N) indices to unshuffle
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def encode(self, x, mask=True):
        """Encode image patches.

        Args:
            x: (B, C, H, W) input image
            mask: if True, apply random masking (pretraining).
                  if False, encode all patches (inference).

        Returns:
            latent: encoder output
            mask: binary mask (if masking applied)
            ids_restore: restoration indices (if masking applied)
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if mask:
            x, mask_out, ids_restore = self.random_masking(x)
        else:
            mask_out = None
            ids_restore = None

        # Prepend CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        return x, mask_out, ids_restore

    def decode(self, x, ids_restore):
        """Decode: insert mask tokens and reconstruct patches."""
        x = self.decoder_embed(x)

        # Insert mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Predict pixel values (skip CLS)
        x = self.decoder_pred(x[:, 1:, :])
        return x

    def patchify(self, imgs):
        """Convert images to patch sequences for loss computation.

        (B, C, H, W) -> (B, N, patch_size**2 * C)
        """
        p = self.patch_size
        c = self.in_chans
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(imgs.shape[0], h * w, p * p * c)
        return x

    def forward(self, imgs):
        """Forward pass for pretraining.

        Returns:
            loss: reconstruction loss (cosine similarity + MSE) on masked patches
            pred: reconstructed patches
            mask: binary mask
        """
        latent, mask, ids_restore = self.encode(imgs, mask=True)
        pred = self.decode(latent, ids_restore)
        target = self.patchify(imgs)

        # Loss only on masked patches
        loss = self.reconstruction_loss(pred, target, mask)
        return loss, pred, mask

    def reconstruction_loss(self, pred, target, mask):
        """Combined MSE + Cosine similarity loss (Cherif 2025 style)."""
        # MSE loss on masked patches
        mse = (pred - target) ** 2
        mse = mse.mean(dim=-1)  # per-patch MSE
        mse = (mse * mask).sum() / mask.sum()

        if self.cosine_weight > 0:
            # Cosine similarity on masked patches
            mask_idx = mask.bool()
            pred_masked = pred[mask_idx]
            target_masked = target[mask_idx]
            cos = 1 - F.cosine_similarity(pred_masked, target_masked, dim=-1).mean()
            loss = mse + self.cosine_weight * cos
        else:
            loss = mse

        return loss

    def get_encoder(self):
        """Extract encoder for downstream fine-tuning."""
        return MAEEncoder(self)


class MAEEncoder(nn.Module):
    """Standalone encoder extracted from pretrained MAE for fine-tuning."""

    def __init__(self, mae: MAE2D):
        super().__init__()
        self.patch_embed = mae.patch_embed
        self.cls_token = mae.cls_token
        self.pos_embed = mae.pos_embed
        self.blocks = mae.encoder_blocks
        self.norm = mae.encoder_norm
        self.embed_dim = mae.pos_embed.shape[-1]

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x[:, 0]  # CLS token as global representation


class MAERegressionModel(nn.Module):
    """MAE encoder + regression head for trait prediction."""

    def __init__(self, encoder: MAEEncoder, n_traits: int = 8,
                 freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.embed_dim, encoder.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(encoder.embed_dim, n_traits),
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sin-cos positional embeddings."""
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)

    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)

    if cls_token:
        emb = np.concatenate([np.zeros((1, embed_dim)), emb], axis=0)
    return emb


def get_1d_sincos_pos_embed(embed_dim, pos):
    """Generate 1D sin-cos positional embeddings."""
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)
