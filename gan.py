"""
gan.py
======
Conditional DCGAN with Residual Connections for Glasses / No-Glasses dataset.

  Generator   : noise z + class label → 64×64 RGB image
  Discriminator: 64×64 RGB image + class label → real/fake score

Residual connections: x = x + layer(x) as instructed by TA.
Used in all intermediate layers via ResBlock modules.
For layers where channel sizes change, a 1x1 conv projection is used
to match dimensions before adding (standard ResNet practice).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation (DCGAN paper recommendation)
# ─────────────────────────────────────────────────────────────────────────────
def weights_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block for Generator (transposed conv, upsample)
# ─────────────────────────────────────────────────────────────────────────────
class ResBlockG(nn.Module):
    """
    Residual block for Generator.
    Upsamples by 2x using transposed conv, then adds a skip connection.
    Skip connection uses interpolation to match spatial dims + 1x1 conv
    to match channel dims — making it broadcastable as TA instructed.

    x_out = upsample_path(x) + skip(x)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block for Discriminator (strided conv, downsample)
# ─────────────────────────────────────────────────────────────────────────────
class ResBlockD(nn.Module):
    """
    Residual block for Discriminator.
    Downsamples by 2x using strided conv, then adds a skip connection.
    Skip uses average pooling + 1x1 conv to match dims.

    x_out = downsample_path(x) + skip(x)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Residual transposed-conv tower: (z + label embedding) → 64×64 RGB image.

    Args:
        z_dim      : noise vector size          (ablation: 64 / 128 / 256)
        num_classes: number of label classes    (2 for this dataset)
        embed_dim  : label embedding dimension  (ablation: 16 / 32 / 64)
        ngf        : base feature-map count     (ablation: 32 / 64)
    """

    def __init__(self, z_dim=128, num_classes=2, embed_dim=32, ngf=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        in_ch = z_dim + embed_dim

        # First layer: no residual (1x1 spatial, nothing to skip)
        self.first = nn.Sequential(
            nn.ConvTranspose2d(in_ch, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )
        # Residual upsample blocks
        self.res1 = ResBlockG(ngf * 8, ngf * 4)   # 4x4   → 8x8
        self.res2 = ResBlockG(ngf * 4, ngf * 2)   # 8x8   → 16x16
        self.res3 = ResBlockG(ngf * 2, ngf)        # 16x16 → 32x32

        # Final layer: no residual on output
        self.last = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x   = torch.cat([z, emb], dim=1)
        x   = x.unsqueeze(-1).unsqueeze(-1)
        x   = self.first(x)
        x   = self.res1(x)
        x   = self.res2(x)
        x   = self.res3(x)
        return self.last(x)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Residual strided-conv classifier: (image, label) → real/fake probability.

    Args:
        num_classes: number of label classes   (2 for this dataset)
        embed_dim  : label embedding dimension (ablation: 16 / 32 / 64)
        ndf        : base feature-map count    (ablation: 32 / 64)
        dropout    : dropout probability       (ablation: 0.0 / 0.3 / 0.5)
    """

    def __init__(self, num_classes=2, embed_dim=32, ndf=64, dropout=0.0):
        super().__init__()
        self.label_emb  = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Linear(embed_dim, 64 * 64)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # First layer: 4ch input (no residual)
        self.first = nn.Sequential(
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Residual downsample blocks
        self.res1 = ResBlockD(ndf,     ndf * 2)   # 32x32 → 16x16
        self.res2 = ResBlockD(ndf * 2, ndf * 4)   # 16x16 → 8x8
        self.res3 = ResBlockD(ndf * 4, ndf * 8)   # 8x8   → 4x4

        # Final classifier
        self.last = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        emb       = self.label_emb(labels)
        label_map = self.label_proj(emb).view(-1, 1, 64, 64)
        x         = torch.cat([images, label_map], 1)
        x = self.first(x)
        x = self.drop(x)
        x = self.res1(x)
        x = self.drop(x)
        x = self.res2(x)
        x = self.drop(x)
        x = self.res3(x)
        return self.last(x).view(-1, 1)