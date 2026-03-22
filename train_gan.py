"""
train_gan.py
============
Trains the Conditional DCGAN defined in gan.py.

Baseline run:
    python train_gan.py

Ablation examples (change ONE hyperparameter at a time):
    python train_gan.py --run_name abl_z64      --z_dim 64
    python train_gan.py --run_name abl_z256     --z_dim 256
    python train_gan.py --run_name abl_d2       --d_steps 2
    python train_gan.py --run_name abl_bs128    --batch_size 128
    python train_gan.py --run_name abl_drop03   --dropout 0.3
    python train_gan.py --run_name abl_smooth   --label_smooth 0.1
    python train_gan.py --run_name abl_ngf32    --ngf 32 --ndf 32
    python train_gan.py --run_name abl_lr_low   --lr_g 1e-4 --lr_d 1e-4

Generated images are saved to outputs/gan/<run_name>/
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, DEVICE, MODEL_DIR, NUM_EPOCHS,
                    NUM_WORKERS, OUTPUT_DIR)
from dataset import FacesDataset
from gan import Discriminator, Generator, weights_init


# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Train Conditional DCGAN")

    # Run identity
    p.add_argument("--run_name",     default="baseline",
                   help="Name for this run – used in output paths")

    # Model hyperparameters
    p.add_argument("--z_dim",        type=int,   default=128,
                   help="Noise vector dimension (ablation: 64 / 128 / 256)")
    p.add_argument("--ngf",          type=int,   default=64,
                   help="Generator base feature maps (ablation: 32 / 64)")
    p.add_argument("--ndf",          type=int,   default=64,
                   help="Discriminator base feature maps (ablation: 32 / 64)")
    p.add_argument("--embed_dim",    type=int,   default=32,
                   help="Class embedding dimension")
    p.add_argument("--dropout",      type=float, default=0.0,
                   help="Discriminator dropout (ablation: 0.0 / 0.3 / 0.5)")

    # Training hyperparameters
    p.add_argument("--epochs",       type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--lr_g",         type=float, default=2e-4,
                   help="Generator learning rate")
    p.add_argument("--lr_d",         type=float, default=2e-4,
                   help="Discriminator learning rate")
    p.add_argument("--beta1",        type=float, default=0.5,
                   help="Adam beta1 – GAN heuristic keeps this at 0.5")
    p.add_argument("--d_steps",      type=int,   default=1,
                   help="Discriminator updates per generator update (ablation: 1 / 2 / 3)")
    p.add_argument("--label_smooth", type=float, default=0.0,
                   help="One-sided label smoothing for real targets (ablation: 0.0 / 0.1)")

    p.add_argument("--save_every",   type=int,   default=10,
                   help="Save checkpoint every N epochs")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_dirs(args):
    img_dir  = os.path.join(OUTPUT_DIR, "gan", args.run_name, "images")
    ckpt_dir = os.path.join(MODEL_DIR,  "gan", args.run_name)
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return img_dir, ckpt_dir


def save_samples(G, fixed_z, fixed_labels, img_dir, epoch, device):
    """Generate 6 images (3 glasses + 3 no-glasses) and save as a grid."""
    G.eval()
    with torch.no_grad():
        fake = G(fixed_z, fixed_labels)         # (6, 3, 64, 64) in [-1,1]
    # Denormalise → [0, 1] for saving
    fake = (fake + 1) / 2
    path = os.path.join(img_dir, f"epoch_{epoch:03d}.png")
    vutils.save_image(fake, path, nrow=3, padding=2)
    G.train()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = DEVICE
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Run name    : {args.run_name}")
    print(f"[INFO] z_dim={args.z_dim}  ngf={args.ngf}  ndf={args.ndf}  "
          f"d_steps={args.d_steps}  dropout={args.dropout}  "
          f"label_smooth={args.label_smooth}  batch={args.batch_size}")

    img_dir, ckpt_dir = make_dirs(args)

    # ── Data ────────────────────────────────────────────────────────────────
    # Use only labelled images (label != -1) for GAN training
    full_dataset = FacesDataset(augment=True)
    labelled_idx = [i for i, (_, lbl) in enumerate(full_dataset)
                    if lbl.item() != -1]

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, labelled_idx)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,          # keeps batch sizes uniform
    )
    print(f"[INFO] Labelled samples : {len(train_dataset)} | "
          f"Batches/epoch : {len(loader)}")

    # ── Models ──────────────────────────────────────────────────────────────
    G = Generator(z_dim=args.z_dim, num_classes=2,
                  embed_dim=args.embed_dim, ngf=args.ngf).to(device)
    D = Discriminator(num_classes=2, embed_dim=args.embed_dim,
                      ndf=args.ndf, dropout=args.dropout).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # ── Loss & Optimisers ───────────────────────────────────────────────────
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=args.lr_g,
                       betas=(args.beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr_d,
                       betas=(args.beta1, 0.999))

    # ── Fixed noise for consistent sample images each epoch ─────────────────
    # 3 glasses (label=1) + 3 no-glasses (label=0)
    fixed_z      = torch.randn(6, args.z_dim, device=device)
    fixed_labels = torch.tensor([1, 1, 1, 0, 0, 0],
                                dtype=torch.long, device=device)

    # ── Loss log ────────────────────────────────────────────────────────────
    log_path = os.path.join(OUTPUT_DIR, "gan", args.run_name, "loss_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,loss_D,loss_G\n")

    # ── Main loop ───────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0

        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            B         = real_imgs.size(0)

            # Label tensors with optional smoothing on real targets
            real_tgt = torch.full((B, 1), 1.0 - args.label_smooth,
                                  device=device)
            fake_tgt = torch.zeros(B, 1, device=device)

            # ── Train Discriminator (d_steps times) ──────────────────────
            for _ in range(args.d_steps):
                D.zero_grad()

                # Real pass
                loss_real = criterion(D(real_imgs, labels), real_tgt)

                # Fake pass
                z         = torch.randn(B, args.z_dim, device=device)
                fake_imgs = G(z, labels).detach()   # detach: don't train G here
                loss_fake = criterion(D(fake_imgs, labels), fake_tgt)

                loss_D = loss_real + loss_fake
                loss_D.backward()
                opt_D.step()

            # ── Train Generator ───────────────────────────────────────────
            G.zero_grad()

            z         = torch.randn(B, args.z_dim, device=device)
            fake_imgs = G(z, labels)
            # G wants D to think fakes are real
            loss_G = criterion(D(fake_imgs, labels), real_tgt)
            loss_G.backward()
            opt_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()

        # ── End-of-epoch ─────────────────────────────────────────────────
        avg_D = epoch_loss_D / len(loader)
        avg_G = epoch_loss_G / len(loader)
        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"Loss_D: {avg_D:.4f}  Loss_G: {avg_G:.4f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_D:.4f},{avg_G:.4f}\n")

        # Save sample images every epoch
        img_path = save_samples(G, fixed_z, fixed_labels,
                                img_dir, epoch, device)
        if epoch == 1 or epoch % 10 == 0:
            print(f"  → Sample saved : {img_path}")

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "opt_G":   opt_G.state_dict(),
                "opt_D":   opt_D.state_dict(),
                "args":    vars(args),
            }
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:03d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  → Checkpoint  : {ckpt_path}")

    print("\n[DONE] Training complete.")
    print(f"       Images     → {img_dir}")
    print(f"       Checkpoints→ {ckpt_dir}")
    print(f"       Loss log   → {log_path}")

    # ── Final submission samples ─────────────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "gan", args.run_name,
                              "final_6_samples.png")
    G.eval()
    with torch.no_grad():
        fake = G(fixed_z, fixed_labels)
    vutils.save_image((fake + 1) / 2, final_path, nrow=3, padding=4)
    print(f"\n[SUBMISSION] 6 final samples (3 glasses | 3 no-glasses):")
    print(f"             {final_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()
    train(args)