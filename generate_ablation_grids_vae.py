import os
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import save_image
from vae import VAE
from config import DEVICE, OUTPUT_DIR, MODEL_DIR

RUNS = {
    "baseline":   dict(latent_dim=256, filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "latent_64":  dict(latent_dim=64,  filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "latent_512": dict(latent_dim=512, filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "filter_5":   dict(latent_dim=256, filter_size=5, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "filter_7":   dict(latent_dim=256, filter_size=7, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "deconv":     dict(latent_dim=256, filter_size=3, num_layers=3, activation="elu",        decoder_type="deconv",        num_res_blocks=2),
    "act_relu":   dict(latent_dim=256, filter_size=3, num_layers=3, activation="relu",       decoder_type="interpolation", num_res_blocks=2),
    "act_leaky":  dict(latent_dim=256, filter_size=3, num_layers=3, activation="leaky_relu", decoder_type="interpolation", num_res_blocks=2),
    "layers_2":   dict(latent_dim=256, filter_size=3, num_layers=2, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "layers_4":   dict(latent_dim=256, filter_size=3, num_layers=4, activation="elu",        decoder_type="interpolation", num_res_blocks=2),
    "no_res":     dict(latent_dim=256, filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=0),
    "res_1":      dict(latent_dim=256, filter_size=3, num_layers=3, activation="elu",        decoder_type="interpolation", num_res_blocks=1),
}

grids_dir = os.path.join(OUTPUT_DIR, "ablations", "grids")
os.makedirs(grids_dir, exist_ok=True)

for run_name, cfg in RUNS.items():
    ckpt = os.path.join(MODEL_DIR, "ablations", run_name, "best.pth")
    if not os.path.exists(ckpt):
        print(f"[SKIP] {run_name}")
        continue

    print(f"Generating grid for {run_name}...")
    model = VAE(num_classes=2, **cfg).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        glasses    = (model.generate(label=1, n=3, device=DEVICE) * 0.5 + 0.5).clamp(0, 1)
        no_glasses = (model.generate(label=0, n=3, device=DEVICE) * 0.5 + 0.5).clamp(0, 1)

    img_paths = []
    for i, img in enumerate(glasses):
        path = os.path.join(grids_dir, f"{run_name}_glasses_{i+1}.png")
        save_image(img, path)
        img_paths.append(path)
    for i, img in enumerate(no_glasses):
        path = os.path.join(grids_dir, f"{run_name}_noglasses_{i+1}.png")
        save_image(img, path)
        img_paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle(run_name, fontsize=14)
    labels = [f"Glasses {i+1}" for i in range(3)] + [f"No Glasses {i+1}" for i in range(3)]
    for ax, path, label in zip(axes.flatten(), img_paths, labels):
        ax.imshow(mpimg.imread(path))
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(grids_dir, f"{run_name}_grid.png"), dpi=150)
    plt.close()
    print(f"  Saved {run_name}_grid.png")

print(f"\nAll grids saved to {grids_dir}")