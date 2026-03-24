import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from vae import VAE
from config import DEVICE
import os

os.makedirs("outputs/final", exist_ok=True)

model = VAE(
    latent_dim     = 256,
    num_classes    = 2,
    filter_size    = 3,
    num_layers     = 3,
    activation     = "elu",
    decoder_type   = "interpolation",
    num_res_blocks = 2
).to(DEVICE)

model.load_state_dict(torch.load("models/vae_best.pth", map_location=DEVICE))
model.eval()

with torch.no_grad():
   glasses    = (model.generate(label=1, n=3, device=DEVICE) * 0.5 + 0.5).clamp(0, 1)
   no_glasses = (model.generate(label=0, n=3, device=DEVICE) * 0.5 + 0.5).clamp(0, 1)

for i, img in enumerate(glasses):
    save_image(img, f"outputs/final/glasses_{i+1}.png")

for i, img in enumerate(no_glasses):
    save_image(img, f"outputs/final/no_glasses_{i+1}.png")

fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for i, ax in enumerate(axes[0]):
    import matplotlib.image as mpimg
    ax.imshow(mpimg.imread(f"outputs/final/glasses_{i+1}.png"))
    ax.set_title(f"Glasses {i+1}")
    ax.axis("off")

for i, ax in enumerate(axes[1]):
    import matplotlib.image as mpimg
    ax.imshow(mpimg.imread(f"outputs/final/no_glasses_{i+1}.png"))
    ax.set_title(f"No Glasses {i+1}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/final/final_grid.png", dpi=150)
plt.show()
print("Done. Saved to outputs/final/")