import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_WORKERS
from dataset import FacesDataset


dataset = FacesDataset(augment=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

print(f"Total samples: {len(dataset)}")

batch_imgs, batch_labels = next(iter(loader))

print(f"Batch image shape : {batch_imgs.shape}")
print(f"Batch label shape : {batch_labels.shape}")
print(f"Image dtype       : {batch_imgs.dtype}")
print(f"Image min/max     : {batch_imgs.min():.3f} / {batch_imgs.max():.3f}")
print(f"Unique labels     : {batch_labels.unique().tolist()}")
print("Pipeline works :)")