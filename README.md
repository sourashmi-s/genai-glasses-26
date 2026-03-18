# genai-glasses-26
Generative AI models (VAE, GAN, Diffusion) on Glasses or No Glasses dataset.
---

## Setup

1. Clone the repo
```bash
git clone https://github.com/cfcmadlad/genai-glasses-26.git
cd genai-glasses-26
```

2. Create and activate environment
```bash
conda create -n genai python=3.10
conda activate genai
```

3. Install dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Data

Dataset: https://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses

Get from Google Drive (shared separately):
- `resized64.zip` → unzip and place as `data/resized/`
- `train_corrected.csv` → rename to `train.csv`, place at `data/train.csv`

Also place the original `test.csv` at `data/test.csv` (from Kaggle dataset download).

---

## Update Paths

Open `config.py` and update these to match your machine:

```python
IMG_DIR    = "data/faces-spring-2020/faces-spring-2020/"
RESIZED_DIR = "data/resized/"
TRAIN_CSV  = "data/train.csv"
TEST_CSV   = "data/test.csv"
```

---

## Verify Setup

```bash
python skeleton_test.py
```

Expected output:
```
Total samples: 5000
Batch image shape : torch.Size([64, 3, 64, 64])
Image min/max     : -1.000 / 1.000
Unique labels     : [-1, 0, 1]
Pipeline works :)
```

---

## Files

| File | What it does |
|------|-------------|
| `config.py` | All constants and paths |
| `dataset.py` | PyTorch dataset class with augmentations |
| `preprocess.py` | Resizes all 5000 images to 64x64 |
| `visualize_labels.py` | Shows images with labels overlaid |
| `aug_verification.py` | Verifies augmentations visually |
| `skeleton_test.py` | Verifies full pipeline end to end |