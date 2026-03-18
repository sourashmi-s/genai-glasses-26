import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE_COL  = "id"
LABEL_COL = "glasses"

IMG_DIR     = "data/faces-spring-2020/faces-spring-2020/"
RESIZED_DIR = "data/resized/"
TRAIN_CSV   = "data/train.csv"
TEST_CSV    = "data/test.csv"
MODEL_DIR   = "models/"
OUTPUT_DIR  = "outputs/"

IMG_SIZE    = 64
BATCH_SIZE  = 64
NUM_EPOCHS  = 50
LR          = 1e-3
NUM_WORKERS = 0
NUM_CLASSES = 2