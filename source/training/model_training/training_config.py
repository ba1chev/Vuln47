import os

import torch

# device: prefer Apple MPS, else CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# data splits produced by preprocessing
DATA_DIR = os.path.join("data", "processed")
TRAIN_PATH = os.path.join(DATA_DIR, "train.pt")
VALID_PATH = os.path.join(DATA_DIR, "valid.pt")
TEST_PATH = os.path.join(DATA_DIR, "test.pt")

# where the best checkpoint is written
CHECKPOINT_PATH = os.path.join("data", "model", "vuln_gnn.pt")

# optimization
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
BATCH_SIZE = 64

# metric used to track the best checkpoint
BEST_METRIC = "pr_auc"
