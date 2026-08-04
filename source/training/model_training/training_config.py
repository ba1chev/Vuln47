import os
import torch

# device: prefer CUDA (e.g. Colab GPU), then Apple MPS, else CPU
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# data splits produced by preprocessing
DATA_DIR = os.path.join("data", "processed")
TRAIN_PATH = os.path.join(DATA_DIR, "train.pt")
VALID_PATH = os.path.join(DATA_DIR, "valid.pt")
TEST_PATH = os.path.join(DATA_DIR, "test.pt")

# where the best checkpoint is written
CHECKPOINT_PATH = os.path.join("data", "model", "vuln_gnn.pt")

# reproducibility
SEED = 42

# optimization
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
BATCH_SIZE = 64

# metric used to track the best checkpoint
BEST_METRIC = "pr_auc"

# model capacity
HIDDEN_DIM = 192
TOKEN_EMB_DIM = 32

# loss: "weighted_ce" (class-weighted cross-entropy) or "focal"
LOSS_TYPE = "focal"
FOCAL_GAMMA = 2.0

# decision-threshold calibration: after training, pick the threshold that maximizes
# F1 on the valid split and store it in the checkpoint (used at eval time)
DEFAULT_THRESHOLD = 0.5

# learning-rate schedule: halve the LR when the tracked metric plateaus
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2
MIN_LR = 1e-6

# early stopping: stop after this many epochs without an improvement
EARLY_STOPPING_PATIENCE = 5
