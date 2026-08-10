import os

# reproducibility (matches training_config.SEED)
SEED = 42

# raw PrimeVul splits — the baseline reads the same JSONL source of truth as the
# GNN pipeline, NOT the graph tensors under data/processed
TRAIN_PATH = os.path.join("data", "primevul_train.jsonl")
VALID_PATH = os.path.join("data", "primevul_valid.jsonl")
TEST_PATH = os.path.join("data", "primevul_test.jsonl")

# bag-of-tokens hashing: fixed feature space, no fitted vocabulary
N_FEATURES = 2 ** 18
USE_TFIDF = True

# logistic-regression hyperparameters
C = 1.0
MAX_ITER = 1000
CLASS_WEIGHT = "balanced"  # counter the ~1:32 imbalance without resampling

# decision-threshold calibration
DEFAULT_THRESHOLD = 0.5
