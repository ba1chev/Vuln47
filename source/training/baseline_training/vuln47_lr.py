import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from source.training.trainer import Trainer
from source.training.baseline_training.baseline_config import C, MAX_ITER, CLASS_WEIGHT, SEED


class Vuln47LR(Trainer[tuple, LogisticRegression]):
    """A class-weighted logistic-regression baseline over bag-of-tokens features.

    The structure-blind control for the GNN: same data, same metrics, but no graph.
    """

    def __init__(self, class_weight=CLASS_WEIGHT,
        C: float = C, max_iter: int = MAX_ITER, seed: int = SEED):
        self.model = LogisticRegression(
            class_weight=class_weight,
            C=C,
            max_iter=max_iter,
            random_state=seed
        )

    def train(self, input: tuple[csr_matrix, np.ndarray]) -> LogisticRegression:
        X, y = input
        self.model.fit(X, y)
        return self.model

    def predict_scores(self, X: csr_matrix) -> np.ndarray:
        """P(vulnerable) per row — mirrors the GNN's softmax score for the positive class."""
        return self.model.predict_proba(X)[:, 1]
