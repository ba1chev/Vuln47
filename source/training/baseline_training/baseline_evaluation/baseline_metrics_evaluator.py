import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from source.training.evaluator import Evaluator
from source.training.baseline_training.vuln47_lr import Vuln47LR
from source.training.model_training.model_evaluation.metrics_evaluator import best_f1_threshold


class BaselineMetricsEvaluator(Evaluator[tuple, dict[str, float]]):
    """Scores the LR baseline on the same imbalance-aware metrics as the GNN.

    Returns the identical {f1, precision, recall, pr_auc} dict as MetricsEvaluator so the
    two models drop into the same comparison table, and reuses the shared best-F1
    threshold calibration.
    """

    def __init__(self, model: Vuln47LR):
        self.model = model

    def _collect(self, input: tuple[csr_matrix, np.ndarray]):
        X, y = input
        return y, self.model.predict_scores(X)

    def evaluate(self, input: tuple[csr_matrix, np.ndarray], threshold: float = 0.5) -> dict[str, float]:
        y, score = self._collect(input)
        pred = (score >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        pr_auc = average_precision_score(y, score)
        return {"f1": f1, "precision": precision, "recall": recall, "pr_auc": pr_auc}

    def best_threshold(self, input: tuple[csr_matrix, np.ndarray]) -> float:
        y, score = self._collect(input)
        return best_f1_threshold(y, score)
