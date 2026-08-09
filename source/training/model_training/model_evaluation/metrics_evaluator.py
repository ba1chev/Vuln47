import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, precision_recall_curve

from source.training.evaluator import Evaluator


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    p, r = precision[:-1], recall[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


class MetricsEvaluator(Evaluator[DataLoader, dict[str, float]]):
    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    @torch.no_grad()
    def _collect(self, input: DataLoader):
        self.model.eval()
        all_labels, all_scores = [], []
        for batch in input:
            batch = batch.to(self.device)
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.append(batch.y.view(-1).cpu())
            all_scores.append(probs.cpu())
        y = torch.cat(all_labels).numpy()
        score = torch.cat(all_scores).numpy()
        return y, score

    def evaluate(self, input: DataLoader, threshold: float = 0.5) -> dict[str, float]:
        y, score = self._collect(input)
        pred = (score >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        pr_auc = average_precision_score(y, score)
        return {"f1": f1, "precision": precision, "recall": recall, "pr_auc": pr_auc}

    def best_threshold(self, input: DataLoader) -> float:
        y, score = self._collect(input)
        return best_f1_threshold(y, score)
