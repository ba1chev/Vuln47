import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from source.training.evaluator import Evaluator


class MetricsEvaluator(Evaluator[DataLoader, dict[str, float]]):
    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, input: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_labels, all_preds, all_scores = [], [], []
        for batch in input:
            batch = batch.to(self.device)
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_labels.append(batch.y.view(-1).cpu())
            all_preds.append(preds.cpu())
            all_scores.append(probs.cpu())

        y = torch.cat(all_labels).numpy()
        pred = torch.cat(all_preds).numpy()
        score = torch.cat(all_scores).numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        pr_auc = average_precision_score(y, score)
        return {"f1": f1, "precision": precision, "recall": recall, "pr_auc": pr_auc}
