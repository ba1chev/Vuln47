import os
import torch

from source.vuln47_gnn_model import build_model
from source.training.training_pipeline import TrainingPipeline
from source.training.model_training.model_trainer import ModelTrainer
from source.training.model_training.batch_loading.graph_batch_loader import GraphBatchLoader
from source.training.model_training.model_evaluation.metrics_evaluator import MetricsEvaluator
from source.training.model_training.training_config import (
    DEVICE, TRAIN_PATH, VALID_PATH, TEST_PATH, CHECKPOINT_PATH,
    LR, WEIGHT_DECAY, EPOCHS, BEST_METRIC
)


class ModelTrainingPipeline(TrainingPipeline[None, dict]):
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.device = DEVICE
        self.train_batch_loader = GraphBatchLoader(shuffle=True)
        self.eval_batch_loader = GraphBatchLoader(shuffle=False)

    def run(self, input: None = None) -> dict:
        train_loader = self.train_batch_loader.load(TRAIN_PATH)
        valid_loader = self.eval_batch_loader.load(VALID_PATH)

        model = build_model(self.vocab).to(self.device)
        class_weight = self._compute_class_weight(train_loader)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        trainer = ModelTrainer(model, optimizer, class_weight, self.device)
        evaluator = MetricsEvaluator(model, self.device)

        best = -1.0
        best_state = None
        best_metrics, best_epoch = None, -1
        for epoch in range(EPOCHS):
            loss = trainer.train(train_loader)
            metrics = evaluator.evaluate(valid_loader)
            print(f"epoch {epoch}: loss={loss:.4f} "
                  + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if metrics[BEST_METRIC] > best:
                best = metrics[BEST_METRIC]
                best_metrics, best_epoch = metrics, epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save(best_state, CHECKPOINT_PATH)

        model.load_state_dict(best_state)
        test_metrics = evaluator.evaluate(self.eval_batch_loader.load(TEST_PATH))
        return {
            "best_epoch": best_epoch,
            "best_valid": best_metrics,
            "test": test_metrics,
            "checkpoint": CHECKPOINT_PATH,
        }

    def _compute_class_weight(self, train_loader) -> torch.Tensor:
        n_vuln = n_safe = 0
        for batch in train_loader:
            y = batch.y.view(-1)
            n_vuln += int((y == 1).sum())
            n_safe += int((y == 0).sum())
        ratio = n_safe / max(n_vuln, 1)
        return torch.tensor([1.0, ratio], dtype=torch.float)