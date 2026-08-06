import os
import torch

from source.vuln47_gnn_model import build_model
from source.training.training_pipeline import TrainingPipeline
from source.training.model_training.model_trainer import ModelTrainer
from source.training.model_training.early_stopping.early_stopper import EarlyStopper
from source.training.model_training.batch_loading.graph_batch_loader import GraphBatchLoader
from source.training.model_training.model_evaluation.metrics_evaluator import MetricsEvaluator
from source.training.model_training.training_config import (
    DEVICE, TRAIN_PATH, VALID_PATH, TEST_PATH, CHECKPOINT_PATH, RESUME_PATH,
    LR, WEIGHT_DECAY, EPOCHS, BEST_METRIC, SEED,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, MIN_LR, EARLY_STOPPING_PATIENCE,
    HIDDEN_DIM, TOKEN_EMB_DIM
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

        model = build_model(
            self.vocab, hidden_dim=HIDDEN_DIM, token_emb_dim=TOKEN_EMB_DIM
        ).to(self.device)
        class_weight = self._compute_class_weight(train_loader)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE, min_lr=MIN_LR
        )
        early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE, mode="max")

        trainer = ModelTrainer(model, optimizer, class_weight, self.device)
        evaluator = MetricsEvaluator(model, self.device)

        best = -1.0
        best_state = None
        best_metrics, best_epoch = None, -1
        start_epoch = 0

        if os.path.exists(RESUME_PATH):
            ckpt = torch.load(RESUME_PATH, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            early_stopper.load_state_dict(ckpt["early_stopper"])
            torch.set_rng_state(ckpt["rng_state"].cpu().to(torch.uint8))
            best = ckpt["best"]
            best_state = ckpt["best_state"]
            best_metrics, best_epoch = ckpt["best_metrics"], ckpt["best_epoch"]
            start_epoch = ckpt["epoch"] + 1
            print(f"resuming from epoch {start_epoch} "
                f"(best {BEST_METRIC}={best:.4f} @ epoch {best_epoch})")
        else:
            torch.manual_seed(SEED)

        for epoch in range(start_epoch, EPOCHS):
            loss = trainer.train(train_loader)
            metrics = evaluator.evaluate(valid_loader)
            tracked = metrics[BEST_METRIC]
            scheduler.step(tracked)
            early_stopper.update(tracked)
            print(f"epoch {epoch}: loss={loss:.4f} "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                + f" lr={optimizer.param_groups[0]['lr']:.2e}")
            if tracked > best:
                best = tracked
                best_metrics, best_epoch = metrics, epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self._save_resume(
                epoch, model, optimizer, scheduler, early_stopper,
                best, best_state, best_metrics, best_epoch
            )
            if early_stopper.should_stop():
                print(f"early stopping at epoch {epoch} "
                    f"(no {BEST_METRIC} improvement in {EARLY_STOPPING_PATIENCE} epochs)")
                break

        model.load_state_dict(best_state)
        threshold = evaluator.best_threshold(valid_loader)
        test_metrics = evaluator.evaluate(
            self.eval_batch_loader.load(TEST_PATH), threshold=threshold
        )
        valid_at_threshold = evaluator.evaluate(valid_loader, threshold=threshold)

        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save({
            "state_dict": best_state,
            "threshold": threshold,
            "config": {
                "hidden_dim": HIDDEN_DIM,
                "token_emb_dim": TOKEN_EMB_DIM
            },
        }, CHECKPOINT_PATH)

        if os.path.exists(RESUME_PATH):
            os.remove(RESUME_PATH)

        return {
            "best_epoch": best_epoch,
            "best_valid": best_metrics,
            "threshold": threshold,
            "valid_at_threshold": valid_at_threshold,
            "test": test_metrics,
            "checkpoint": CHECKPOINT_PATH
        }

    def _save_resume(self, epoch, model, optimizer, scheduler, early_stopper,
        best, best_state, best_metrics, best_epoch) -> None:
        os.makedirs(os.path.dirname(RESUME_PATH), exist_ok=True)
        tmp = RESUME_PATH + ".tmp"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stopper": early_stopper.state_dict(),
            "rng_state": torch.get_rng_state(),
            "best": best,
            "best_state": best_state,
            "best_metrics": best_metrics,
            "best_epoch": best_epoch
        }, tmp)
        os.replace(tmp, RESUME_PATH)

    def _compute_class_weight(self, train_loader) -> torch.Tensor:
        n_vuln = n_safe = 0
        for batch in train_loader:
            y = batch.y.view(-1)
            n_vuln += int((y == 1).sum())
            n_safe += int((y == 0).sum())
        ratio = n_safe / max(n_vuln, 1)
        return torch.tensor([1.0, ratio], dtype=torch.float)