class EarlyStopper:
    """Stops training after 'patience' epochs with no metric improvement.

    Works in either direction via 'mode' ("max" for metrics like PR-AUC,
    "min" for a loss). Exposes 'state_dict'/'load_state_dict' so its counter
    survives a resume-from-interrupt alongside the model and optimizer.
    """

    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.epochs_without_improvement = 0

    def _is_improvement(self, metric: float) -> bool:
        if self.best is None:  # first update is always an improvement
            return True
        return metric > self.best if self.mode == "max" else metric < self.best

    def update(self, metric: float) -> None:
        if self._is_improvement(metric):
            self.best = metric
            self.epochs_without_improvement = 0  # reset the patience counter
        else:
            self.epochs_without_improvement += 1

    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def state_dict(self) -> dict:
        return {
            "best": self.best,
            "epochs_without_improvement": self.epochs_without_improvement
        }

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
