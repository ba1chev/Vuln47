class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.epochs_without_improvement = 0

    def _is_improvement(self, metric: float) -> bool:
        if self.best is None:
            return True
        return metric > self.best if self.mode == "max" else metric < self.best

    def update(self, metric: float) -> None:
        if self._is_improvement(metric):
            self.best = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience
