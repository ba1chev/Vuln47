import numpy as np

from source.preprocessing.data_preprocessing.data_loading.data_loader import DataLoader


class BaselineDataset:
    """Loads a PrimeVul split into (source strings, labels) for the bag-of-tokens baseline.

    Reuses the pipeline's own DataLoader so the baseline reads the exact same records the
    GNN does — only the representation differs downstream.
    """

    def __init__(self, loader: DataLoader | None = None):
        self.loader = loader or DataLoader()

    def load(self, path: str) -> tuple[list[str], np.ndarray]:
        texts, labels = [], []
        for rec in self.loader.load(path):
            texts.append(rec.get("func", ""))
            labels.append(int(rec.get("target", 0)))
        return texts, np.array(labels)
