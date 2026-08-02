import torch
from torch_geometric.loader import DataLoader

from source.training.batch_loader import BatchLoader
from source.training.model_training.training_config import BATCH_SIZE


class GraphBatchLoader(BatchLoader[str, DataLoader]):
    def __init__(self, batch_size: int = BATCH_SIZE, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load(self, input: str) -> DataLoader:
        graphs = torch.load(input, weights_only=False)
        return DataLoader(graphs, batch_size=self.batch_size, shuffle=self.shuffle)
