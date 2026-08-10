import torch
from torch_geometric.loader import DataLoader

from source.training.batch_loader import BatchLoader
from source.training.model_training.training_config import BATCH_SIZE


class GraphBatchLoader(BatchLoader[str, DataLoader]):
    """Loads a saved '.pt' split and wraps it in a PyG 'DataLoader'.

    PyG's 'DataLoader' merges the graphs of a mini-batch into one big
    disconnected graph (plus a 'batch' vector), which is what the model pools
    over. Train uses 'shuffle=True' eval splits keep insertion order.
    """

    def __init__(self, batch_size: int = BATCH_SIZE, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load(self, input: str) -> DataLoader:
        # weights_only=False: the file stores pickled PyG Data objects, not bare tensors
        graphs = torch.load(input, weights_only=False)
        return DataLoader(graphs, batch_size=self.batch_size, shuffle=self.shuffle)
