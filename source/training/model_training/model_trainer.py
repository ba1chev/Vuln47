import torch
from torch import nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from source.training.trainer import Trainer
from source.training.model_training.loss.focal_loss import FocalLoss
from source.training.model_training.training_config import LOSS_TYPE, FOCAL_GAMMA


class ModelTrainer(Trainer[DataLoader, float]):
    """Runs one training epoch with an imbalance-aware loss.

    Builds either focal loss or class-weighted cross-entropy (per
    'training_config.LOSS_TYPE') from the given class weight, then trains a
    single pass over the batches and returns the mean loss for the epoch.
    """

    def __init__(self, model, optimizer, class_weight: torch.Tensor, device: str):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        weight = class_weight.to(device)
        # focal loss adds an easy-example down-weighting term on top of the class weight
        if LOSS_TYPE == "focal":
            self.criterion = FocalLoss(alpha=weight, gamma=FOCAL_GAMMA).to(device)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=weight)

    def train(self, input: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for batch in tqdm(input, desc="train"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y.view(-1))  # y is [num_graphs, 1] -> flatten
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(input)
