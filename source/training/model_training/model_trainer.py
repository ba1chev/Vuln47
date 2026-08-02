import torch
from torch import nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from source.training.trainer import Trainer


class ModelTrainer(Trainer[DataLoader, float]):
    def __init__(self, model, optimizer, class_weight: torch.Tensor, device: str):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

    def train(self, input: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for batch in tqdm(input, desc="train"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y.view(-1))
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(input)
