from typing import TypeVar, Generic
from abc import ABC, abstractmethod

TrainerInput = TypeVar("TrainerInput")
TrainerOutput = TypeVar("TrainerOutput")


class Trainer(ABC, Generic[TrainerInput, TrainerOutput]):
    """Abstract base for a single training pass over batched data.

    A trainer owns the model, optimizer and loss, and its 'train' method runs
    one epoch over the given batches, returning a summary of that pass (e.g. the
    mean loss).
    """

    @abstractmethod
    def train(self, input: TrainerInput) -> TrainerOutput:
        raise NotImplementedError("Must be implemented")
