from typing import TypeVar, Generic
from abc import ABC, abstractmethod

TrainerInput = TypeVar("TrainerInput")
TrainerOutput = TypeVar("TrainerOutput")


class Trainer(ABC, Generic[TrainerInput, TrainerOutput]):
    @abstractmethod
    def train(self, input: TrainerInput) -> TrainerOutput:
        raise NotImplementedError("Must be implemented")
