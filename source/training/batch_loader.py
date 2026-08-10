from typing import TypeVar, Generic
from abc import ABC, abstractmethod

BatchLoaderInput = TypeVar("BatchLoaderInput")
BatchLoaderOutput = TypeVar("BatchLoaderOutput")


class BatchLoader(ABC, Generic[BatchLoaderInput, BatchLoaderOutput]):
    """Abstract base for turning a stored split into an iterable of mini-batches.

    Concrete loaders decide how the raw split (e.g. a '.pt' path) is read and
    grouped; 'load' returns a batch iterator the trainer/evaluator consume.
    """

    @abstractmethod
    def load(self, input: BatchLoaderInput) -> BatchLoaderOutput:
        raise NotImplementedError("Must be implemented")
