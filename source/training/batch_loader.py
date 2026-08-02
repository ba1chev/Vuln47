from typing import TypeVar, Generic
from abc import ABC, abstractmethod

BatchLoaderInput = TypeVar("BatchLoaderInput")
BatchLoaderOutput = TypeVar("BatchLoaderOutput")


class BatchLoader(ABC, Generic[BatchLoaderInput, BatchLoaderOutput]):
    @abstractmethod
    def load(self, input: BatchLoaderInput) -> BatchLoaderOutput:
        raise NotImplementedError("Must be implemented")
