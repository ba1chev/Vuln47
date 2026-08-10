from typing import TypeVar, Generic
from abc import ABC, abstractmethod

LoaderInput = TypeVar("LoaderInput")
LoaderOutput = TypeVar("LoaderOutput")


class Loader(ABC, Generic[LoaderInput, LoaderOutput]):
    """Abstract base for anything that reads a raw source into in-memory records.

    Generic over the input handle (e.g. a file path) and the produced output
    (e.g. an iterator of records), so concrete loaders can specialize both ends
    while sharing this single-method contract.
    """

    @abstractmethod
    def load(self, input: LoaderInput) -> LoaderOutput:
        raise NotImplementedError("Must be implemented")