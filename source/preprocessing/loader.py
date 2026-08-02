from typing import TypeVar, Generic
from abc import ABC, abstractmethod

LoaderInput = TypeVar("LoaderInput")
LoaderOutput = TypeVar("LoaderOutput")


class Loader(ABC, Generic[LoaderInput, LoaderOutput]):
    @abstractmethod
    def load(self, input: LoaderInput) -> LoaderOutput:
        raise NotImplementedError("Must be implemented")