from typing import TypeVar, Generic
from abc import ABC, abstractmethod

FetcherInput = TypeVar("FetcherInput")
FetcherOutput = TypeVar("FetcherOutput")


class Fetcher(ABC, Generic[FetcherInput, FetcherOutput]):
    @abstractmethod
    def fetch(self, input: FetcherInput) -> FetcherOutput:
        raise NotImplementedError("Must be implemented")
