from typing import TypeVar, Generic
from abc import ABC, abstractmethod

FetcherInput = TypeVar("FetcherInput")
FetcherOutput = TypeVar("FetcherOutput")


class Fetcher(ABC, Generic[FetcherInput, FetcherOutput]):
    """Abstract base for obtaining raw data before it can be loaded.

    A fetcher is responsible for making the dataset present locally (e.g.
    downloading missing splits), returning a handle to what it produced. It is
    expected to be idempotent: inputs already present are reused, not re-fetched.
    """

    @abstractmethod
    def fetch(self, input: FetcherInput) -> FetcherOutput:
        raise NotImplementedError("Must be implemented")
