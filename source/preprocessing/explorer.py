from typing import TypeVar, Generic
from abc import ABC, abstractmethod

ExplorerInput = TypeVar("ExplorerInput")


class Explorer(ABC, Generic[ExplorerInput]):
    @abstractmethod
    def explore(self, input: ExplorerInput) -> None:
        raise NotImplementedError("Must be implemented")
