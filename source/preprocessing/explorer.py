from typing import TypeVar, Generic
from abc import ABC, abstractmethod

ExplorerInput = TypeVar("ExplorerInput")


class Explorer(ABC, Generic[ExplorerInput]):
    """Abstract base for exploratory data analysis over a dataset split.

    An explorer consumes an input handle (e.g. a split path) and produces no
    return value — it exists purely for its side effects (printing stats,
    drawing plots), so 'explore' returns 'None.
    """

    @abstractmethod
    def explore(self, input: ExplorerInput) -> None:
        raise NotImplementedError("Must be implemented")
