from typing import TypeVar, Generic
from abc import ABC, abstractmethod

RepresentatorInput = TypeVar("RepresentatorInput")
RepresentatorOutput = TypeVar("RepresentatorOutput")


class DomainRepresentator(ABC, Generic[RepresentatorInput, RepresentatorOutput]):
    """Abstract base for turning one domain object into another representation.

    Used at every step where a domain concept is re-expressed for the model —
    e.g. C source into an AST graph, or an AST node into a feature vector. The
    generic parameters fix the concrete input/output types per implementation.
    """

    @abstractmethod
    def represent(self, input: RepresentatorInput) -> RepresentatorOutput:
        raise NotImplementedError("Must be implemented")
