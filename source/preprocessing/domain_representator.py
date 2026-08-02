from typing import TypeVar, Generic
from abc import ABC, abstractmethod

RepresentatorInput = TypeVar("RepresentatorInput")
RepresentatorOutput = TypeVar("RepresentatorOutput")


class DomainRepresentator(ABC, Generic[RepresentatorInput, RepresentatorOutput]):
    @abstractmethod
    def represent(self, input: RepresentatorInput) -> RepresentatorOutput:
        raise NotImplementedError("Must be implemented")
