from typing import TypeVar, Generic
from abc import ABC, abstractmethod

EvaluatorInput = TypeVar("EvaluatorInput")
EvaluatorOutput = TypeVar("EvaluatorOutput")


class Evaluator(ABC, Generic[EvaluatorInput, EvaluatorOutput]):
    @abstractmethod
    def evaluate(self, input: EvaluatorInput) -> EvaluatorOutput:
        raise NotImplementedError("Must be implemented")
