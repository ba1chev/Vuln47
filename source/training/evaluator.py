from typing import TypeVar, Generic
from abc import ABC, abstractmethod

EvaluatorInput = TypeVar("EvaluatorInput")
EvaluatorOutput = TypeVar("EvaluatorOutput")


class Evaluator(ABC, Generic[EvaluatorInput, EvaluatorOutput]):
    """Abstract base for evaluating a model on a data split.

    An evaluator runs the model over the given batches (no gradient updates)
    and returns a summary of quality — for this project a dict of the
    imbalance-aware metrics (F1, precision, recall, PR-AUC).
    """

    @abstractmethod
    def evaluate(self, input: EvaluatorInput) -> EvaluatorOutput:
        raise NotImplementedError("Must be implemented")
