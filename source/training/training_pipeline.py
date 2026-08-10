from typing import TypeVar, Generic
from abc import ABC, abstractmethod

PipelineInput = TypeVar("PipelineInput")
PipelineOutput = TypeVar("PipelineOutput")


class TrainingPipeline(ABC, Generic[PipelineInput, PipelineOutput]):
    """Abstract base for the full train -> validate -> test orchestration.

    A training pipeline wires the batch loaders, model, trainer and evaluator
    together and drives the epoch loop; 'run' returns a summary of the run
    (best epoch, validation and test metrics, checkpoint path).
    """

    @abstractmethod
    def run(self, input: PipelineInput) -> PipelineOutput:
        raise NotImplementedError("Must be implemented")
