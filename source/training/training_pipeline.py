from typing import TypeVar, Generic
from abc import ABC, abstractmethod

PipelineInput = TypeVar("PipelineInput")
PipelineOutput = TypeVar("PipelineOutput")


class TrainingPipeline(ABC, Generic[PipelineInput, PipelineOutput]):
    @abstractmethod
    def run(self, input: PipelineInput) -> PipelineOutput:
        raise NotImplementedError("Must be implemented")
