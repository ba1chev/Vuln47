from typing import TypeVar, Generic
from abc import ABC, abstractmethod

PipelineInput = TypeVar("PipelineInput")
PipelineOutput = TypeVar("PipelineOutput")


class PreprocessingPipeline(ABC, Generic[PipelineInput, PipelineOutput]):
    """Abstract base for a preprocessing pipeline that chains several stages.

    A pipeline orchestrates loaders and representators end to end, exposing a
    single 'run' entry point that takes a raw input (e.g. a split path) and
    returns the model-ready output (e.g. a list of PyG graphs).
    """

    @abstractmethod
    def run(self, input: PipelineInput) -> PipelineOutput:
        raise NotImplementedError("Must be implemented")
