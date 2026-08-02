from dataclasses import dataclass, field


@dataclass
class CodeGraph:
    node_types: list[str] = field(default_factory=list)
    node_tokens: list[str] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.node_types)
