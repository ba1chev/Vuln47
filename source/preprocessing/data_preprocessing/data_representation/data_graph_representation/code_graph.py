from dataclasses import dataclass, field


@dataclass
class CodeGraph:
    """A lightweight AST of one function held as parallel lists.

    'node_types[i]' / 'node_tokens[i]' describe node 'i' (tokens are only
    set for leaves), and 'edges[j]' / 'edge_types[j]' describe edge 'j'.
    Kept deliberately framework-agnostic — conversion to tensors happens later
    in the pipeline, not here.
    """

    node_types: list[str] = field(default_factory=list)
    node_tokens: list[str] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    edge_types: list[int] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.node_types)
