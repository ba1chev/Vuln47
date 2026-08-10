from dataclasses import dataclass


@dataclass
class CodeNode:
    """One AST node passed to the node representator: its type plus, for leaves,
    the literal source token (empty for internal/unnamed nodes)."""

    node_type: str
    token: str = ""