from dataclasses import dataclass


@dataclass
class CodeNode:
    node_type: str
    token: str = ""