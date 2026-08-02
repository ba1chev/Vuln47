import tree_sitter_c as tsc
from tree_sitter import Language, Parser

from source.preprocessing.domain_representator import DomainRepresentator
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph import CodeGraph
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_config import (
    MAX_CODE_BYTES, MAX_NODES
)

# single shared parser (construction is expensive)
_C_LANGUAGE = Language(tsc.language())
_PARSER = Parser(_C_LANGUAGE)


class CodeGraphRepresentator(DomainRepresentator[str, "CodeGraph | None"]):
    def represent(self, input: str) -> CodeGraph | None:
        code_bytes = input.encode("utf-8", "replace")
        if len(code_bytes) == 0 or len(code_bytes) > MAX_CODE_BYTES:
            return None

        tree = _PARSER.parse(code_bytes)
        root = tree.root_node
        if root.child_count == 0:
            return None

        graph = CodeGraph()
        node_index: dict[int, int] = {}

        def add_node(node) -> int:
            idx = len(graph.node_types)
            graph.node_types.append(node.type)
            if node.child_count == 0 and node.is_named:
                tok = code_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace")
                graph.node_tokens.append(tok)
            else:
                graph.node_tokens.append("")
            node_index[id(node)] = idx
            return idx

        add_node(root)
        stack = [root]
        while stack:
            parent = stack.pop()
            parent_idx = node_index[id(parent)]
            for child in parent.children:
                if not child.is_named:
                    continue
                child_idx = add_node(child)
                graph.edges.append((parent_idx, child_idx))
                stack.append(child)
                if graph.num_nodes >= MAX_NODES:
                    stack.clear()
                    break

        if graph.num_nodes < 2 or not graph.edges:
            return None
        return graph
