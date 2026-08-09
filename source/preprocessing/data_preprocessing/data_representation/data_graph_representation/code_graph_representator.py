import tree_sitter_c as tsc
from tree_sitter import Language, Parser

from source.preprocessing.domain_representator import DomainRepresentator
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph import CodeGraph
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_config import (
    MAX_CODE_BYTES, MAX_NODES, AST_CHILD, NEXT_SIBLING, USE_DEF
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
        last_use: dict[str, int] = {}  # identifier name -> last node index (use-def chain)

        def add_edge(src: int, dst: int, etype: int) -> None:
            graph.edges.append((src, dst))
            graph.edge_types.append(etype)

        def add_node(node) -> int:
            idx = len(graph.node_types)
            graph.node_types.append(node.type)
            if node.child_count == 0 and node.is_named:
                tok = code_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace")
                graph.node_tokens.append(tok)
    
                if node.type == "identifier":
                    prev = last_use.get(tok)
                    if prev is not None:
                        add_edge(prev, idx, USE_DEF)
                    last_use[tok] = idx
            else:
                graph.node_tokens.append("")
            node_index[id(node)] = idx
            return idx

        add_node(root)
        stack = [root]
        capped = False
        while stack and not capped:
            parent = stack.pop()
            parent_idx = node_index[id(parent)]
            prev_sibling_idx: int | None = None
            for child in parent.children:
                if not child.is_named:
                    continue
                child_idx = add_node(child)
                add_edge(parent_idx, child_idx, AST_CHILD)
                
                if prev_sibling_idx is not None:
                    add_edge(prev_sibling_idx, child_idx, NEXT_SIBLING)
                prev_sibling_idx = child_idx
                stack.append(child)
                if graph.num_nodes >= MAX_NODES:
                    capped = True
                    break

        if graph.num_nodes < 2 or not graph.edges:
            return None
        return graph
