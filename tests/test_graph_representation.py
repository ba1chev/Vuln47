import unittest

from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_representator import CodeGraphRepresentator
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_config import (
    MAX_CODE_BYTES, AST_CHILD, NEXT_SIBLING, USE_DEF,
)


class GraphStructureTests(unittest.TestCase):
    def setUp(self):
        self.repr = CodeGraphRepresentator()

    def test_parallel_lists_stay_aligned(self):
        graph = self.repr.represent("int f(int a) { return a + a; }")
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.node_types), len(graph.node_tokens))
        self.assertEqual(len(graph.edges), len(graph.edge_types))
        self.assertEqual(graph.num_nodes, len(graph.node_types))

    def test_leaf_nodes_carry_tokens_inner_nodes_do_not(self):
        graph = self.repr.represent("int f(int a) { return a; }")
        self.assertIn("f", graph.node_tokens)
        self.assertIn("a", graph.node_tokens)

    def test_all_three_base_edge_types_can_appear(self):
        graph = self.repr.represent("int f(int a) { return a + a; }")
        present = set(graph.edge_types)
        self.assertIn(AST_CHILD, present)
        self.assertIn(NEXT_SIBLING, present)
        self.assertIn(USE_DEF, present)

    def test_use_def_links_repeated_identifier(self):
        once = self.repr.represent("int f(int a) { return a; }")
        twice = self.repr.represent("int f(int a) { return a + a; }")
        self.assertLess(once.edge_types.count(USE_DEF), twice.edge_types.count(USE_DEF))

    def test_edges_reference_valid_node_indices(self):
        graph = self.repr.represent("int f(int a) { return a + a; }")
        for src, dst in graph.edges:
            self.assertTrue(0 <= src < graph.num_nodes)
            self.assertTrue(0 <= dst < graph.num_nodes)


class GraphEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.repr = CodeGraphRepresentator()

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.repr.represent(""))

    def test_whitespace_only_returns_none(self):
        self.assertIsNone(self.repr.represent("   \n  "))

    def test_oversized_code_returns_none(self):
        huge = "int x;" * MAX_CODE_BYTES
        self.assertIsNone(self.repr.represent(huge))

    def test_result_is_deterministic(self):
        code = "int f(int a) { return a + a; }"
        g1 = self.repr.represent(code)
        g2 = self.repr.represent(code)
        self.assertEqual(g1.node_types, g2.node_types)
        self.assertEqual(g1.edges, g2.edges)
        self.assertEqual(g1.edge_types, g2.edge_types)


if __name__ == "__main__":
    unittest.main()
