import unittest

from source.preprocessing.data_preprocessing.data_preprocessing_pipeline import DataPreprocessingPipeline
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph import CodeGraph
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_config import NUM_BASE_EDGE_TYPES
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_representator import CodeNodeRepresentator


class ToDataTests(unittest.TestCase):
    def setUp(self):
        vocab = CodeNodeRepresentator.load_or_build_vocab()
        self.pipeline = DataPreprocessingPipeline(vocab)
        self.graph = CodeGraph(
            node_types=["function_definition", "identifier", "identifier"],
            node_tokens=["", "memcpy", "len"],
            edges=[(0, 1), (1, 2)],
            edge_types=[0, 1]
        )

    def test_x_shape_is_num_nodes_by_seven(self):
        data = self.pipeline._to_data(self.graph, label=1)
        self.assertEqual(tuple(data.x.shape), (3, 7))

    def test_edges_are_made_bidirectional(self):
        data = self.pipeline._to_data(self.graph, label=0)
        self.assertEqual(data.edge_index.shape[1], 2 * len(self.graph.edges))
        self.assertEqual(data.edge_attr.shape[0], 2 * len(self.graph.edges))

    def test_reverse_edges_get_offset_type(self):
        data = self.pipeline._to_data(self.graph, label=0)
        attrs = data.edge_attr.tolist()
        forward = attrs[: len(self.graph.edges)]
        reverse = attrs[len(self.graph.edges):]
        self.assertEqual(forward, self.graph.edge_types)
        self.assertEqual(reverse, [t + NUM_BASE_EDGE_TYPES for t in self.graph.edge_types])

    def test_reverse_edge_index_swaps_src_and_dst(self):
        data = self.pipeline._to_data(self.graph, label=0)
        e = data.edge_index.tolist()
        n = len(self.graph.edges)
        src, dst = e[0], e[1]
        self.assertEqual(src[n:], dst[:n])
        self.assertEqual(dst[n:], src[:n])

    def test_label_is_length_one_long_tensor(self):
        data = self.pipeline._to_data(self.graph, label=1)
        self.assertEqual(data.y.tolist(), [1])

    def test_graph_without_edges_yields_empty_edge_tensors(self):
        graph = CodeGraph(node_types=["identifier"], node_tokens=["x"], edges=[], edge_types=[])
        data = self.pipeline._to_data(graph, label=0)
        self.assertEqual(data.edge_index.shape, (2, 0))
        self.assertEqual(data.edge_attr.shape, (0,))


if __name__ == "__main__":
    unittest.main()
