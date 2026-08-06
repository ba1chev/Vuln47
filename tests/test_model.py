import torch
import unittest
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from source.vuln47_gnn_model import build_model
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_config import NUM_EDGE_TYPES
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_representator import CodeNodeRepresentator


def _synthetic_graph(num_nodes: int, num_types: int, label: int) -> Data:
    x = torch.zeros((num_nodes, 7))
    x[:, 0] = torch.randint(0, num_types, (num_nodes,))
    x[:, 1] = torch.randint(0, 4096, (num_nodes,))
    edge_index = torch.tensor(
        [list(range(num_nodes - 1)), list(range(1, num_nodes))], dtype=torch.long
    )
    edge_attr = torch.randint(0, NUM_EDGE_TYPES, (num_nodes - 1,))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))


class ModelForwardTests(unittest.TestCase):
    def setUp(self):
        self.vocab = CodeNodeRepresentator.load_or_build_vocab()
        self.model = build_model(self.vocab)
        self.model.eval()
        graphs = [
            _synthetic_graph(5, len(self.vocab), 1),
            _synthetic_graph(4, len(self.vocab), 0),
            _synthetic_graph(6, len(self.vocab), 1)
        ]
        self.batch = next(iter(DataLoader(graphs, batch_size=3)))

    def test_output_is_two_logits_per_graph(self):
        with torch.no_grad():
            out = self.model(self.batch)
        self.assertEqual(tuple(out.shape), (3, 2))

    def test_output_is_finite(self):
        with torch.no_grad():
            out = self.model(self.batch)
        self.assertTrue(torch.isfinite(out).all())

    def test_deterministic_in_eval_mode(self):
        with torch.no_grad():
            a = self.model(self.batch)
            b = self.model(self.batch)
        self.assertTrue(torch.allclose(a, b))

    def test_type_embedding_table_matches_vocab_size(self):
        self.assertEqual(self.model.type_emb.num_embeddings, len(self.vocab))


if __name__ == "__main__":
    unittest.main()
