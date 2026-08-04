import os
import torch
from tqdm import tqdm
from torch_geometric.data import Data

from source.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from source.preprocessing.data_preprocessing.data_loading.data_loader import DataLoader
from source.preprocessing.data_preprocessing.data_fetching.data_fetcher import DataFetcher
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node import CodeNode
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph import CodeGraph
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_representator import CodeNodeRepresentator
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_representator import CodeGraphRepresentator


class DataPreprocessingPipeline(PreprocessingPipeline[str, list[Data]]):
    def __init__(self, vocab: dict[str, int]):
        self.fetcher = DataFetcher()
        self.loader = DataLoader()
        self.graph_representator = CodeGraphRepresentator()
        self.node_representator = CodeNodeRepresentator(vocab)

    def run(self, input: str) -> list[Data]:
        if not os.path.exists(input):
            self.fetcher.fetch(os.path.dirname(input) or ".")

        graphs: list[Data] = []
        for rec in tqdm(self.loader.load(input), desc="preprocess"):
            code_graph = self.graph_representator.represent(rec["func"])
            if code_graph is None:
                continue
            label = int(rec.get("target", 0))
            graphs.append(self._to_data(code_graph, label))
        return graphs

    def _to_data(self, graph: CodeGraph, label: int) -> Data:
        feats = [
            self.node_representator.represent(
                CodeNode(graph.node_types[i], graph.node_tokens[i])
            )
            for i in range(graph.num_nodes)
        ]
        x = torch.tensor(feats, dtype=torch.float)

        if graph.edges:
            src = [e[0] for e in graph.edges]
            dst = [e[1] for e in graph.edges]
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
        )
