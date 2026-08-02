import os
import json

from source.preprocessing.domain_representator import DomainRepresentator
from source.preprocessing.data_preprocessing.data_loading.data_loader import DataLoader
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node import CodeNode
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_representator import CodeGraphRepresentator
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_config import (
    VOCAB_PATH, DANGEROUS_FUNCS, NUM_TOKEN_FEATURES
)


class CodeNodeRepresentator(DomainRepresentator[CodeNode, list[float]]):
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.unk = vocab["<unk>"]

    def represent(self, input: CodeNode) -> list[float]:
        type_id = self.vocab.get(input.node_type, self.unk)
        return [float(type_id)] + self._token_features(input.token)

    def _token_features(self, token: str) -> list[float]:
        if not token:
            return [0.0] * NUM_TOKEN_FEATURES
        lower = token.lower()
        return [
            1.0,
            1.0 if token in DANGEROUS_FUNCS else 0.0,
            min(len(token) / 20.0, 1.0),
            1.0 if token.isdigit() else 0.0,
            1.0 if any(k in lower for k in ("len", "size", "buf", "count", "idx"))
            else 0.0
        ]

    @classmethod
    def load_or_build_vocab(cls, train_path: str | None = None) -> dict[str, int]:
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH) as f:
                return json.load(f)
        if train_path is None:
            raise FileNotFoundError(
                f"{VOCAB_PATH} missing and no train_path given to build it."
            )
        return cls.build_vocab(train_path)

    @classmethod
    def build_vocab(cls, train_path: str) -> dict[str, int]:
        loader = DataLoader()
        grapher = CodeGraphRepresentator()
        types: set[str] = set()

        for rec in loader.load(train_path):
            graph = grapher.represent(rec["func"])
            if graph is None:
                continue
            types.update(graph.node_types)
        vocab = {"<unk>": 0}
        for i, t in enumerate(sorted(types), start=1):
            vocab[t] = i
        with open(VOCAB_PATH, "w") as f:
            json.dump(vocab, f, indent=2)
        return vocab

    @staticmethod
    def feature_dims(vocab: dict[str, int], type_emb_dim: int = 64) -> dict:
        return {
            "num_types": len(vocab),
            "type_emb_dim": type_emb_dim,
            "num_token_features": NUM_TOKEN_FEATURES,
            "node_feature_dim": type_emb_dim + NUM_TOKEN_FEATURES
        }
