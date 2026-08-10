import os
import json

from source.preprocessing.domain_representator import DomainRepresentator
from source.preprocessing.data_preprocessing.data_loading.data_loader import DataLoader
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node import CodeNode
from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_representator import CodeGraphRepresentator
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_config import (
    VOCAB_PATH, DANGEROUS_FUNCS, NUM_TOKEN_FEATURES, NUM_TOKEN_BUCKETS, token_bucket
)


class CodeNodeRepresentator(DomainRepresentator[CodeNode, list[float]]):
    """Turns an AST node into the 7-D feature vector the model consumes.

    Layout: '[type_id, token_bucket] + 5 hand-crafted token features'. The two
    integer columns index learned embedding tables inside the model; the five
    features encode C-vulnerability priors (dangerous call, size-like name, ...).
    Also owns building/loading the AST-node-type vocabulary.
    """

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.unk = vocab["<unk>"]  # fallback id for types unseen at vocab-build time

    def represent(self, input: CodeNode) -> list[float]:
        type_id = self.vocab.get(input.node_type, self.unk)
        bucket = token_bucket(input.token)
        return [float(type_id), float(bucket)] + self._token_features(input.token)

    def _token_features(self, token: str) -> list[float]:
        """The 5 hand-crafted per-token priors (all zero for the empty token)."""
        if not token:
            return [0.0] * NUM_TOKEN_FEATURES
        lower = token.lower()
        return [
            1.0,                                                # is a real leaf token
            1.0 if token in DANGEROUS_FUNCS else 0.0,           # known dangerous call
            min(len(token) / 20.0, 1.0),                        # normalized length
            1.0 if token.isdigit() else 0.0,                    # numeric literal
            1.0 if any(k in lower for k in ("len", "size", "buf", "count", "idx"))
            else 0.0                                            # size/index-like name
        ]

    @classmethod
    def load_or_build_vocab(cls, train_path: str | None = None) -> dict[str, int]:
        """Return the cached AST-type vocab, building it from train if absent."""
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
        """Scan the train split for every AST node type and persist a stable map."""
        loader = DataLoader()
        grapher = CodeGraphRepresentator()
        types: set[str] = set()

        for rec in loader.load(train_path):
            graph = grapher.represent(rec["func"])
            if graph is None:
                continue
            types.update(graph.node_types)
        # id 0 is reserved for unknown types; the rest are sorted for determinism
        vocab = {"<unk>": 0}
        for i, t in enumerate(sorted(types), start=1):
            vocab[t] = i
        with open(VOCAB_PATH, "w") as f:
            json.dump(vocab, f, indent=2)
        return vocab

    @staticmethod
    def feature_dims(vocab: dict[str, int], type_emb_dim: int = 64,
        token_emb_dim: int = 32) -> dict:
        """Report the feature/embedding dimensions (handy for sizing the model)."""
        return {
            "num_types": len(vocab),
            "type_emb_dim": type_emb_dim,
            "num_token_buckets": NUM_TOKEN_BUCKETS,
            "token_emb_dim": token_emb_dim,
            "num_token_features": NUM_TOKEN_FEATURES,
            "node_feature_dim": type_emb_dim + token_emb_dim + NUM_TOKEN_FEATURES
        }
