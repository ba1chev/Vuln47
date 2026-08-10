import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, JumpingKnowledge, global_max_pool, global_mean_pool

from source.preprocessing.data_preprocessing.data_representation.data_graph_representation.code_graph_config import NUM_EDGE_TYPES
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_config import NUM_TOKEN_FEATURES, NUM_TOKEN_BUCKETS


class Vuln47GNN(nn.Module):
    """Edge-aware GIN (GINE) classifier for function-level vulnerability detection.

    A node's syntactic type and hashed token bucket index learned embeddings,
    concatenated with the hand-crafted token features and projected to
    'hidden_dim'. A stack of 'GINEConv' layers (each consuming a learned
    edge-type embedding) with BatchNorm, ReLU, dropout and residual connections
    passes messages over the typed AST; a JumpingKnowledge readout concatenates
    every layer's output. Graph-level mean+max pooling then feeds a 2-class head.
    """

    def __init__(self, num_types: int, num_token_features: int,
        num_token_buckets: int = NUM_TOKEN_BUCKETS, num_edge_types: int = NUM_EDGE_TYPES,
        type_emb_dim: int = 64, token_emb_dim: int = 32,
        hidden_dim: int = 192, num_layers: int = 4, dropout: float = 0.3):
        super().__init__()

        self.dropout = dropout
        # learned embeddings for the two integer feature columns
        self.type_emb = nn.Embedding(num_types, type_emb_dim)
        self.token_emb = nn.Embedding(num_token_buckets, token_emb_dim)
        # node input = [type emb | token emb | hand-crafted token features]
        in_dim = type_emb_dim + token_emb_dim + num_token_features
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        # edge-type embedding is what makes the conv "edge-aware" (GINE)
        self.edge_emb = nn.Embedding(num_edge_types, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # JumpingKnowledge(cat): classifier sees features from every message-passing depth
        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = num_layers * hidden_dim

        # 2 * jk_dim because mean and max pooled graph vectors are concatenated
        self.classifier = nn.Sequential(
            nn.Linear(2 * jk_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # split the packed 7-D feature rows back into their parts
        type_ids = x[:, 0].long()
        bucket_ids = x[:, 1].long()
        token_feats = x[:, 2:]
        edge_attr = self.edge_emb(data.edge_attr.long())  # edge type id -> vector

        h = torch.cat([self.type_emb(type_ids), self.token_emb(bucket_ids), token_feats], dim=1)
        h = F.relu(self.input_proj(h))

        layer_outs = []
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index, edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new           # residual connection for stable deep training
            layer_outs.append(h)

        h = self.jk(layer_outs)
        # collapse each graph to one vector via both mean (overall) and max (most salient)
        pooled = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch)
        ], dim=1)

        return self.classifier(pooled)


def build_model(vocab, **kwargs) -> Vuln47GNN:
    """Construct a 'Vuln47GNN' sized to the given AST-type vocabulary.

    The vocab length fixes the type-embedding table; extra kwargs (e.g.
    'hidden_dim', 'token_emb_dim') override the model defaults so a saved
    checkpoint's config can rebuild the exact architecture it was trained with.
    """
    return Vuln47GNN(
        num_types=len(vocab),
        num_token_features=NUM_TOKEN_FEATURES,
        **kwargs
    )