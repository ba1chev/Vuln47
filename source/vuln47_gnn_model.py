import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_config import NUM_TOKEN_FEATURES


class Vuln47GNN(nn.Module):
    def __init__(self, num_types: int, num_token_features: int, type_emb_dim: int = 64,
        hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.3):
        super().__init__()

        self.dropout = dropout
        self.type_emb = nn.Embedding(num_types, type_emb_dim)
        in_dim = type_emb_dim + num_token_features
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        type_ids = x[:, 0].long()
        token_feats = x[:, 1:]

        h = torch.cat([self.type_emb(type_ids), token_feats], dim=1)
        h = F.relu(self.input_proj(h))

        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new

        pooled = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch)
        ], dim=1)

        return self.classifier(pooled)


def build_model(vocab, **kwargs) -> Vuln47GNN:
    return Vuln47GNN(
        num_types=len(vocab),
        num_token_features=NUM_TOKEN_FEATURES,
        **kwargs
    )