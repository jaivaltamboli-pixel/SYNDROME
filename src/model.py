import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool


class ResidualGINEBlock(nn.Module):
    """Residual edge-aware GINEConv block with LayerNorm and GELU."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(mlp, edge_dim=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return x + h


class SyndromeCorrArbiter(nn.Module):
    """
    Calibrated binary validity arbiter predicting logit P(MWPM fails | S, M).
    Uses LayerNorm before the final MLP to prevent additive-pooling saturation at d >= 5.
    """

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [ResidualGINEBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = self.node_enc(data.x)
        edge_attr = self.edge_enc(data.edge_attr)
        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr)
        h_add = global_add_pool(x, data.batch)
        h_mean = global_mean_pool(x, data.batch)
        h = torch.cat([h_add, h_mean], dim=-1)
        return self.head(h).squeeze(-1)
