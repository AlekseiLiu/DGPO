"""Graph Transformer for discrete graph diffusion.

Faithful rewrite of the original DiGress Graph Transformer with positional
encoding. Architecture: input projection MLPs -> sinusoidal positional encoding
-> stack of XEyTransformerLayer (NodeEdgeBlock + FFN) -> output projection MLPs
with skip connections.

No PyTorch Lightning, no wandb, no Hydra.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Ensure `src.*` resolves to companion/src even when launched from repo root.
_companion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _companion_root not in sys.path:
    sys.path.insert(0, _companion_root)

from src.utils import PlaceHolder


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------

class Xtoy(nn.Module):
    """Aggregate node features to a global feature vector."""
    def __init__(self, dx: int, dy: int):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X: Tensor) -> Tensor:
        # X: (bs, n, dx)
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class Etoy(nn.Module):
    """Aggregate edge features to a global feature vector."""
    def __init__(self, de: int, dy: int):
        super().__init__()
        self.lin = nn.Linear(4 * de, dy)

    def forward(self, E: Tensor) -> Tensor:
        # E: (bs, n, n, de)
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


def masked_softmax(x: Tensor, mask: Tensor, **kwargs) -> Tensor:
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


# ---------------------------------------------------------------------------
# NodeEdgeBlock — self-attention with edge modulation (FiLM)
# ---------------------------------------------------------------------------

class NodeEdgeBlock(nn.Module):
    """Self-attention layer that updates node, edge, and global features."""
    def __init__(self, dx: int, de: int, dy: int, n_head: int, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)

        # FiLM E to X
        self.e_add = nn.Linear(de, dx)
        self.e_mul = nn.Linear(de, dx)

        # FiLM y to E (note: output is dx, not de)
        self.y_e_mul = nn.Linear(dy, dx)
        self.y_e_add = nn.Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)

        # Process y
        self.y_y = nn.Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor):
        """
        X: (bs, n, dx)    E: (bs, n, n, de)    y: (bs, dy)    node_mask: (bs, n)
        Returns: newX, newE, new_y with the same shapes.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Q, K from nodes
        Q = self.q(X) * x_mask                  # (bs, n, dx)
        K = self.k(X) * x_mask

        Q = Q.reshape(bs, n, self.n_head, self.df)
        K = K.reshape(bs, n, self.n_head, self.df)

        Q = Q.unsqueeze(2)                      # (bs, n, 1, n_head, df)
        K = K.unsqueeze(1)                      # (bs, 1, n, n_head, df)

        # Unnormalized attention
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))

        # Edge modulation (FiLM)
        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E1 = E1.reshape(bs, n, n, self.n_head, self.df)

        E2 = self.e_add(E) * e_mask1 * e_mask2
        E2 = E2.reshape(bs, n, n, self.n_head, self.df)

        Y = Y * (E1 + 1) + E2                   # (bs, n, n, n_head, df)

        # Global modulation on edges
        newE = Y.flatten(start_dim=3)            # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2

        # Softmax attention
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = masked_softmax(Y, softmax_mask, dim=2)

        # V from nodes
        V = self.v(X) * x_mask
        V = V.reshape(bs, n, self.n_head, self.df)
        V = V.unsqueeze(1)                       # (bs, 1, n, n_head, df)

        # Weighted sum
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, dx

        # Global modulation on nodes
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask

        # Process y from X and E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)

        return newX, newE, new_y


# ---------------------------------------------------------------------------
# XEyTransformerLayer — attention block + FFN with residual + layer norm
# ---------------------------------------------------------------------------

class XEyTransformerLayer(nn.Module):
    """Transformer layer updating node, edge, and global features."""
    def __init__(self, dx: int, de: int, dy: int, n_head: int,
                 dim_ffX: int = 2048, dim_ffE: int = 128, dim_ffy: int = 2048,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5,
                 device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = nn.Linear(dx, dim_ffX, **kw)
        self.linX2 = nn.Linear(dim_ffX, dx, **kw)
        self.normX1 = nn.LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = nn.LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = nn.Dropout(dropout)
        self.dropoutX2 = nn.Dropout(dropout)
        self.dropoutX3 = nn.Dropout(dropout)

        self.linE1 = nn.Linear(de, dim_ffE, **kw)
        self.linE2 = nn.Linear(dim_ffE, de, **kw)
        self.normE1 = nn.LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = nn.LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = nn.Dropout(dropout)
        self.dropoutE2 = nn.Dropout(dropout)
        self.dropoutE3 = nn.Dropout(dropout)

        self.lin_y1 = nn.Linear(dy, dim_ffy, **kw)
        self.lin_y2 = nn.Linear(dim_ffy, dy, **kw)
        self.norm_y1 = nn.LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = nn.LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = nn.Dropout(dropout)
        self.dropout_y2 = nn.Dropout(dropout)
        self.dropout_y3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor):
        """X: (bs, n, dx)  E: (bs, n, n, de)  y: (bs, dy)  node_mask: (bs, n)"""
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        # Residual + LayerNorm for X
        X = self.normX1(X + self.dropoutX1(newX))
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        X = self.normX2(X + self.dropoutX3(ff_outputX))

        # Residual + LayerNorm for E
        E = self.normE1(E + self.dropoutE1(newE))
        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        E = self.normE2(E + self.dropoutE3(ff_outputE))

        # Residual + LayerNorm for y
        y = self.norm_y1(y + self.dropout_y1(new_y))
        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        y = self.norm_y2(y + self.dropout_y3(ff_output_y))

        return X, E, y


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to node features."""
    def __init__(self, dim: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(1e4) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, node_mask: Tensor | None = None) -> Tensor:
        bs, n, d = x.shape
        if n > self.pe.size(0):
            self._extend(n)
        enc = self.pe[:n].unsqueeze(0).expand(bs, -1, -1)
        enc = enc.to(x.device, x.dtype)
        if node_mask is not None:
            mask_bool = node_mask.bool() if node_mask.dtype != torch.bool else node_mask
            enc = enc * mask_bool.unsqueeze(-1)
        x = x + enc
        return self.dropout(x)

    def _extend(self, new_len: int):
        device = self.pe.device
        dim = self.dim
        position = torch.arange(0, new_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(1e4) / dim)
        )
        pe = torch.zeros(new_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe


# ---------------------------------------------------------------------------
# GraphTransformer — the full model
# ---------------------------------------------------------------------------

class GraphTransformer(nn.Module):
    """Graph Transformer with positional encoding for discrete graph diffusion.

    Args:
        n_layers: number of transformer layers.
        input_dims: dict with keys 'X', 'E', 'y' — input feature dimensions.
        hidden_mlp_dims: dict with keys 'X', 'E', 'y' — MLP intermediate dims.
        hidden_dims: dict with keys 'dx', 'de', 'dy', 'n_head', 'dim_ffX',
                     'dim_ffE', optionally 'dim_ffy', 'max_nodes', 'pos_dropout'.
        output_dims: dict with keys 'X', 'E', 'y' — output feature dimensions.
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 hidden_dims: dict, output_dims: dict,
                 act_fn_in: nn.Module = nn.ReLU(),
                 act_fn_out: nn.Module = nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]), act_fn_in,
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]), act_fn_in,
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]), act_fn_in,
        )

        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(
                dx=hidden_dims["dx"], de=hidden_dims["de"], dy=hidden_dims["dy"],
                n_head=hidden_dims["n_head"],
                dim_ffX=hidden_dims["dim_ffX"],
                dim_ffE=hidden_dims["dim_ffE"],
                # NOTE: original codebase does not pass dim_ffy here, so the
                # XEyTransformerLayer default (2048) is always used regardless
                # of config.  We replicate that behavior for result parity.
            )
            for _ in range(n_layers)
        ])

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )
        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
        )

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            dim=hidden_dims["dx"],
            max_len=hidden_dims.get("max_nodes", 15),
            dropout=hidden_dims.get("pos_dropout", 0.0),
        )

    def forward(self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor) -> PlaceHolder:
        """
        X: (bs, n, input_X)  E: (bs, n, n, input_E)  y: (bs, input_y)
        node_mask: (bs, n)
        Returns: PlaceHolder(X, E, y) with output dimensions.
        """
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n, device=X.device)
        diag_mask = (~diag_mask.bool()).unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        # Skip connections
        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        # Input projections + symmetrize E
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = PlaceHolder(
            X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)
        ).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        # Positional encoding
        X = self.pos_enc(X, node_mask=node_mask)

        # Transformer stack
        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        # Output projections
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        # Skip connections + masking
        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        # Enforce symmetry
        E = 0.5 * (E + E.transpose(1, 2))

        return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== model.py smoke test ===")

    # Dimensions matching NB101 setup
    n_layers = 8
    dx, de, dy = 256, 64, 64
    n_head = 8
    num_node_types = 5   # NB101: 5 op types
    num_edge_types = 2   # edge / no-edge
    # Extra features: timestep (1 dim for y)
    input_X = num_node_types + 0   # noisy X_t features
    input_E = num_edge_types + 0
    input_y = 0 + 1                # timestep embedding

    input_dims = {"X": input_X, "E": input_E, "y": input_y}
    hidden_mlp_dims = {"X": 256, "E": 128, "y": 128}
    hidden_dims = {
        "dx": dx, "de": de, "dy": dy, "n_head": n_head,
        "dim_ffX": 256, "dim_ffE": 128, "dim_ffy": 128,
        "max_nodes": 15,
    }
    output_dims = {"X": num_node_types, "E": num_edge_types, "y": 0}

    model = GraphTransformer(
        n_layers=n_layers,
        input_dims=input_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created: {total_params:,} parameters")

    # Forward pass: batch of 4 graphs with 7 nodes
    bs, n = 4, 7
    X = torch.randn(bs, n, input_X)
    E = torch.randn(bs, n, n, input_E)
    E = (E + E.transpose(1, 2)) / 2
    y = torch.randn(bs, input_y)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    node_mask[0, 5:] = False  # first graph has 5 nodes
    node_mask[1, 6:] = False  # second graph has 6 nodes

    model.eval()
    with torch.no_grad():
        out = model(X, E, y, node_mask)

    print(f"  Input:  X={X.shape} E={E.shape} y={y.shape}")
    print(f"  Output: X={out.X.shape} E={out.E.shape} y={out.y.shape}")
    assert out.X.shape == (bs, n, num_node_types), f"Bad X shape: {out.X.shape}"
    assert out.E.shape == (bs, n, n, num_edge_types), f"Bad E shape: {out.E.shape}"
    assert out.y.shape == (bs, 0), f"Bad y shape: {out.y.shape}"

    # Check masking: padded positions should be zero
    assert (out.X[0, 5:] == 0).all(), "Padded X not zeroed"
    assert (out.E[0, 5:, :] == 0).all(), "Padded E rows not zeroed"
    assert (out.E[0, :, 5:] == 0).all(), "Padded E cols not zeroed"
    print("  Masking OK")

    # Check symmetry
    assert torch.allclose(out.E, out.E.transpose(1, 2)), "E not symmetric"
    print("  Symmetry OK")

    # Check diagonal is zero
    for i in range(bs):
        diag = torch.diagonal(out.E[i], dim1=0, dim2=1)
        assert (diag == 0).all(), f"Diagonal not zero for graph {i}"
    print("  Diagonal zeroed OK")

    print("All model.py tests passed.")
