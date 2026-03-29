"""Utility classes and functions for the companion codebase.

PlaceHolder — lightweight container for (X, E, y) triples with masking.
load_config — read a YAML file into a nested SimpleNamespace for dot-access.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# PlaceHolder
# ---------------------------------------------------------------------------

class PlaceHolder:
    """Lightweight container for node (X), edge (E), and global (y) features."""

    def __init__(self, X: torch.Tensor, E: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor) -> "PlaceHolder":
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def to(self, device) -> "PlaceHolder":
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device)
        return self

    def mask(self, node_mask: torch.Tensor, collapse: bool = False) -> "PlaceHolder":
        """Zero out features at padded positions.

        Args:
            node_mask: (bs, n) boolean or float mask, 1 = real node.
            collapse: if True, argmax features and mark padded as -1.
        """
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)
            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


# ---------------------------------------------------------------------------
# Dense conversion
# ---------------------------------------------------------------------------

def encode_no_edge(E: torch.Tensor) -> torch.Tensor:
    """Ensure the 'no-edge' class (index 0) is set for absent edges."""
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E



# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def load_config(path: str) -> SimpleNamespace:
    """Load a YAML config file and return a SimpleNamespace with dot-access."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    return _dict_to_namespace(raw)


def namespace_to_dict(ns: SimpleNamespace) -> dict:
    """Convert a SimpleNamespace back to a plain dict (recursively)."""
    d = {}
    for k, v in vars(ns).items():
        if isinstance(v, SimpleNamespace):
            d[k] = namespace_to_dict(v)
        else:
            d[k] = v
    return d


# ---------------------------------------------------------------------------
# Masking helpers (used by diffusion)
# ---------------------------------------------------------------------------

def assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        "Variables not masked properly."


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1).sum(dim=-1)


def sample_discrete_features(probX: torch.Tensor, probE: torch.Tensor,
                              node_mask: torch.Tensor) -> PlaceHolder:
    """Sample node and edge types from multinomial distributions.

    Args:
        probX: (bs, n, dx) node-type probabilities.
        probE: (bs, n, n, de) edge-type probabilities.
        node_mask: (bs, n) boolean mask.

    Returns:
        PlaceHolder with integer X (bs, n), E (bs, n, n), y empty.
    """
    bs, n, _ = probX.shape

    # Masked rows get uniform distribution so multinomial doesn't fail
    probX[~node_mask] = 1 / probX.shape[-1]
    probX = probX.reshape(bs * n, -1)
    X_t = probX.multinomial(1).reshape(bs, n)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)

    E_t = probE.multinomial(1).reshape(bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def sample_discrete_feature_noise(limit_dist: PlaceHolder,
                                   node_mask: torch.Tensor) -> PlaceHolder:
    """Sample from the limit distribution of the diffusion process."""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)

    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Upper-triangular symmetrization
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)
    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)


def compute_batched_over0_posterior_distribution(
    X_t: torch.Tensor, Qt: torch.Tensor,
    Qsb: torch.Tensor, Qtb: torch.Tensor
) -> torch.Tensor:
    """Compute xt @ Qt.T * x0 @ Qsb / (x0 @ Qtb @ xt.T) for each possible x0.

    Args:
        X_t: (bs, n, dt) or (bs, n, n, dt)
        Qt:  (bs, d_{t-1}, dt)
        Qsb: (bs, d0, d_{t-1})
        Qtb: (bs, d0, dt)

    Returns:
        (bs, N, d0, d_{t-1}) unnormalized posterior.
    """
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)

    Qt_T = Qt.transpose(-1, -2)
    left_term = X_t @ Qt_T                       # bs, N, d_{t-1}
    left_term = left_term.unsqueeze(dim=2)       # bs, N, 1, d_{t-1}

    right_term = Qsb.unsqueeze(1)                # bs, 1, d0, d_{t-1}
    numerator = left_term * right_term           # bs, N, d0, d_{t-1}

    X_t_transposed = X_t.transpose(-1, -2)       # bs, dt, N
    prod = Qtb @ X_t_transposed                  # bs, d0, N
    prod = prod.transpose(-1, -2)                # bs, N, d0
    denominator = prod.unsqueeze(-1)             # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    return numerator / denominator


def mask_distributions(
    true_X: torch.Tensor, true_E: torch.Tensor,
    pred_X: torch.Tensor, pred_E: torch.Tensor,
    node_mask: torch.Tensor,
):
    """Set masked rows to valid distributions so they don't contribute to loss."""
    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.0
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.0

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    pred_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    true_X = true_X + 1e-7
    pred_X = pred_X + 1e-7
    true_E = true_E + 1e-7
    pred_E = pred_E + 1e-7

    true_X = true_X / torch.sum(true_X, dim=-1, keepdim=True)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    true_E = true_E / torch.sum(true_E, dim=-1, keepdim=True)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    return true_X, true_E, pred_X, pred_E


def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    """Compute posterior distributions q(z_s | z_t, z_0) for X and E."""
    prob_X = _compute_posterior(M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)
    prob_E = _compute_posterior(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)
    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)


def _compute_posterior(M, M_t, Qt_M, Qsb_M, Qtb_M):
    """Compute xt @ Qt.T * x0 @ Qsb / (x0 @ Qtb @ xt.T)."""
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)

    Qt_M_T = torch.transpose(Qt_M, -2, -1)

    left_term = M_t @ Qt_M_T
    right_term = M @ Qsb_M
    product = left_term * right_term

    denom = M @ Qtb_M
    denom = (denom * M_t).sum(dim=-1)

    return product / denom.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Cosine noise schedule helpers
# ---------------------------------------------------------------------------

def cosine_beta_schedule_discrete(timesteps: int, s: float = 0.008):
    """Cosine schedule producing per-step betas for discrete diffusion."""
    import numpy as np
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== utils.py smoke test ===")

    # PlaceHolder mask test
    bs, n, dx, de = 2, 5, 3, 4
    X = torch.randn(bs, n, dx)
    E = torch.randn(bs, n, n, de)
    E = (E + E.transpose(1, 2)) / 2  # symmetrize
    y = torch.randn(bs, 8)
    node_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)

    ph = PlaceHolder(X=X.clone(), E=E.clone(), y=y.clone())
    ph.mask(node_mask)

    # Check that padded positions are zero
    assert (ph.X[0, 3:] == 0).all(), "Padded X not zeroed"
    assert (ph.X[0, :3] != 0).any(), "Real X should not be all zero"
    assert (ph.E[0, 3:, :] == 0).all(), "Padded E rows not zeroed"
    assert (ph.E[0, :, 3:] == 0).all(), "Padded E cols not zeroed"
    print("  PlaceHolder.mask() OK")

    # type_as test
    ph2 = PlaceHolder(X=torch.randn(1, 2, 3), E=torch.randn(1, 2, 2, 3),
                       y=torch.randn(1, 1))
    ref = torch.zeros(1, dtype=torch.float64)
    ph2.type_as(ref)
    assert ph2.X.dtype == torch.float64, "type_as failed"
    print("  PlaceHolder.type_as() OK")

    # to() test
    ph2.to("cpu")
    print("  PlaceHolder.to() OK")

    # load_config test
    import tempfile
    cfg_dict = {"model": {"n_layers": 8, "hidden_dims": {"dx": 256}}, "train": {"lr": 0.0002}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_dict, f)
        tmp_path = f.name
    cfg = load_config(tmp_path)
    assert cfg.model.n_layers == 8
    assert cfg.model.hidden_dims.dx == 256
    assert cfg.train.lr == 0.0002
    os.unlink(tmp_path)
    print("  load_config() OK")

    # namespace_to_dict roundtrip
    d = namespace_to_dict(cfg)
    assert d["model"]["n_layers"] == 8
    print("  namespace_to_dict() OK")

    # sample_discrete_features test
    probX = torch.ones(2, 4, 3) / 3
    probE = torch.ones(2, 4, 4, 2) / 2
    nm = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    sampled = sample_discrete_features(probX, probE, nm)
    assert sampled.X.shape == (2, 4), f"Bad X shape: {sampled.X.shape}"
    assert sampled.E.shape == (2, 4, 4), f"Bad E shape: {sampled.E.shape}"
    assert (sampled.E == sampled.E.transpose(1, 2)).all(), "E not symmetric"
    print("  sample_discrete_features() OK")

    print("All utils.py tests passed.")
