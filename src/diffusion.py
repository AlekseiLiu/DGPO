"""Discrete diffusion engine for graph generation.

Implements:
- PredefinedNoiseScheduleDiscrete: cosine beta schedule -> alpha_bar lookup.
- DiscreteUniformTransition: uniform transition matrices Q_t.
- MarginalUniformTransition: transition matrices from marginal distribution.
- Forward process: apply_noise(X, E, y, node_mask, t).
- Reverse process: sample_p_zs_given_zt(...), sample_batch(...).

Self-contained — no imports from model.py.
No PyTorch Lightning, no wandb, no Hydra.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure companion/src resolves correctly
_companion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _companion_root not in sys.path:
    sys.path.insert(0, _companion_root)

from src.utils import (
    PlaceHolder,
    compute_batched_over0_posterior_distribution,
    cosine_beta_schedule_discrete,
    sample_discrete_feature_noise,
    sample_discrete_features,
)


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class PredefinedNoiseScheduleDiscrete(nn.Module):
    """Lookup table for cosine (or custom) discrete noise schedules."""

    def __init__(self, noise_schedule: str, timesteps: int):
        super().__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            betas = cosine_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(f"Unknown noise schedule: {noise_schedule}")

        self.register_buffer("betas", torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)
        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert (t_normalized is None) != (t_int is None)
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert (t_normalized is None) != (t_int is None)
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


# ---------------------------------------------------------------------------
# Transition matrices
# ---------------------------------------------------------------------------

class DiscreteUniformTransition:
    """Uniform transition: Q_t = (1 - beta_t) * I + beta_t / K."""

    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, x_classes, x_classes) / x_classes if x_classes > 0 else torch.ones(1, x_classes, x_classes)
        self.u_e = torch.ones(1, e_classes, e_classes) / e_classes if e_classes > 0 else torch.ones(1, e_classes, e_classes)
        self.u_y = torch.ones(1, y_classes, y_classes) / y_classes if y_classes > 0 else torch.ones(1, y_classes, y_classes)

    def get_Qt(self, beta_t, device):
        beta_t = beta_t.unsqueeze(1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)
        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        alpha_bar_t = alpha_bar_t.unsqueeze(1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y
        return PlaceHolder(X=q_x, E=q_e, y=q_y)


class MarginalUniformTransition:
    """Transition where the limit distribution matches training data marginals."""

    def __init__(self, x_marginals: torch.Tensor, e_marginals: torch.Tensor, y_classes: int):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        # Each row of u_x is the marginal distribution (limit)
        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        beta_t = beta_t.unsqueeze(1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)
        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        alpha_bar_t = alpha_bar_t.unsqueeze(1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y
        return PlaceHolder(X=q_x, E=q_e, y=q_y)


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------

def apply_noise(X, E, y, node_mask, noise_schedule, transition_model, T, training=True):
    """Sample noise and apply it to produce noisy data z_t.

    Args:
        X: (bs, n, dx) one-hot node features.
        E: (bs, n, n, de) one-hot edge features.
        y: (bs, dy) global features.
        node_mask: (bs, n) boolean mask.
        noise_schedule: PredefinedNoiseScheduleDiscrete instance.
        transition_model: DiscreteUniformTransition or MarginalUniformTransition.
        T: int, total diffusion steps.
        training: if True, sample t from [0, T]; if False, from [1, T].

    Returns:
        dict with keys: t_int, t, beta_t, alpha_s_bar, alpha_t_bar, X_t, E_t, y_t, node_mask.
    """
    device = X.device
    lowest_t = 0 if training else 1
    t_int = torch.randint(lowest_t, T + 1, size=(X.size(0), 1), device=device).float()
    s_int = t_int - 1

    t_float = t_int / T
    s_float = s_int / T

    beta_t = noise_schedule(t_normalized=t_float)
    alpha_s_bar = noise_schedule.get_alpha_bar(t_normalized=s_float)
    alpha_t_bar = noise_schedule.get_alpha_bar(t_normalized=t_float)

    Qtb = transition_model.get_Qt_bar(alpha_t_bar, device=device)
    assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all()
    assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

    # Compute transition probabilities
    Xdim_output = Qtb.X.shape[-1]
    Edim_output = Qtb.E.shape[-1]

    probX = X @ Qtb.X       # (bs, n, dx_out)
    probE = E @ Qtb.E.unsqueeze(1)   # (bs, n, n, de_out)

    sampled_t = sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

    X_t = F.one_hot(sampled_t.X, num_classes=Xdim_output).float()
    E_t = F.one_hot(sampled_t.E, num_classes=Edim_output).float()
    assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

    z_t = PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

    return {
        "t_int": t_int, "t": t_float, "beta_t": beta_t,
        "alpha_s_bar": alpha_s_bar, "alpha_t_bar": alpha_t_bar,
        "X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask,
    }


# ---------------------------------------------------------------------------
# Reverse process
# ---------------------------------------------------------------------------

def compute_extra_data(noisy_data):
    """Compute extra input features (timestep embedding appended to y)."""
    t = noisy_data["t"]
    bs = t.shape[0]
    n = noisy_data["X_t"].shape[1]
    dx = noisy_data["X_t"].shape[2]
    de = noisy_data["E_t"].shape[3]

    # No extra node/edge features in companion (no domain features)
    extra_X = torch.zeros(bs, n, 0, device=t.device)
    extra_E = torch.zeros(bs, n, n, 0, device=t.device)
    extra_y = t  # timestep as the only extra global feature

    return PlaceHolder(X=extra_X, E=extra_E, y=extra_y)


def forward_pass(model, noisy_data, node_mask):
    """Concatenate noisy data with extra features and run through model."""
    extra_data = compute_extra_data(noisy_data)
    X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
    E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
    y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
    return model(X, E, y, node_mask)


def sample_p_zs_given_zt(s, t, X_t, E_t, y_t, node_mask,
                          model, noise_schedule, transition_model,
                          Xdim_output, Edim_output):
    """Sample z_s ~ p(z_s | z_t). One reverse denoising step.

    Args:
        s: (bs, 1) normalized timestep s = (t-1)/T.
        t: (bs, 1) normalized timestep t/T.
        X_t: (bs, n, dx) one-hot noisy node features.
        E_t: (bs, n, n, de) one-hot noisy edge features.
        y_t: (bs, dy) global features.
        node_mask: (bs, n).
        model: GraphTransformer instance.
        noise_schedule, transition_model: diffusion components.
        Xdim_output, Edim_output: number of classes for one-hot encoding.

    Returns:
        (out_one_hot, out_discrete): PlaceHolder pairs.
    """
    bs, n, dxs = X_t.shape
    device = X_t.device

    beta_t = noise_schedule(t_normalized=t)
    alpha_s_bar = noise_schedule.get_alpha_bar(t_normalized=s)
    alpha_t_bar = noise_schedule.get_alpha_bar(t_normalized=t)

    Qtb = transition_model.get_Qt_bar(alpha_t_bar, device)
    Qsb = transition_model.get_Qt_bar(alpha_s_bar, device)
    Qt = transition_model.get_Qt(beta_t, device)

    # Neural net prediction
    noisy_data = {"X_t": X_t, "E_t": E_t, "y_t": y_t, "t": t, "node_mask": node_mask}
    pred = forward_pass(model, noisy_data, node_mask)

    pred_X = F.softmax(pred.X, dim=-1)
    pred_E = F.softmax(pred.E, dim=-1)

    # Posterior p(z_s | z_t, z_0) marginalized over predicted z_0
    p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(
        X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X)
    p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(
        X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)

    # Weight by predicted clean distribution
    weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
    unnormalized_prob_X = weighted_X.sum(dim=2)
    unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
    prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)

    pred_E_flat = pred_E.reshape((bs, -1, pred_E.shape[-1]))
    weighted_E = pred_E_flat.unsqueeze(-1) * p_s_and_t_given_0_E
    unnormalized_prob_E = weighted_E.sum(dim=-2)
    unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
    prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
    prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

    assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
    assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

    sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

    X_s = F.one_hot(sampled_s.X, num_classes=Xdim_output).float()
    E_s = F.one_hot(sampled_s.E, num_classes=Edim_output).float()

    assert (E_s == torch.transpose(E_s, 1, 2)).all()
    assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

    out_one_hot = PlaceHolder(X=X_s, E=E_s, y=torch.zeros(bs, 0, device=device))
    out_discrete = PlaceHolder(X=X_s, E=E_s, y=torch.zeros(bs, 0, device=device))

    return (out_one_hot.mask(node_mask).type_as(y_t),
            out_discrete.mask(node_mask, collapse=True).type_as(y_t))


@torch.no_grad()
def sample_batch(model, n_samples, T, limit_dist, node_dist,
                 noise_schedule, transition_model,
                 Xdim_output, Edim_output, device, num_nodes=None):
    """Generate a batch of graphs via full reverse diffusion.

    Returns symmetric graphs (not DAG-extracted). Callers that need upper-
    triangular DAG form for NASBench API queries must apply extraction
    downstream (e.g. in reward computation or evaluation).

    Args:
        model: GraphTransformer instance.
        n_samples: number of graphs to generate.
        T: number of diffusion steps.
        limit_dist: PlaceHolder with marginal distributions (X, E, y).
        node_dist: distribution over number of nodes (callable .sample_n or Tensor).
        noise_schedule: PredefinedNoiseScheduleDiscrete instance.
        transition_model: transition model instance.
        Xdim_output, Edim_output: number of output classes.
        device: torch device.
        num_nodes: optional int or Tensor specifying number of nodes per graph.

    Returns:
        list of [atom_types (n,), edge_types (n, n)] pairs.
    """
    # Determine number of nodes per graph
    if num_nodes is None:
        if hasattr(node_dist, "sample_n"):
            n_nodes = node_dist.sample_n(n_samples, device)
        elif callable(node_dist):
            n_nodes = node_dist(n_samples).to(device)
        else:
            # node_dist is a Tensor probability distribution over node counts
            n_nodes = torch.multinomial(node_dist.float(), n_samples, replacement=True).to(device)
    elif isinstance(num_nodes, int):
        n_nodes = num_nodes * torch.ones(n_samples, device=device, dtype=torch.int)
    else:
        n_nodes = num_nodes.to(device)

    n_max = torch.max(n_nodes).item()

    # Build masks
    arange = torch.arange(n_max, device=device).unsqueeze(0).expand(n_samples, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    # Sample from limit distribution (pure noise)
    z_T = sample_discrete_feature_noise(limit_dist=limit_dist, node_mask=node_mask)
    X, E, y = z_T.X, z_T.E, z_T.y

    assert (E == torch.transpose(E, 1, 2)).all()

    # Iteratively denoise: p(z_s | z_t) for t = T, T-1, ..., 1
    for s_int in reversed(range(0, T)):
        s_array = s_int * torch.ones((n_samples, 1), device=device).type_as(y)
        t_array = s_array + 1
        s_norm = s_array / T
        t_norm = t_array / T

        sampled_s, discrete_sampled_s = sample_p_zs_given_zt(
            s_norm, t_norm, X, E, y, node_mask,
            model, noise_schedule, transition_model,
            Xdim_output, Edim_output,
        )
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

    # Collapse to discrete indices
    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

    # Build output list
    graph_list = []
    for i in range(n_samples):
        n_i = int(n_nodes[i].item())
        atom_types = X[i, :n_i].cpu()
        edge_types = E[i, :n_i, :n_i].cpu()
        graph_list.append([atom_types, edge_types])

    return graph_list


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== diffusion.py smoke test ===")

    T = 800
    dx_out, de_out = 5, 2  # NB101-like

    # 1. Noise schedule
    ns = PredefinedNoiseScheduleDiscrete("cosine", T)
    t_test = torch.tensor([[0.5]])
    beta = ns(t_normalized=t_test)
    alpha_bar = ns.get_alpha_bar(t_normalized=t_test)
    print(f"  t=0.5: beta={beta.item():.6f}, alpha_bar={alpha_bar.item():.6f}")
    assert 0 < beta.item() < 1, "beta out of range"
    assert 0 < alpha_bar.item() < 1, "alpha_bar out of range"
    print("  Noise schedule OK")

    # 2. Transition models
    x_marginals = torch.tensor([0.1, 0.1, 0.3, 0.3, 0.2])
    e_marginals = torch.tensor([0.7, 0.3])

    tm_uniform = DiscreteUniformTransition(dx_out, de_out, 0)
    tm_marginal = MarginalUniformTransition(x_marginals, e_marginals, 0)

    beta_t = torch.tensor([[0.01], [0.02]])  # (bs, 1) as noise_schedule returns
    Qt = tm_marginal.get_Qt(beta_t, "cpu")
    assert Qt.X.shape == (2, dx_out, dx_out), f"Qt.X shape: {Qt.X.shape}"
    assert (abs(Qt.X.sum(dim=-1) - 1) < 1e-5).all(), "Qt.X rows don't sum to 1"
    print("  Transition models OK")

    # 3. apply_noise
    bs, n = 4, 7
    X = F.one_hot(torch.randint(0, dx_out, (bs, n)), dx_out).float()
    # Build valid symmetric one-hot E: upper triangle -> symmetrize -> set diagonal to no-edge
    E_idx = torch.randint(0, de_out, (bs, n, n))
    E_idx = torch.triu(E_idx, diagonal=1)
    E_idx = E_idx + E_idx.transpose(1, 2)  # symmetric integer indices
    E = F.one_hot(E_idx, de_out).float()
    # Zero diagonal (no self-edges), set no-edge class
    diag_idx = torch.arange(n)
    E[:, diag_idx, diag_idx, :] = 0
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    node_mask[0, 5:] = False

    noisy = apply_noise(X, E, y, node_mask, ns, tm_marginal, T)
    assert noisy["X_t"].shape == X.shape, f"Noisy X shape: {noisy['X_t'].shape}"
    assert noisy["E_t"].shape == E.shape, f"Noisy E shape: {noisy['E_t'].shape}"
    assert noisy["t_int"].shape == (bs, 1)
    print("  apply_noise() OK")

    # 4. Single reverse step (needs a model — use a tiny one)
    from src.model import GraphTransformer

    input_dims = {"X": dx_out, "E": de_out, "y": 1}
    hidden_mlp_dims = {"X": 32, "E": 16, "y": 16}
    hidden_dims = {
        "dx": 32, "de": 16, "dy": 16, "n_head": 4,
        "dim_ffX": 32, "dim_ffE": 16, "dim_ffy": 16,
        "max_nodes": 15,
    }
    output_dims = {"X": dx_out, "E": de_out, "y": 0}

    tiny_model = GraphTransformer(
        n_layers=2, input_dims=input_dims,
        hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims,
        output_dims=output_dims,
    )
    tiny_model.eval()

    with torch.no_grad():
        s_norm = torch.tensor([[0.5 - 1/T]] * bs)
        t_norm = torch.tensor([[0.5]] * bs)
        out_oh, out_disc = sample_p_zs_given_zt(
            s_norm, t_norm,
            noisy["X_t"], noisy["E_t"], noisy["y_t"], node_mask,
            tiny_model, ns, tm_marginal, dx_out, de_out,
        )
    assert out_oh.X.shape == X.shape, f"Reverse X shape: {out_oh.X.shape}"
    assert out_oh.E.shape == E.shape, f"Reverse E shape: {out_oh.E.shape}"
    print("  sample_p_zs_given_zt() OK")

    # 5. Full sampling loop (tiny T for speed)
    tiny_T = 5
    tiny_ns = PredefinedNoiseScheduleDiscrete("cosine", tiny_T)
    limit_dist = PlaceHolder(X=x_marginals, E=e_marginals,
                              y=torch.ones(0))

    with torch.no_grad():
        graphs = sample_batch(
            model=tiny_model, n_samples=2, T=tiny_T,
            limit_dist=limit_dist,
            node_dist=torch.tensor([0, 0, 0, 0, 0.3, 0.4, 0.2, 0.1]),
            noise_schedule=tiny_ns, transition_model=tm_marginal,
            Xdim_output=dx_out, Edim_output=de_out, device="cpu",
        )
    assert len(graphs) == 2, f"Expected 2 graphs, got {len(graphs)}"
    for i, (atoms, edges) in enumerate(graphs):
        print(f"  Graph {i}: {atoms.shape[0]} nodes, edges {edges.shape}")
        assert atoms.dim() == 1
        assert edges.dim() == 2
        assert edges.shape[0] == edges.shape[1]
    print("  sample_batch() OK")

    print("All diffusion.py tests passed.")
