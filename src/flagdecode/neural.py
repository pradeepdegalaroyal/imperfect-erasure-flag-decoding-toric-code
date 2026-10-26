"""Small multilayer-perceptron decoder taking syndrome, flags and flag efficiency."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .lattice import ToricCode, sample_noise

torch.set_num_threads(6)


def _features(synd: np.ndarray, flags: np.ndarray, eta: float, use_flags: bool) -> torch.Tensor:
    cols = [synd.astype(np.float32)]
    if use_flags:
        cols.append(flags.astype(np.float32))
        cols.append(np.full((len(synd), 1), eta, dtype=np.float32))
    return torch.from_numpy(np.concatenate(cols, axis=1))


def _target(code: ToricCode, err: np.ndarray, synd: np.ndarray) -> torch.Tensor:
    """Cut parity of (error + reference correction): a function of the true error class."""
    z = code.logical(err) ^ code.reference_class(synd)
    return torch.from_numpy(z.astype(np.float32))


def train_mlp(code: ToricCode, pe: float, ps: list[float], etas: list[float], seed: int,
              use_flags: bool = True, steps: int = 3000, batch: int = 512,
              hidden: int = 256) -> nn.Module:
    """Train on online-sampled noise drawn uniformly from ``ps`` x ``etas``."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    din = code.n_nodes + (code.n_edges + 1 if use_flags else 0)
    net = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 2))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        p, eta = ps[rng.integers(len(ps))], etas[rng.integers(len(etas))]
        err, flags = sample_noise(code, p, pe, eta, batch, rng)
        synd = code.syndrome(err)
        loss = lossf(net(_features(synd, flags, eta, use_flags)), _target(code, err, synd))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net


@torch.no_grad()
def mlp_failures(net: nn.Module, code: ToricCode, err: np.ndarray, flags: np.ndarray,
                 eta: float, use_flags: bool = True) -> np.ndarray:
    """Boolean logical-failure indicator per shot for the MLP decoder."""
    synd = code.syndrome(err)
    pred = (net(_features(synd, flags, eta, use_flags)) > 0).numpy().astype(np.uint8)
    return (pred != _target(code, err, synd).numpy().astype(np.uint8)).any(axis=1)
