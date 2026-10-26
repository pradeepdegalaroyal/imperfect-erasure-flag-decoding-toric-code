"""Minimum-weight perfect-matching (MWPM) decoders with flag-dependent edge weights."""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .lattice import ToricCode

EPS = 1e-9


def calibrated_prior(p: float, pe: float, eta: float) -> float:
    """P(X error | qubit not flagged), accounting for erasures that escaped the flag."""
    hidden = pe * (1.0 - eta)
    return ((1 - pe) * p + 0.5 * hidden) / ((1 - pe) + hidden)


def mwpm_weights(flags: np.ndarray, q_unflagged: float) -> np.ndarray:
    """Edge weights: flagged edges ~0 (prior 1/2), others log((1-q)/q)."""
    q = float(np.clip(q_unflagged, 1e-6, 0.499))
    w = np.full(flags.shape, np.log((1 - q) / q))
    w[flags.astype(bool)] = EPS
    return w


def _decode_one(code: ToricCode, synd: np.ndarray, w: np.ndarray) -> np.ndarray:
    defects = np.flatnonzero(synd)
    corr = np.zeros(code.n_edges, dtype=np.uint8)
    if len(defects) == 0:
        return corr
    a, b = code.ends[:, 0], code.ends[:, 1]
    wt = np.maximum(w, EPS)
    g = csr_matrix((np.r_[wt, wt], (np.r_[a, b], np.r_[b, a])), shape=(code.n_nodes,) * 2)
    dist, pred = dijkstra(g, indices=defects, return_predecessors=True)
    G = nx.Graph()
    for x in range(len(defects)):
        for y in range(x + 1, len(defects)):
            G.add_edge(x, y, weight=-dist[x, defects[y]])
    for x, y in nx.max_weight_matching(G, maxcardinality=True):
        node = int(defects[y])
        while node != defects[x]:
            prev = int(pred[x, node])
            corr[code.edge_of[(prev, node)]] ^= 1
            node = prev
    return corr


def mwpm_decode_batch(code: ToricCode, synd: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Decode a batch; returns corrections (shots, n_edges)."""
    return np.stack([_decode_one(code, s, w) for s, w in zip(synd, weights)])


def mwpm_failures(code: ToricCode, err: np.ndarray, synd: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
    """Boolean logical-failure indicator per shot."""
    corr = mwpm_decode_batch(code, synd, weights)
    return code.logical(err ^ corr).any(axis=1)
