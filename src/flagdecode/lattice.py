"""Toric code on an L x L torus, X-error (code-capacity) model with flagged erasures.

Qubits are edges.  X errors flip the two adjacent plaquette checks, so errors live on the
dual lattice: nodes are plaquettes, edges are qubits.  Edge ``h(i,j)`` joins plaquettes
(i-1,j)-(i,j); edge ``v(i,j)`` joins (i,j-1)-(i,j).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToricCode:
    """Dual-lattice description of the L x L toric code (L >= 3)."""

    L: int

    def __post_init__(self) -> None:
        L = self.L
        self.n_nodes = L * L
        self.n_edges = 2 * L * L
        ends = []
        for i in range(L):
            for j in range(L):  # h edges: index i*L+j
                ends.append((self._node(i - 1, j), self._node(i, j)))
        for i in range(L):
            for j in range(L):  # v edges: index L*L + i*L+j
                ends.append((self._node(i, j - 1), self._node(i, j)))
        self.ends = np.array(ends)
        H = np.zeros((self.n_edges, self.n_nodes), dtype=np.uint8)
        for e, (a, b) in enumerate(ends):
            H[e, a] ^= 1
            H[e, b] ^= 1
        self.H = H  # error (edges) @ H = syndrome (nodes)
        # logical cuts: parity of X on h(0, :) and on v(:, 0)
        C = np.zeros((self.n_edges, 2), dtype=np.uint8)
        for j in range(L):
            C[j, 0] = 1
        for i in range(L):
            C[L * L + i * L, 1] = 1
        self.cut = C
        self.edge_of: dict[tuple[int, int], int] = {}
        for e, (a, b) in enumerate(ends):
            self.edge_of[(int(a), int(b))] = e
            self.edge_of[(int(b), int(a))] = e
        # reference correction: tree path to node 0 for every node
        R = np.zeros((self.n_nodes, self.n_edges), dtype=np.uint8)
        parent: dict[int, tuple[int, int] | None] = {0: None}
        order = [0]
        for u in order:
            for e, (a, b) in enumerate(ends):
                if u in (a, b):
                    w = int(b) if u == a else int(a)
                    if w not in parent:
                        parent[w] = (u, e)
                        order.append(w)
        for v in range(1, self.n_nodes):
            x = v
            while parent[x] is not None:
                u, e = parent[x]  # type: ignore[misc]
                R[v, e] ^= 1
                x = u
        self.R = R
        self.RC = (R.astype(int) @ C.astype(int) % 2).astype(np.uint8)

    def _node(self, i: int, j: int) -> int:
        return (i % self.L) * self.L + (j % self.L)

    def syndrome(self, err: np.ndarray) -> np.ndarray:
        """Plaquette syndrome of a batch of X-error vectors (shots, n_edges)."""
        return (err.astype(np.int64) @ self.H % 2).astype(np.uint8)

    def logical(self, err: np.ndarray) -> np.ndarray:
        """Cut parities (shots, 2) of an error vector or closed chain."""
        return (err.astype(np.int64) @ self.cut % 2).astype(np.uint8)

    def reference_class(self, synd: np.ndarray) -> np.ndarray:
        """Cut parity of the reference correction for each syndrome (shots, 2)."""
        return (synd.astype(np.int64) @ self.RC % 2).astype(np.uint8)


def sample_noise(code: ToricCode, p: float, pe: float, eta: float, shots: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample errors and flags.

    Each qubit is erased with probability ``pe`` (then X with probability 1/2) and
    otherwise suffers X with probability ``p``.  An erasure is flagged with
    probability ``eta`` (flag efficiency); flags are never raised on unerased qubits.
    Returns ``(error, flags)``, both uint8 arrays (shots, n_edges).
    """
    n = code.n_edges
    erased = rng.random((shots, n)) < pe
    xe = rng.random((shots, n)) < 0.5
    xp = rng.random((shots, n)) < p
    err = np.where(erased, xe, xp).astype(np.uint8)
    flags = (erased & (rng.random((shots, n)) < eta)).astype(np.uint8)
    return err, flags
