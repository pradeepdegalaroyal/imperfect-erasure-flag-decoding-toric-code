# Method

**Noise model (code capacity, X errors).** Each qubit of an L x L toric code is erased with probability p_e; an erased qubit carries an X error with probability 1/2. Otherwise it suffers an X error with probability p. An erasure is flagged with probability eta (flag efficiency); unerased qubits are never flagged. The decoder sees the plaquette syndrome and the flags.

**Decoders.**
1. Minimum-weight perfect matching (MWPM) is the standard classical baseline (Dijkstra distances on the dual torus, exact matching via networkx). Three weightings: *flags ignored* (uniform prior p_eff = (1-p_e)p + p_e/2), *naive flags* (flagged edges weight ~0, unflagged edges use the nominal prior p), and *calibrated flags* (as naive, but the unflagged prior is P(X | not flagged) = ((1-p_e)p + p_e(1-eta)/2) / ((1-p_e) + p_e(1-eta)), which accounts for hidden erasures).
2. Multilayer perceptron (MLP) neural decoder: two hidden layers of 256 units; input is the syndrome, the flags and eta; output is the cut parity of (error + a fixed reference correction), so a correct prediction fixes the logical class. Trained online on 3000 batches of 512 samples at p_e fixed and eta drawn from {0, 0.25, 0.5, 0.75, 1}.
3. Ablation: the same MLP without flag and eta inputs, and MWPM with flags ignored.

**Metric.** Logical failure probability (a logical class flips after correction), mean and standard deviation over seeds. Each seed retrains the MLP and uses a fresh test sample.

**Experiments.**
- A (flag-efficiency sweep): L = 5, p_e = 0.12, p = 0.01, eta in {0, 0.25, 0.5, 0.75, 1}, 1500 test shots per cell, 3 seeds.
- B (erasure-rate scaling; robustness sweep over noise strength): L in {3, 5, 7}, p = 0.01, eta = 0.5, p_e in {0.04, 0.06, 0.08, 0.12, 0.16}, 1200 shots per cell, 3 seeds, naive vs calibrated MWPM, log-log slope fitted on the three lowest p_e.

**Reproduce.** `python experiments/run_all.py` (full) or `--quick` (smoke test).
