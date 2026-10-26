"""Reproduce every result and figure.  ``python experiments/run_all.py [--quick]``."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from flagdecode import (ToricCode, calibrated_prior, mlp_failures, mwpm_failures,  # noqa: E402
                        mwpm_weights, sample_noise, train_mlp)
from plots import make_figures  # noqa: E402

ETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def mwpm_variants(code, err, flags, p, pe, eta, only=None):
    """Failure indicators of the three matching decoders on one shared sample."""
    synd = code.syndrome(err)
    no_flags = np.zeros_like(flags)
    p_eff = (1 - pe) * p + 0.5 * pe
    ws = {
        "mwpm_ignore_flags": lambda: mwpm_weights(no_flags, p_eff),
        "mwpm_naive_flags": lambda: mwpm_weights(flags, p),
        "mwpm_calibrated": lambda: mwpm_weights(flags, calibrated_prior(p, pe, eta)),
    }
    out = {k: mwpm_failures(code, err, synd, w()) for k, w in ws.items()
           if only is None or k in only}
    return {k: v for k, v in out.items() if only is None or k in only}


def eta_sweep(cfg, seeds):
    """Experiment A: logical failure versus flag efficiency (L fixed)."""
    code = ToricCode(cfg["L_a"])
    rows = []
    for s in seeds:
        nets = {uf: train_mlp(code, cfg["pe_a"], [cfg["p_a"]], ETAS, seed=s, use_flags=uf,
                              steps=cfg["steps"]) for uf in (True, False)}
        rng = np.random.default_rng(1000 + s)
        for eta in ETAS:
            err, flags = sample_noise(code, cfg["p_a"], cfg["pe_a"], eta, cfg["shots_a"], rng)
            res = mwpm_variants(code, err, flags, cfg["p_a"], cfg["pe_a"], eta)
            res["mlp_with_flags"] = mlp_failures(nets[True], code, err, flags, eta, True)
            res["mlp_no_flags"] = mlp_failures(nets[False], code, err, flags, eta, False)
            for name, f in res.items():
                rows.append(dict(exp="eta_sweep", L=code.L, seed=s, eta=eta, decoder=name,
                                 pe=cfg["pe_a"], p=cfg["p_a"], fail=float(f.mean()),
                                 shots=cfg["shots_a"]))
        print(f"  eta sweep seed {s} done", flush=True)
    return pd.DataFrame(rows)


def erasure_scaling(cfg, seeds):
    """Experiment B: calibrated-MWPM logical failure versus erasure rate for L = 3, 5, 7."""
    rows = []
    for L in cfg["Ls_b"]:
        code = ToricCode(L)
        for s in seeds:
            rng = np.random.default_rng(2000 + 17 * L + s)
            for pe in cfg["pes_b"]:
                err, flags = sample_noise(code, cfg["p_b"], pe, cfg["eta_b"], cfg["shots_b"], rng)
                f = mwpm_variants(code, err, flags, cfg["p_b"], pe, cfg["eta_b"],
                                  only=("mwpm_naive_flags", "mwpm_calibrated"))
                for name in ("mwpm_naive_flags", "mwpm_calibrated"):
                    rows.append(dict(exp="scaling", L=L, seed=s, eta=cfg["eta_b"], decoder=name,
                                     pe=pe, p=cfg["p_b"], fail=float(f[name].mean()),
                                     shots=cfg["shots_b"]))
        print(f"  scaling L={L} done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        cfg = dict(L_a=3, pe_a=0.12, p_a=0.01, shots_a=200, steps=150, Ls_b=[3, 5],
                   pes_b=[0.06, 0.12], p_b=0.01, eta_b=0.5, shots_b=100)
        seeds, out, figd = [0, 1], ROOT / "results" / "quick", ROOT / "figures" / "quick"
    else:
        cfg = dict(L_a=5, pe_a=0.12, p_a=0.01, shots_a=1500, steps=3000, Ls_b=[3, 5, 7],
                   pes_b=[0.04, 0.06, 0.08, 0.12, 0.16], p_b=0.01, eta_b=0.5, shots_b=1200)
        seeds, out, figd = [0, 1, 2], ROOT / "results", ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    da = eta_sweep(cfg, seeds)
    db = erasure_scaling(cfg, seeds)
    da.to_csv(out / "eta_sweep.csv", index=False)
    db.to_csv(out / "erasure_scaling.csv", index=False)
    summ = (pd.concat([da, db]).groupby(["exp", "L", "eta", "pe", "decoder"])["fail"]
            .agg(["mean", "std"]).reset_index())
    summ.to_csv(out / "summary.csv", index=False)
    make_figures(da, db, figd)
    (out / "run_info.json").write_text(json.dumps(
        dict(config=cfg, seeds=seeds, runtime_s=round(time.time() - t0, 1)), indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
