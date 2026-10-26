"""Figures: small multiples, violin+seeds, log-log scaling with fitted slopes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ACC, GREY, DARK = "#c4501b", "#9a9a9a", "#444444"
NAMES = {"mwpm_ignore_flags": "MWPM, flags ignored", "mwpm_naive_flags": "MWPM, naive flags",
         "mwpm_calibrated": "MWPM, calibrated flags", "mlp_with_flags": "MLP with flags",
         "mlp_no_flags": "MLP without flags"}
STYLE = {"mwpm_ignore_flags": (GREY, "o", ":"), "mwpm_naive_flags": (DARK, "s", "--"),
         "mwpm_calibrated": (ACC, "D", "-"), "mlp_with_flags": (DARK, "^", "-"),
         "mlp_no_flags": (GREY, "v", "-")}


def make_figures(da: pd.DataFrame, db: pd.DataFrame, out: Path) -> None:
    # 1. small multiples: failure vs eta, one panel per decoder family group
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, group, ttl in zip(axes, (["mwpm_ignore_flags", "mwpm_naive_flags", "mwpm_calibrated"],
                                     ["mlp_no_flags", "mlp_with_flags", "mwpm_calibrated"]),
                              ("Matching decoders", "Neural decoder vs best matching")):
        for d in group:
            g = da[da.decoder == d].groupby("eta")["fail"].agg(["mean", "std"])
            c, m, ls = STYLE[d]
            ax.errorbar(g.index, g["mean"], g["std"], color=c, marker=m, ls=ls, capsize=3,
                        label=NAMES[d])
        ax.set_title(ttl)
        ax.set_xlabel("Flag efficiency $\\eta$ (fraction of erasures flagged)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Logical failure probability")
    fig.suptitle("Decoder failure vs flag efficiency (mean $\\pm$ std over seeds)")
    fig.tight_layout()
    fig.savefig(out / "fig1_eta_small_multiples.png", dpi=200)
    plt.close(fig)

    # 2. violin: per-seed failure at eta = 0.5 for every decoder
    sub = da[np.isclose(da.eta, 0.5)]
    order = list(NAMES)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    data = [sub[sub.decoder == d]["fail"].values for d in order]
    parts = ax.violinplot(data, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(GREY)
        b.set_alpha(0.4)
    for i, v in enumerate(data, 1):
        ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=25,
                   color=ACC if order[i - 1] == "mwpm_calibrated" else DARK, zorder=3)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([NAMES[d].replace(", ", "\n").replace(" with", "\nwith").replace(" without", "\nwithout")
                        for d in order], fontsize=8)
    ax.set_ylabel("Logical failure probability")
    ax.set_title("Per-seed spread at flag efficiency $\\eta$ = 0.5")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig2_violin_eta05.png", dpi=200)
    plt.close(fig)

    # 3. log-log: failure vs erasure rate for each L, fitted slopes
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, d in zip(axes, ("mwpm_naive_flags", "mwpm_calibrated")):
        for L, shade in zip(sorted(db.L.unique()), (GREY, DARK, ACC)):
            g = db[(db.decoder == d) & (db.L == L)].groupby("pe")["fail"].agg(["mean", "std"])
            g = g[g["mean"] > 0]
            ax.errorbar(g.index, g["mean"], g["std"], color=shade, marker="o", capsize=3,
                        label=f"L = {L}")
            if len(g) >= 3:
                lo = g.iloc[: min(3, len(g))]
                s = np.polyfit(np.log(lo.index), np.log(lo["mean"]), 1)[0]
                ax.annotate(f"slope {s:.1f}", (lo.index[1], lo["mean"].iloc[1]),
                            textcoords="offset points", xytext=(6, -12), fontsize=8, color=shade)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(NAMES[d] + " ($\\eta$ = 0.5)")
        ax.set_xlabel("Erasure probability per qubit $p_e$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Logical failure probability")
    fig.suptitle("Scaling with erasure rate (slope fitted on the three lowest $p_e$)")
    fig.tight_layout()
    fig.savefig(out / "fig3_loglog_erasure_scaling.png", dpi=200)
    plt.close(fig)
