# Research gap: decoding when erasure flags are imperfect

## State of the art (from the supplied literature)

Surface-code memories now operate below threshold (Zobrist et al., 2025, doi:10.1038/s41586-024-08449-y; Acharya et al., 2023, doi:10.1038/s41586-022-05434-1), and logical processors have been run on neutral-atom arrays (Bluvstein et al., 2024, doi:10.1038/s41586-023-06927-3). Mid-circuit erasure conversion turns a large share of physical errors into *flagged* erasures (Ma et al., 2023, doi:10.1038/s41586-023-06438-1), which a decoder can exploit because the location of the fault is known. Recent code work targets low overhead (Bravyi et al., 2024, doi:10.1038/s41586-024-07107-7; Xu et al., 2024, doi:10.1038/s41567-024-02479-z), and recent decoder work explores learned decoding (Joo, 2026, doi:10.1140/epjqt/s40507-026-00492-0) and erasure recovery with fewer measurements (Matsumoto, 2027, doi:10.1007/s40314-026-03881-4).

Note on the brief: the supplied literature file labels all 30 papers with the single role "other", so the table below groups them by function (experiment, code, decoder, survey, adjacent) instead of by the four standard roles.

## Related work

| Paper | Year | Function | What it does | What it does NOT cover |
|---|---|---|---|---|
| Zobrist et al., doi:10.1038/s41586-024-08449-y | 2025 | experiment | Below-threshold surface-code memory | Flag-assisted decoding; no erasure flags |
| Acharya et al., doi:10.1038/s41586-022-05434-1 | 2023 | experiment | Scaling a surface-code logical qubit | Heralded erasures |
| Bluvstein et al., doi:10.1038/s41586-023-06927-3 | 2024 | experiment | Logical processor on atom arrays | Decoder sensitivity to flag quality |
| Ma et al., doi:10.1038/s41586-023-06438-1 | 2023 | experiment | Mid-circuit erasure conversion in an atomic qubit | How a finite conversion efficiency changes the optimal decoder weights |
| Sivak et al., doi:10.1038/s41586-023-05782-6 | 2023 | experiment | Real-time quantum error correction (QEC) beyond break-even | Erasure-aware decoding |
| Ni et al., doi:10.1038/s41586-023-05784-4 | 2023 | experiment | Break-even with a discrete-variable logical qubit | Decoder design |
| Bravyi et al., doi:10.1038/s41586-024-07107-7 | 2024 | code | High-threshold low-overhead memory | Erasure-biased noise |
| Xu et al., doi:10.1038/s41567-024-02479-z | 2024 | code | Constant-overhead fault-tolerant (FT) computation on atom arrays | Imperfect flag information |
| Matsumoto, doi:10.1007/s40314-026-03881-4 | 2027 | decoder theory | Fewer measurements for erasure correction | Flag efficiency below one; mixed erasure and Pauli noise |
| Joo, doi:10.1140/epjqt/s40507-026-00492-0 | 2026 | decoder | Transformer surface-code decoder | Flag inputs; calibrated classical baseline under noise mismatch |
| Wang et al., doi:10.15302/frontphys.2027.023301 | 2027 | survey | Variational quantum algorithms toward fault tolerance | Decoding |
| Caleffi et al., doi:10.1016/j.comnet.2024.110672 | 2024 | survey | Distributed quantum computing | Decoding |
| Bartolucci et al., doi:10.1038/s41467-023-36493-1 | 2023 | adjacent | Fusion-based photonic computation | Erasure-weighted decoding study |

## Gap

Erasure conversion is usually analysed with perfect flags (an erased qubit is always known to be erased). In hardware the conversion has finite efficiency: some erasures escape the flag and look like ordinary, high-rate Pauli errors. A matching decoder that trusts the flags and assigns the rest the nominal Pauli prior mis-weights exactly those hidden erasures. It is open how much this costs, whether a one-line recalibration of the unflagged prior recovers it, and whether a learned decoder given the flags and the efficiency does better than the calibrated classical baseline (learned decoders are typically compared with an uncalibrated matcher).

## Hypothesis (falsifiable)

At flag efficiency eta = 0.5 on a toric code with distance L = 5, 12% erasure and 1% Pauli noise, (H1) naive-flag matching has a logical failure probability that is higher than calibrated-flag matching by more than the seed standard deviation, and (H2) a small learned decoder with flag inputs does not beat calibrated matching by more than the seed standard deviation. Success metric: logical failure probability, mean and standard deviation over 3 seeds. H1 or H2 failing is reported as a negative result.

## Novelty (one sentence)

A controlled simulation that isolates the cost of *imperfect* erasure flags for matching decoding and benchmarks a flag-aware neural decoder against a prior-calibrated matcher rather than an uncalibrated one.

Scale: simulation only, code-capacity X-error model, toric code L = 3-7; no circuit-level noise, no repeated rounds.
