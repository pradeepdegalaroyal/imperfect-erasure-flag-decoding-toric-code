import numpy as np

from flagdecode import ToricCode, calibrated_prior, mwpm_decode_batch, mwpm_weights, sample_noise


def test_syndrome_even():
    code = ToricCode(3)
    err, _ = sample_noise(code, 0.1, 0.1, 1.0, 50, np.random.default_rng(0))
    assert (code.syndrome(err).sum(axis=1) % 2 == 0).all()


def test_reference_correction_matches_syndrome():
    code = ToricCode(3)
    err, _ = sample_noise(code, 0.2, 0.0, 1.0, 50, np.random.default_rng(1))
    s = code.syndrome(err)
    ref = (s.astype(int) @ code.R % 2).astype(np.uint8)
    assert (code.syndrome(ref) == s).all()


def test_mwpm_fixes_single_errors():
    code = ToricCode(5)
    for e in range(code.n_edges):
        err = np.zeros((1, code.n_edges), dtype=np.uint8)
        err[0, e] = 1
        s = code.syndrome(err)
        corr = mwpm_decode_batch(code, s, mwpm_weights(np.zeros_like(err), 0.05))
        assert not code.syndrome(err ^ corr).any()
        assert not code.logical(err ^ corr).any()


def test_calibrated_prior_limits():
    assert abs(calibrated_prior(0.03, 0.1, 1.0) - 0.03) < 1e-12
    assert calibrated_prior(0.03, 0.1, 0.0) > 0.03
