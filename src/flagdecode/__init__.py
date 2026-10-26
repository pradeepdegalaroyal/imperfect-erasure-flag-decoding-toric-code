"""Flag-aware decoding of the toric code under imperfectly flagged erasure noise."""
from .lattice import ToricCode, sample_noise
from .decoders import mwpm_decode_batch, mwpm_failures, mwpm_weights, calibrated_prior
from .neural import train_mlp, mlp_failures

__all__ = ["ToricCode", "sample_noise", "mwpm_decode_batch", "mwpm_failures", "mwpm_weights",
           "calibrated_prior", "train_mlp", "mlp_failures"]
__version__ = "0.1.0"
