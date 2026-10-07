from pathlib import Path

import torch

from app.cg_gnn_model import CGSchNetLike, HarmonicBondPrior, CGForceField


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

BONDED_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4)]

_prior_params = torch.load(
    MODEL_DIR / "prior_params.pt",
    map_location="cpu",
)

_prior = HarmonicBondPrior(
    BONDED_PAIRS,
    _prior_params["d0"],
    _prior_params["k"],
)

_correction = CGSchNetLike(
    n_beads=5,
    n_features=32,
    n_rbf=32,
    n_interactions=2,
    d_max=1.0,
)

_correction.load_state_dict(
    torch.load(
        MODEL_DIR / "cgnet_correction_state.pt",
        map_location="cpu",
    )
)
_correction.eval()

_model = CGForceField(_prior, _correction)
_model.eval()


def get_model() -> CGForceField:
    """Return the loaded coarse-grained force field."""
    return _model
