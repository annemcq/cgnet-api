# ============================================================
# MODEL LOADER
# CG Force Field Energy Landscape API
# ============================================================

import torch
from pathlib import Path

from app.cg_gnn_model import CGSchNetLike, HarmonicBondPrior, CGForceField

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

BONDED_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4)]

print("\nLoading CG force field (harmonic prior + GNN correction)...\n")

_prior_params = torch.load(MODEL_DIR / "prior_params.pt", map_location="cpu")
_prior = HarmonicBondPrior(BONDED_PAIRS, _prior_params["d0"], _prior_params["k"])

_correction = CGSchNetLike(n_beads=5, n_features=32, n_rbf=32, n_interactions=2, d_max=1.0)
_correction.load_state_dict(torch.load(MODEL_DIR / "cgnet_correction_state.pt", map_location="cpu"))
_correction.eval()

_model = CGForceField(_prior, _correction)
_model.eval()

print("Model loaded successfully.")


def get_model() -> CGForceField:
    """Return the loaded CG force field (prior + GNN correction)."""
    return _model
