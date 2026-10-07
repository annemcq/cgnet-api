import numpy as np
import torch

from app.model_loader import get_model
from app.structure_builder import build_structure_from_dihedrals


def predict_from_dihedrals(phi: float, psi: float) -> dict:
    """Reconstruct a five-bead structure and evaluate the CG force field."""
    coords = build_structure_from_dihedrals(phi, psi)
    return _predict_from_coords(coords, phi=phi, psi=psi)


def predict_from_positions(positions_nm) -> dict:
    """Evaluate the CG force field on a user-supplied five-bead structure."""
    coords = np.asarray(positions_nm, dtype=np.float64)

    if coords.shape != (5, 3):
        raise ValueError(
            f"expected 5 positions of 3 coordinates each, got shape {coords.shape}"
        )

    return _predict_from_coords(coords, phi=None, psi=None)


def _predict_from_coords(coords: np.ndarray, phi, psi) -> dict:
    model = get_model()

    pos_t = torch.as_tensor(
        coords,
        dtype=torch.float32,
    ).unsqueeze(0)

    energy, forces = model(pos_t)
    force_mag = torch.norm(forces[0], dim=-1).tolist()

    return {
        "phi": phi,
        "psi": psi,
        "energy_kj_mol": round(float(energy.item()), 3),
        "force_magnitude_kj_mol_nm": [
            round(f, 2) for f in force_mag
        ],
        "model": "Harmonic bond prior + SchNet-style GNN correction",
    }


def compute_landscape(n_grid: int = 25) -> dict:
    """Evaluate the model energy on a regular pseudo-phi/psi grid."""
    phi_grid = np.linspace(-180, 180, n_grid)
    psi_grid = np.linspace(-180, 180, n_grid)

    model = get_model()
    energies = np.zeros((n_grid, n_grid))

    for i, phi in enumerate(phi_grid):
        coords_batch = np.stack(
            [
                build_structure_from_dihedrals(phi, psi)
                for psi in psi_grid
            ]
        )

        pos_t = torch.as_tensor(
            coords_batch,
            dtype=torch.float32,
        )

        energy, _ = model(pos_t)
        energies[i, :] = energy.detach().numpy()

    return {
        "phi_grid": phi_grid.tolist(),
        "psi_grid": psi_grid.tolist(),
        "energy_kj_mol": energies.tolist(),
    }
