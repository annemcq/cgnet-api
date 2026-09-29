"""
Tests for the FastAPI app. Use FastAPI's TestClient (in-process, no server
needed) so these run fast and don't depend on a port being free.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_predict_from_dihedrals_valid_input():
    response = client.post("/predict/from_dihedrals", json={"phi": -150.0, "psi": 150.0})
    assert response.status_code == 200
    data = response.json()
    assert data["phi"] == -150.0
    assert data["psi"] == 150.0
    assert isinstance(data["energy_kj_mol"], float)
    assert len(data["force_magnitude_kj_mol_nm"]) == 5


def test_predict_from_dihedrals_rejects_out_of_range():
    response = client.post("/predict/from_dihedrals", json={"phi": 200.0, "psi": 0.0})
    assert response.status_code == 422  # pydantic validation error


def test_predict_from_structure_valid_input():
    positions = [[0.0, 0.0, 0.0], [0.19, 0.0, 0.0], [0.35, 0.1, 0.0],
                 [0.55, 0.05, 0.1], [0.7, 0.15, 0.1]]
    response = client.post("/predict/from_structure", json={"positions_nm": positions})
    assert response.status_code == 200
    data = response.json()
    assert data["phi"] is None and data["psi"] is None
    assert len(data["force_magnitude_kj_mol_nm"]) == 5


def test_predict_from_structure_rejects_wrong_shape():
    response = client.post("/predict/from_structure", json={"positions_nm": [[0.0, 0.0, 0.0]]})
    assert response.status_code == 422  # fails min_length=5 schema validation


def test_landscape_shape_matches_grid_size():
    response = client.get("/landscape", params={"n_grid": 8})
    assert response.status_code == 200
    data = response.json()
    assert len(data["phi_grid"]) == 8
    assert len(data["psi_grid"]) == 8
    assert len(data["energy_kj_mol"]) == 8
    assert all(len(row) == 8 for row in data["energy_kj_mol"])


def test_landscape_rejects_absurd_grid_size():
    response = client.get("/landscape", params={"n_grid": 500})
    assert response.status_code == 400


def test_dihedral_reconstruction_recovers_requested_angles():
    """The NeRF structure builder should reproduce the requested phi/psi exactly."""
    import numpy as np
    from app.structure_builder import build_structure_from_dihedrals

    def dihedral(p0, p1, p2, p3):
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))

    for target_phi, target_psi in [(-150, 150), (-70, -40), (60, 45), (0, 0)]:
        coords = build_structure_from_dihedrals(target_phi, target_psi)
        phi = dihedral(coords[0], coords[1], coords[2], coords[3])
        psi = dihedral(coords[1], coords[2], coords[3], coords[4])
        assert abs(phi - target_phi) < 1e-3
        assert abs(psi - target_psi) < 1e-3
