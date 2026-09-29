# ============================================================
# FASTAPI APPLICATION
# CG Force Field Energy Landscape API
# ============================================================

from fastapi import FastAPI, HTTPException

from app.schemas import DihedralInput, StructureInput, EnergyOutput, LandscapeOutput
from app.predict import predict_from_dihedrals, predict_from_positions, compute_landscape

# ============================================================
# INITIALIZE API
# ============================================================

app = FastAPI(
    title="CG Force Field Energy Landscape API",
    description=(
        "Serves a machine-learned coarse-grained (CG) force field for alanine "
        "dipeptide -- a harmonic bonded prior plus a SchNet-style graph neural "
        "network correction, trained by force matching against an all-atom "
        "reference trajectory. Predicts CG energy and forces for a given "
        "structure, or for a structure built from backbone phi/psi angles."
    ),
    version="1.0.0",
)

# ============================================================
# ROOT ENDPOINT
# ============================================================


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "CG Force Field Energy Landscape API is running."}


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================


@app.post("/predict/from_dihedrals", response_model=EnergyOutput)
def predict_dihedrals(input_data: DihedralInput):
    """
    Predict CG energy/forces for a structure built from backbone phi/psi
    angles (mean bond lengths and angles from the training trajectory).
    """
    try:
        return predict_from_dihedrals(phi=input_data.phi, psi=input_data.psi)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/from_structure", response_model=EnergyOutput)
def predict_structure(input_data: StructureInput):
    """Predict CG energy/forces for a user-supplied 5-bead structure."""
    try:
        return predict_from_positions(input_data.positions_nm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/landscape", response_model=LandscapeOutput)
def landscape(n_grid: int = 25):
    """
    Evaluate the learned CG energy on a regular (phi, psi) grid -- powers
    the energy-landscape heatmap in the Streamlit app.
    """
    if not (5 <= n_grid <= 60):
        raise HTTPException(status_code=400, detail="n_grid must be between 5 and 60")
    return compute_landscape(n_grid=n_grid)
