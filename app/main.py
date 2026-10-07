from fastapi import FastAPI, HTTPException

from app.schemas import DihedralInput, StructureInput, EnergyOutput, LandscapeOutput
from app.predict import predict_from_dihedrals, predict_from_positions, compute_landscape


app = FastAPI(
    title="CG Force Field Energy Landscape API",
    description=(
        "Serves a machine-learned coarse-grained force field for alanine "
        "dipeptide: a harmonic bonded prior plus a SchNet-style graph neural "
        "network correction trained by force matching."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "CG Force Field Energy Landscape API is running."}


@app.post("/predict/from_dihedrals", response_model=EnergyOutput)
def predict_dihedrals(input_data: DihedralInput):
    """Evaluate a structure reconstructed from pseudo-phi/psi angles."""
    try:
        return predict_from_dihedrals(phi=input_data.phi, psi=input_data.psi)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/from_structure", response_model=EnergyOutput)
def predict_structure(input_data: StructureInput):
    """Evaluate a user-supplied five-bead structure."""
    try:
        return predict_from_positions(input_data.positions_nm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/landscape", response_model=LandscapeOutput)
def landscape(n_grid: int = 25):
    """Evaluate the CG energy on a regular pseudo-phi/psi grid."""
    if not (5 <= n_grid <= 60):
        raise HTTPException(status_code=400, detail="n_grid must be between 5 and 60")
    return compute_landscape(n_grid=n_grid)
