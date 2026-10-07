# CG Force Field API for Alanine Dipeptide

[![Tests](https://github.com/annemcq/cgnet-api/actions/workflows/tests.yml/badge.svg)](https://github.com/annemcq/cgnet-api/actions/workflows/tests.yml)

This project packages the coarse-grained force field developed in my
`mlcg-gnn-alanine` project as a small prediction API and interactive
visualization.

The underlying model combines a harmonic bonded prior with a SchNet-style
graph neural network correction trained by force matching on an atomistic
trajectory of alanine dipeptide.

The API allows the model to be evaluated either from five coarse-grained
bead coordinates or from a pair of backbone pseudo-dihedral angles. A
Streamlit interface provides an interactive view of the resulting energy
landscape.

## From dihedral angles to model predictions

The trained model expects the Cartesian coordinates of five coarse-grained
beads. For interactive use, however, it is more convenient to describe a
conformation using the two backbone pseudo-dihedral angles, phi and psi.

The application therefore reconstructs a five-bead structure from these
angles using a NeRF-style internal-coordinate construction. Bond lengths and
bond angles are fixed to mean values obtained from the reference trajectory,
while phi and psi are supplied by the user.

The resulting structure is then passed to the same force field used in the
original coarse-graining project:

```text
(phi, psi)
    ↓
5-bead structure reconstruction
    ↓
harmonic bonded prior + GNN correction
    ↓
energy and forces
```

A lower-level endpoint is also available for evaluating arbitrary five-bead
Cartesian structures directly.

## API

The FastAPI application provides three main endpoints.

### Prediction from dihedral angles

```text
POST /predict/from_dihedrals
```

Example input:

```json
{
  "phi": -150.0,
  "psi": 150.0
}
```

Example output:

```json
{
  "phi": -150.0,
  "psi": 150.0,
  "energy_kj_mol": -21.696,
  "force_magnitude_kj_mol_nm": [544.17, 470.47, 1048.39, 197.51, 1048.93],
  "model": "Harmonic bond prior + SchNet-style GNN correction"
}
```

### Prediction from Cartesian coordinates

```text
POST /predict/from_structure
```

This endpoint accepts the coordinates of the five CG beads directly.

### Energy landscape

```text
GET /landscape?n_grid=35
```

The endpoint evaluates the model over a regular phi/psi grid and is used by
the Streamlit interface to generate the energy-landscape heatmap.

## Interactive interface

The Streamlit app allows phi and psi to be varied with sliders and displays
the corresponding point on the model-predicted energy landscape.

It also reports the predicted energy and force magnitude for each of the five
CG beads.

## Repository structure

```text
cgnet-api/
├── app/
│   ├── main.py
│   ├── model_loader.py
│   ├── structure_builder.py
│   ├── predict.py
│   ├── cg_gnn_model.py
│   └── schemas.py
├── models/
│   ├── cgnet_correction_state.pt
│   ├── prior_params.pt
│   └── mean_bond_angles_deg.npy
├── streamlit_app/
│   └── app.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

The interactive FastAPI documentation is then available at:

```text
http://127.0.0.1:8000/docs
```

Start the Streamlit interface in a second terminal:

```bash
streamlit run streamlit_app/app.py
```

Run the tests with:

```bash
pytest tests/
```

## Docker

The FastAPI backend can also be run in a Docker container:

```bash
docker build -t cgnet-api .
docker run -p 8000:8000 cgnet-api
```

The Streamlit frontend is run separately and can be pointed to the API using
the `CGNET_API_URL` environment variable.

## Validation and limitations

The underlying force field was trained and validated in the companion
`mlcg-gnn-alanine` project.

In short CG simulations, the model samples the main conformational regions of
the reference trajectory in approximately the same areas of pseudo-phi/psi
space. The relative populations and overall distribution are not reproduced
well, however, and the CG trajectory also visits regions that are sparsely
populated in the reference data.

The energy landscape exposed by this application should therefore be treated
as a visualization of the learned model rather than as a quantitatively
validated free-energy surface. Predictions in poorly sampled regions of the
training trajectory are particularly uncertain.

## Tests

The test suite checks the API endpoints and input validation, the dimensions
of the generated energy landscape, and whether the internal-coordinate
reconstruction recovers the requested phi and psi angles.

```bash
pytest tests/
```

## Related project

The model training, coarse-graining procedure and molecular simulation are
contained in `mlcg-gnn-alanine`.

## Technologies

Python, PyTorch, FastAPI, Pydantic, Streamlit, NumPy, Matplotlib, Docker and
pytest.

## License

MIT
