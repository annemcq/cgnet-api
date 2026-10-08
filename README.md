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

## Live demo

**Try the interactive app:** https://annemcq-cgnet-api-streamlit-appapp-obnfhv.streamlit.app

Explore the learned coarse-grained energy landscape and evaluate the model at different pseudo-phi/psi conformations directly in your browser.

## Interactive interface

![Streamlit interface](images/streamlit_demo.png)

The Streamlit interface sends requests to a **separately hosted FastAPI backend** configured through the Streamlit secret `CGNET_API_URL` (or the environment variable of the same name). The deployment provider and public backend URL are not recorded in this repository; to document the exact host, check the deployed Streamlit app's Secrets settings for `CGNET_API_URL`. Locally, the default is `http://127.0.0.1:8000`. The backend loads the packaged model weights from `models/` and returns predicted energies and forces; the Streamlit app does not contain a separate copy of the model.

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

## Repository structure

```text
cgnet-api/
├── .github/
│   └── workflows/
│       └── tests.yml
├── images/
│   └── streamlit_demo.png
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
├── LICENSE
├── pytest.ini
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

The FastAPI backend can also be run in a Docker container. The Dockerfile installs the **CPU-only PyTorch wheel** before the remaining requirements to avoid bundling unnecessary CUDA dependencies:

```bash
docker build -t cgnet-api .
docker run -p 8000:8000 cgnet-api
```

The Streamlit frontend is run separately and can be pointed to the API using
the `CGNET_API_URL` environment variable.

## Validation and limitations

The packaged API serves the same trained force field validated in `mlcg-gnn-alanine`. The companion project contains the training procedure, coarse-graining workflow and molecular-dynamics validation, including the limitations of the learned conformational distribution.

This application is therefore a deployment and exploration layer around that model, not a new validation of the force field. The displayed energy landscape should be treated as a visualization of the learned model rather than as a quantitatively validated free-energy surface, especially in poorly sampled regions. See `mlcg-gnn-alanine` for the underlying validation and quantitative conformational-distribution comparison.

## Tests

The test suite checks the API endpoints and input validation, the dimensions
of the generated energy landscape, and whether the internal-coordinate
reconstruction recovers the requested phi and psi angles.

```bash
pytest tests/
```

## Related project

The model training, coarse-graining procedure and molecular simulation are
contained in [`mlcg-gnn-alanine`](https://github.com/annemcq/mlcg-gnn-alanine).

## Technologies

Python, PyTorch, FastAPI, Pydantic, Streamlit, NumPy, Matplotlib, Docker and
pytest.

## License

MIT
