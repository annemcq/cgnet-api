# Deploying a Coarse-Grained Force Field as an Interactive API

[![Tests](https://github.com/annemcq/cgnet-api/actions/workflows/tests.yml/badge.svg)](https://github.com/annemcq/cgnet-api/actions/workflows/tests.yml)

This project packages a trained machine-learned coarse-grained (CG) force field —
developed in the companion `mlcg-gnn-alanine` project — as a deployable prediction
service with both a REST API and an interactive web interface.

The goal is not to build a more complex predictive model, but to demonstrate how a
trained scientific ML model (a harmonic bonded prior + SchNet-style GNN correction,
trained by force matching) can be turned into a clean, modular, deployable application.
Furthermore, it aims to make the model something a non-expert can explore interactively,
rather than a black box whose only output is a single numeric prediction.

---

## Project goals

This project demonstrates:

- deployment-oriented machine learning engineering, applied to a physics-based model
- API development using FastAPI
- reconstructing 3D structure from internal coordinates (bond lengths/angles/dihedrals)
  for user-friendly input, instead of requiring raw 3D coordinates
- an interactive energy-landscape visualization using Streamlit
- reusable model packaging (the same trained weights as the research project, unmodified)

Users can pick backbone **phi/psi** angles and see the model's predicted coarse-grained
energy — both as a single number and as a full 2D energy-landscape heatmap — through
both a REST API and an interactive web interface.

---

## Why this design

What makes a scientific model meaningful is not just its callability but how useful it is.
A given user does not have intuition for "5 beads at these 15 raw nm coordinates", but
backbone phi/psi angles are the standard, recognizable coordinate for describing peptide
conformations.

Thus, the API reconstructs a plausible 5-bead structure from (phi, psi) via NeRF (Natural
Extension Reference Frame — the standard chain-building algorithm from structural
biology), holding bond lengths and bond angles fixed at the training data's own mean
values, and lets the model evaluate energy/forces at that structure. A second,
lower-level endpoint accepts raw 3D coordinates directly, for anyone who wants to bypass
the reconstruction.

---

## Project architecture

```text
User picks (phi, psi)
    ↓
FastAPI endpoint
    ↓
NeRF structure reconstruction (fixed bond lengths/angles from training data)
    ↓
CG force field (harmonic prior + GNN correction, loaded from mlcg-gnn-alanine)
    ↓
Energy + per-bead force magnitude
```

The Streamlit frontend also queries a `/landscape` endpoint that evaluates the model on
a full (phi, psi) grid, rendered as a heatmap with the user's current point marked. This
makes it so that the whole energy surface can be seen, not just one prediction.

---

## Repository structure

```text
cgnet-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + endpoints
│   ├── model_loader.py         # loads the trained prior + GNN correction weights
│   ├── structure_builder.py    # NeRF: (phi, psi) -> 5-bead 3D structure
│   ├── predict.py              # prediction + energy-landscape logic
│   ├── cg_gnn_model.py         # model classes (copied from mlcg-gnn-alanine/src)
│   └── schemas.py              # pydantic request/response schemas
├── models/
│   ├── cgnet_correction_state.pt
│   ├── prior_params.pt
│   └── mean_bond_angles_deg.npy
├── streamlit_app/
│   └── app.py
├── tests/
│   └── test_api.py
├── README.md
├── requirements.txt
├── pytest.ini
└── Dockerfile
```

---

## API endpoints

### Health check

```
GET /
```

### Predict from phi/psi (the main, user-friendly endpoint)

```
POST /predict/from_dihedrals
{
  "phi": -150.0,
  "psi": 150.0
}
```

Example response:

```json
{
  "phi": -150.0,
  "psi": 150.0,
  "energy_kj_mol": -21.696,
  "force_magnitude_kj_mol_nm": [544.17, 470.47, 1048.39, 197.51, 1048.93],
  "model": "Harmonic bond prior + SchNet-style GNN correction"
}
```

### Predict from raw structure

```
POST /predict/from_structure
{
  "positions_nm": [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], [x3,y3,z3], [x4,y4,z4]]
}
```

### Energy landscape (grid of predictions, powers the Streamlit heatmap)

```
GET /landscape?n_grid=35
```

---

## Running it

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI backend:

```bash
uvicorn app.main:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

In a second terminal, start the Streamlit frontend:

```bash
streamlit run streamlit_app/app.py
```

Frontend: `http://localhost:8501`

Run the tests (in-process, no server needed):

```bash
pytest tests/
```

### Docker

```bash
docker build -t cgnet-api .
docker run -p 8000:8000 cgnet-api
```

(The Streamlit frontend is run separately, outside the container, pointed at the
container's API via the `CGNET_API_URL` environment variable.)

---

## An honest limitation

The underlying model (see `mlcg-gnn-alanine`) reproduces the location of the dominant
conformational basin seen in the real reference simulation, but a short CG dynamics run
did not resolve the second major basin, and the model's energy surface is not
well-constrained in regions the training data rarely visited (e.g. large positive phi).
The energy landscape heatmap in this app makes this visible: values away from the
training data's most-visited region should be treated as extrapolation, not a validated
prediction, exactly as flagged in the training project's own write-up.

---

## Related work

This deployment extends the companion project:

**Machine-Learned Coarse-Grained Force Field (GNN) for Alanine Dipeptide**
(`mlcg-gnn-alanine`) — which covers the reference simulation, the CG mapping, the model
architecture, training by force matching, and the full honest validation this app's
"note" section summarizes.

---

## Technologies used

- Python, PyTorch
- FastAPI, Pydantic
- Streamlit, Matplotlib
- Docker
- pytest

---

## Author

Anne Mc Quaid

MSc Physics — Computational Biophysics and Machine Learning
