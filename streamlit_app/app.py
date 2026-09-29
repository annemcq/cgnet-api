# ============================================================
# STREAMLIT FRONTEND
# CG Force Field Energy Landscape Explorer
# ============================================================

import os

import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt

API_URL = os.environ.get("CGNET_API_URL", "http://127.0.0.1:8000")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CG Force Field Energy Landscape",
    page_icon="🧬",
    layout="centered",
)

st.title("🧬 CG Force Field Energy Landscape Explorer")

st.markdown(
    """
    Explore the coarse-grained (CG) energy surface learned by a graph neural
    network trained by **force matching** on alanine dipeptide (harmonic
    bonded prior + SchNet-style GNN correction — see the companion
    `mlcg-gnn-alanine` project for the full training and validation).

    Move the sliders to pick backbone **phi / psi** angles; the app builds a
    plausible 5-bead structure at those angles (fixed bond lengths/angles
    from the training data) and asks the model for its predicted energy.
    """
)

# ============================================================
# ENERGY LANDSCAPE (cached; queried once per session)
# ============================================================


@st.cache_data(show_spinner="Computing energy landscape from the model...")
def get_landscape(n_grid: int = 35):
    response = requests.get(f"{API_URL}/landscape", params={"n_grid": n_grid}, timeout=30)
    response.raise_for_status()
    return response.json()


st.subheader("Learned energy landscape")

try:
    landscape = get_landscape()
    phi_grid = np.array(landscape["phi_grid"])
    psi_grid = np.array(landscape["psi_grid"])
    energy = np.array(landscape["energy_kj_mol"])
    landscape_available = True
except Exception:
    landscape_available = False
    st.warning("Could not reach the API to compute the landscape. Is the FastAPI server running?")

# ============================================================
# USER INPUT: phi / psi sliders
# ============================================================

st.subheader("Pick a backbone conformation")

col1, col2 = st.columns(2)
with col1:
    phi = st.slider("phi (degrees)", -180.0, 180.0, -150.0, step=1.0)
with col2:
    psi = st.slider("psi (degrees)", -180.0, 180.0, 150.0, step=1.0)

st.caption(
    "Reference (from the real alanine dipeptide trajectory): the extended "
    "beta/C5-like basin sits around phi ≈ -150°, psi ≈ 150°; the alpha-helical "
    "basin sits around phi ≈ -70°, psi ≈ -40°."
)

if landscape_available:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mesh = ax.pcolormesh(phi_grid, psi_grid, energy.T, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="predicted energy (kJ/mol)")
    ax.plot(phi, psi, "r*", markersize=18, markeredgecolor="white")
    ax.set_xlabel("phi (degrees)")
    ax.set_ylabel("psi (degrees)")
    ax.set_title("Model-predicted CG energy landscape")
    st.pyplot(fig)

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("Evaluate model at this (phi, psi)"):
    payload = {"phi": phi, "psi": psi}

    try:
        response = requests.post(f"{API_URL}/predict/from_dihedrals", json=payload, timeout=15)

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction completed successfully.")
            st.subheader("Prediction result")
            st.write(f"**Predicted energy:** {result['energy_kj_mol']} kJ/mol")
            st.write(
                "**Force magnitude per bead (kJ/mol/nm):** "
                + ", ".join(f"{f:.1f}" for f in result["force_magnitude_kj_mol_nm"])
            )
            st.write(f"**Model:** {result['model']}")
        else:
            st.error(f"API error: {response.json().get('detail')}")

    except Exception:
        st.error("Could not connect to API. Make sure the FastAPI server is running.")

st.markdown("---")
st.caption(
    "Built on the `mlcg-gnn-alanine` project. The energy landscape and per-point "
    "predictions come directly from the trained model — including in regions the "
    "reference simulation barely visited, where the model's predictions are "
    "less trustworthy (see that project's README for the honest discussion "
    "of where this model does and doesn't generalize well)."
)
