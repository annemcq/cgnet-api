import os

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st


API_URL = st.secrets.get("CGNET_API_URL", os.environ.get("CGNET_API_URL", "http://127.0.0.1:8000"))

st.set_page_config(
    page_title="CG Force Field Energy Landscape",
    page_icon="🧬",
    layout="centered",
)

st.title("🧬 CG Force Field Energy Landscape Explorer")

st.markdown(
    """
    Explore the energy landscape predicted by a coarse-grained force field
    for alanine dipeptide. The model combines a harmonic bonded prior with
    a SchNet-style GNN correction trained by force matching.

    Choose pseudo-phi and pseudo-psi angles below to reconstruct a five-bead
    structure and evaluate its predicted energy.
    """
)


@st.cache_data(show_spinner="Computing energy landscape from the model...")
def get_landscape(n_grid: int = 35):
    response = requests.get(
        f"{API_URL}/landscape",
        params={"n_grid": n_grid},
        timeout=120,
    )
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
    st.warning(
        "The backend may be waking up after an idle period. "
        "Please wait a minute and refresh the page. "
        "If it still fails, check the API service status."
    )


st.subheader("Pick a backbone conformation")

col1, col2 = st.columns(2)

with col1:
    phi = st.slider(
        "phi (degrees)",
        -180.0,
        180.0,
        -150.0,
        step=1.0,
    )

with col2:
    psi = st.slider(
        "psi (degrees)",
        -180.0,
        180.0,
        150.0,
        step=1.0,
    )

st.caption(
    "The reference trajectory contains an extended beta/C5-like region around "
    "phi ≈ -150°, psi ≈ 150° and an alpha-like region around "
    "phi ≈ -70°, psi ≈ -40°."
)

if landscape_available:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    mesh = ax.pcolormesh(
        phi_grid,
        psi_grid,
        energy.T,
        shading="auto",
        cmap="viridis",
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="predicted energy (kJ/mol)",
    )

    ax.plot(
        phi,
        psi,
        "r*",
        markersize=18,
        markeredgecolor="white",
    )

    ax.set_xlabel("phi (degrees)")
    ax.set_ylabel("psi (degrees)")
    ax.set_title("Model-predicted CG energy landscape")

    st.pyplot(fig)


if st.button("Evaluate model at this (phi, psi)"):
    payload = {
        "phi": phi,
        "psi": psi,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict/from_dihedrals",
            json=payload,
            timeout=120,
        )

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction completed successfully.")
            st.subheader("Prediction result")

            st.write(
                f"**Predicted energy:** "
                f"{result['energy_kj_mol']} kJ/mol"
            )

            st.write(
                "**Force magnitude per bead (kJ/mol/nm):** "
                + ", ".join(
                    f"{f:.1f}"
                    for f in result["force_magnitude_kj_mol_nm"]
                )
            )

            st.write(f"**Model:** {result['model']}")

        else:
            st.error(
                f"API error: {response.json().get('detail')}"
            )

    except Exception:
        st.error(
            "The API may be waking up after an idle period. "
            "Wait a minute and try again; if it persists, check the backend status."
        )


st.markdown("---")

st.caption(
    "The underlying force field was trained and validated in the "
    "`mlcg-gnn-alanine` project. The landscape shown here is a visualization "
    "of the learned model and should not be interpreted as a quantitatively "
    "validated free-energy surface."
)
