from typing import List, Optional

from pydantic import BaseModel, Field


class DihedralInput(BaseModel):
    phi: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Backbone pseudo-phi angle (degrees)",
    )
    psi: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Backbone pseudo-psi angle (degrees)",
    )


class StructureInput(BaseModel):
    positions_nm: List[List[float]] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="5 CG bead positions (nm), each a list of 3 floats [x, y, z]",
    )


class EnergyOutput(BaseModel):
    phi: Optional[float] = None
    psi: Optional[float] = None
    energy_kj_mol: float
    force_magnitude_kj_mol_nm: List[float]
    model: str = "Harmonic bond prior + SchNet-style GNN correction"


class LandscapeOutput(BaseModel):
    phi_grid: List[float]
    psi_grid: List[float]
    energy_kj_mol: List[List[float]]
