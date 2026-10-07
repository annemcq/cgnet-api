import numpy as np


# Mean values measured from the mapped reference trajectory.
BOND_LENGTHS_NM = [0.1863, 0.1938, 0.2430, 0.2176]
BOND_ANGLES_DEG = [140.43, 81.57, 103.17]


def _place_next(
    p0,
    p1,
    p2,
    bond_length,
    bond_angle_deg,
    dihedral_deg,
):
    """Place the next bead from three previous positions and internal coordinates."""
    bond_angle = np.radians(180.0 - bond_angle_deg)
    dihedral = np.radians(dihedral_deg)

    d2 = np.array(
        [
            -bond_length * np.cos(bond_angle),
            bond_length * np.sin(bond_angle) * np.cos(dihedral),
            bond_length * np.sin(bond_angle) * np.sin(dihedral),
        ]
    )

    bc = p2 - p1
    bc /= np.linalg.norm(bc)

    ab = p1 - p0
    n = np.cross(ab, bc)
    n /= np.linalg.norm(n)

    m = np.cross(n, bc)

    frame = np.stack([bc, m, n], axis=1)
    return p2 + frame @ d2


def build_structure_from_dihedrals(
    phi_deg: float,
    psi_deg: float,
) -> np.ndarray:
    """Build a centered five-bead structure from pseudo-phi/psi angles."""
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([BOND_LENGTHS_NM[0], 0.0, 0.0])

    a01 = np.radians(180.0 - BOND_ANGLES_DEG[0])
    p2 = p1 + BOND_LENGTHS_NM[1] * np.array(
        [-np.cos(a01), np.sin(a01), 0.0]
    )

    p3 = _place_next(
        p0,
        p1,
        p2,
        BOND_LENGTHS_NM[2],
        BOND_ANGLES_DEG[1],
        phi_deg,
    )

    p4 = _place_next(
        p1,
        p2,
        p3,
        BOND_LENGTHS_NM[3],
        BOND_ANGLES_DEG[2],
        psi_deg,
    )

    coords = np.stack([p0, p1, p2, p3, p4])
    return coords - coords.mean(axis=0, keepdims=True)
