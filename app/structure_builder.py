# ============================================================
# INTERNAL-COORDINATE STRUCTURE BUILDER
# CG Force Field Energy Landscape API
# ============================================================
#
# Places the 5 CG beads in 3D from internal coordinates (bond
# lengths, bond angles, and the two backbone dihedrals phi/psi),
# using NeRF (Natural Extension Reference Frame) -- the standard
# algorithm for reconstructing chain geometry from internal
# coordinates in structural biology.
#
# Bond lengths and (fixed) bond angles are the mean values measured
# from the training trajectory; phi and psi are the two degrees of
# freedom the user controls. Together, 4 bond lengths + 3 bond
# angles + 2 dihedrals fully determine the internal geometry of a
# 5-point chain (9 internal degrees of freedom for 5 points in 3D).

import numpy as np

BOND_LENGTHS_NM = [0.1863, 0.1938, 0.2430, 0.2176]     # bonds (0,1) (1,2) (2,3) (3,4)
BOND_ANGLES_DEG = [140.43, 81.57, 103.17]                # angles (0,1,2) (1,2,3) (2,3,4)


def _place_next(p0, p1, p2, bond_length, bond_angle_deg, dihedral_deg):
    """
    NeRF: given three previously-placed points p0, p1, p2, place a new
    point p3 at the given bond length (p2-p3), bond angle (p1,p2,p3),
    and dihedral (p0,p1,p2,p3).
    """
    bond_angle = np.radians(180.0 - bond_angle_deg)
    dihedral = np.radians(dihedral_deg)

    d2 = np.array([
        -bond_length * np.cos(bond_angle),
        bond_length * np.sin(bond_angle) * np.cos(dihedral),
        bond_length * np.sin(bond_angle) * np.sin(dihedral),
    ])

    bc = p2 - p1
    bc /= np.linalg.norm(bc)
    ab = p1 - p0
    n = np.cross(ab, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)

    M = np.stack([bc, m, n], axis=1)  # columns = local frame axes in world coords
    return p2 + M @ d2


def build_structure_from_dihedrals(phi_deg: float, psi_deg: float) -> np.ndarray:
    """
    Returns (5, 3) CG bead positions (nm), centered at the origin,
    consistent with the given backbone phi/psi and the mean bond
    lengths/angles measured from the training data.

    phi = dihedral(bead0, bead1, bead2, bead3)
    psi = dihedral(bead1, bead2, bead3, bead4)
    """
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([BOND_LENGTHS_NM[0], 0.0, 0.0])

    # place p2 using bond angle (0,1,2) only, in the xy-plane (no dihedral defined yet)
    a01 = np.radians(180.0 - BOND_ANGLES_DEG[0])
    p2 = p1 + BOND_LENGTHS_NM[1] * np.array([-np.cos(a01), np.sin(a01), 0.0])

    p3 = _place_next(p0, p1, p2, BOND_LENGTHS_NM[2], BOND_ANGLES_DEG[1], phi_deg)
    p4 = _place_next(p1, p2, p3, BOND_LENGTHS_NM[3], BOND_ANGLES_DEG[2], psi_deg)

    coords = np.stack([p0, p1, p2, p3, p4])
    return coords - coords.mean(axis=0, keepdims=True)
