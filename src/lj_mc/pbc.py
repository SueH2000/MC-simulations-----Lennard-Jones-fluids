import numpy as np


def apply_pbc(dx, dy, dz, box_length):
    """Apply periodic boundary conditions using minimum-image convention.

    Works for scalars or numpy arrays.
    """
    dx = dx - np.round(dx / box_length) * box_length
    dy = dy - np.round(dy / box_length) * box_length
    dz = dz - np.round(dz / box_length) * box_length
    return dx, dy, dz

