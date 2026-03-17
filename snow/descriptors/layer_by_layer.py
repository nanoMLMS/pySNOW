import csv
import os
import numpy as np
import matplotlib.pyplot as plt


"DA SISTEMARE"

# ============================================================
# Layer analysis (numpy-based)
# ============================================================

def cut_layers_from_frame(
    elements: np.ndarray,
    coords_frame: np.ndarray,
    lattice_parameter: float,
    species_A: str,
    species_B: str,
):
    """
    Cuts a single frame into layers using z-coordinates.

    Parameters
    ----------
    coords_frame : np.ndarray
        Shape (n_atoms, 3)
    elements : np.ndarray
        Shape (n_atoms,)
        
    Return
    _________
    returns as a tuple for each layer:
    # layer, #tot of atoms in that layer, #atoms species A in that layer, #atoms species B in the layer.
    """

    elements = np.array(elements, dtype=str)  # ora è un array NumPy di stringhe
    elements = np.char.strip(elements)        # rimuove eventuali spazi bianchi


    z = coords_frame[:, 2]
    min_z = z.min()
    max_z = z.max()
    

    n_layers = int((max_z - min_z) / lattice_parameter) + 1

    layer_info = []

    for i in range(n_layers):
    

        z_min = min_z + i * lattice_parameter
        z_max = min_z + (i + 1) * lattice_parameter

        mask = (z >= z_min) & (z < z_max)
        
        tot = np.count_nonzero(mask)
        n_A = np.count_nonzero(mask & (elements == species_A))
        n_B = np.count_nonzero(mask & (elements == species_B))
        
        layer_info.append((i + 1, tot, n_A, n_B))

    return layer_info

