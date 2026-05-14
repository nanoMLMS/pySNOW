import numpy as np
from scipy.special import sph_harm
from scipy.spatial import cKDTree
from snow.descriptors.utils import nearest_neighbours
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not installed, define a dummy tqdm that does nothing.
    def tqdm(iterable, **kwargs):
        return iterable

def peratom_steinhardt(coords: np.ndarray, l: list[int], cut_off: float, pbc : bool=False, box = None):
    """
    Calculate per-atom Steinhardt order parameters for a given frame of atomic coordinates.

    Parameters
    ----------
    coords : ndarray
        Atomic coordinates, shape (n_atoms, 3).
    l : list[int]
        List of spherical harmonic degrees to compute (e.g. [4, 6]).
    cut_off : float
        Cutoff distance for neighbor detection.
    pbc : bool, default False
        whether to use periodic boundary conditions
    box : np.ndarray or list, optional
        simulation box to compute neighbours in pbc. Only required if `pbc` is True

    Returns
    -------
    ndarray, shape (n_l, n_atoms)
        Steinhardt order parameters Q_l for each degree in l and each atom.
    """
    n_atoms = coords.shape[0]
    neigh_list = nearest_neighbours(coords=coords, cut_off=cut_off, pbc=pbc, box=box)
    
    # Array to store Q_l for each atom
    Q_l = np.zeros((len(l), n_atoms))

    # Loop over all atoms
    for id_s, q in enumerate(l):
        #print("\n \n Evaluating Steinhardt parameter of order {} \n \n".format(q))
        for i in tqdm(range(n_atoms)):
            q_lm = np.zeros(2 * q + 1, dtype=complex)
            neighbors = neigh_list[i]

            # Loop over neighbors of atom i
            for j in neighbors:
                if i == j:
                    continue  # Skip self-interaction
                
                # Compute d_ij vector
                d_ij = coords[i] - coords[j]
                magnitude = np.linalg.norm(d_ij)
                if magnitude == 0:
                    continue  # Avoid division by zero
                
                # Compute spherical angles
                theta = np.arccos(d_ij[2] / magnitude)  # Polar angle
                phi = np.arctan2(d_ij[1], d_ij[0])      # Azimuthal angle

                # Accumulate spherical harmonics for all m
                for m in range(-q, q + 1):
                    q_lm[m + q] += sph_harm(m, q, phi, theta)

            # Normalize q_lm and compute Q_l
            n_neigh = len(neighbors)
            if n_neigh > 0:
                q_lm /= n_neigh
                Q_l[id_s,i] = np.sqrt(4 * np.pi / (2 * q + 1) * np.sum(np.abs(q_lm) ** 2))

    return Q_l


#def average_steinhardt():
#    pass
