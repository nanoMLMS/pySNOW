import numpy as np
from scipy.special import sph_harm
from scipy.spatial import cKDTree
from snow.lodispp.utils import nearest_neighbours
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not installed, define a dummy tqdm that does nothing.
    def tqdm(iterable, **kwargs):
        return iterable

def peratom_steinhardt(index_frame: int, coords: np.ndarray, l: list, cut_off: float):
    """
    Calculate per-atom Steinhardt order parameters for a given frame of atomic coordinates.

    Parameters
    ----------
        index_frame: int
            Index of the current frame (for logging or reference).
        coords: ndarray
            Atomic coordinates, shape (n_atoms, 3).
        l: int
            Degree of spherical harmonics.
        cut_off: float
            Cut-off distance to consider neighbors.

    Returns
    -------
        ndarray (len_l x n_atoms)
            Array of Steinhardt parameters (Q_l) for each l and for each atom. 
    """
    n_atoms = coords.shape[0]
    #tree = cKDTree(coords)
    print(cut_off)
    neigh_list = nearest_neighbours(index_frame=index_frame, coords=coords, cut_off=cut_off)
    

    # Array to store Q_l for each atom
    Q_l = np.zeros((len(l), n_atoms))

    # Loop over all atoms
    for id_s, q in enumerate(l):
        print("\n \n Evaluating Steinhardt parameter of order {} \n \n".format(q))
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

                

def average_steinhardt():
    pass