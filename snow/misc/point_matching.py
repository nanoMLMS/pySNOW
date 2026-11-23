import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def match_atoms(ref_coords, new_coords, ref_species=None, new_species=None, big=1e6):
    """
    Return the permutation that most closely maps new_coords to old_coords
    If species are provided, it will match atoms with same species.
    
    Parameters
    ----------
    ref_coords : np.ndarray
        Coordinates of the reference configuration 
    new_coords : np.ndarray
        Coordinates of the new configuration 
    ref_species : list[str]
        List of atomic symbols in the reference configuration
    new_species : list[str]
        List of atomic symbols in the new configuration
    big : float
        A large penalty in loss function for matching different species
    
    Returns
    -------
    perm : list(int) 
        The permutation that maps the new configuration into the reference one
    disp : np.ndarray
        The per-atom displacements
    """
    D = cdist(ref_coords, new_coords)
    if (new_species is not None) and (ref_species is not None):
        mism = np.array(ref_species)[:, None] != np.array(new_species)[None, :]
        D[mism] = big                      # forbid cross-species matches
    rows, cols = linear_sum_assignment(D)
    assert D[rows, cols].max() < big, "species counts differ"
    perm = cols
    disp = D[rows,cols]
    return perm, disp # permutation + per-atom displacement
