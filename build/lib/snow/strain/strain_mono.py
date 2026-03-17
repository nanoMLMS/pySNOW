from scipy.spatial import cKDTree
import numpy as np

def strain_mono(index_frame: int, coords: np.ndarray, dist_0: float, cut_off: float, neigh_list: list = None, coordination: np.ndarray = None):
    """
    Computes the strain for each atom based on neighbor distances.

    Parameters:
        index_frame (int): Frame index (unused in this implementation but included for compatibility).
        coords (np.ndarray): Nx3 array of atomic coordinates.
        dist_0 (float): Reference distance for strain calculation.
        cut_off (float): Cutoff radius for neighbor search.
        neigh_list (list, optional): Precomputed neighbor list to speed up if already calculated (default: None).
        coordination (np.ndarray, optional): Precomputed coordination numbers to speed up if already calculated (default: None).

    Returns:
        np.ndarray: Array of strain values for all atoms.
    """
    n_atoms = coords.shape[0]
    
    # Generate neighbor list if not provided
    if neigh_list is None:
        tree = cKDTree(coords)
        neigh_list = [tree.query_ball_point(coords[i], cut_off) for i in range(n_atoms)]
    
    # Generate coordination numbers if not provided
    if coordination is None:
        coordination = np.array([len(neighs) for neighs in neigh_list])
    
    # Initialize strain array
    strain_syst = np.zeros(n_atoms)
    
    # Build sparse distance matrix
    tree = cKDTree(coords)
    sparse_distmat = tree.sparse_distance_matrix(tree, cut_off)
    
    # Calculate strain for each atom
    for i in range(n_atoms):
        n_neigh = int(coordination[i])
        if n_neigh == 0:  # Avoid division by zero for isolated atoms
            continue
        
        strain_temp = 0
        for j in neigh_list[i]:
            if i == j:  # Skip self-interaction
                continue
            d_ij = sparse_distmat.get((i, j), None)
            if d_ij is not None:  # Ensure distance exists in the matrix
                strain_temp += (d_ij - dist_0) / dist_0
        
        strain_syst[i] = 100 * strain_temp / n_neigh  # Strain as percentage
    
    return strain_syst
