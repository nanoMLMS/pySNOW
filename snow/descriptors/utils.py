import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist, squareform



def distance_matrix(index_frame, coords):
    """
    Computes distance between atoms and saves them in a matrix of distances

    Parameters:
        index_frame (int): index of the current frame if it is a movie, mostrly for reference
        coords (array): array of the 3d coordinates

    Returns:
        dist_mat (array): distance matrix

        dist_max (float): the maximum distance between atoms
        dist_min (float): the minimum distance between atoms
    """
    distances = pdist(coords)
    dist_mat = squareform(distances)
    dist_min = 0.0
    dist_max = np.max(dist_mat)
    return dist_mat, dist_max, dist_min


def adjacency_matrix(index_frame, coords, cutoff):
    """
    Computes the adjacency matrix Ad_ij, where the entry ij is 1 if distance r_ij<=cutoff, else is 0
    """
    dist_mat = distance_matrix(index_frame, coords)[0]
    adj_matrix = dist_mat <= cutoff
    return adj_matrix


def sparse_adjacency_matrix(index_frame, coords, cutoff):
    """
    Create a sparse adjacency matrix where entries are 1 if the distance between points
    is less than or equal to the cutoff.

    Parameters
    ----------
    index_frame : int
        Number of the frame if a movie.
    coords: ndarray
        Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
    cut_off : float
        Cutoff distance for finding neighbors in angstrom.

    Returns
    -------
    ndarray
        A sparse adjacency matrix with ones for adjacent points.
    """
    # Create a KDTree for efficient distance queries
    tree = cKDTree(coords)

    # Get pairs of points within the cutoff distance
    pairs = tree.sparse_distance_matrix(tree, cutoff).keys()

    # Extract row and column indices from pairs
    rows, cols = zip(*pairs)

    # Create a sparse adjacency matrix with ones
    adjacency_matrix = coo_matrix(
        (len(coords), len(coords)), shape=(len(coords), len(coords))
    )
    adjacency_matrix = coo_matrix((np.ones(len(rows), dtype=int), (rows, cols)))

    return adjacency_matrix



def hetero_distance_matrix(index_frame, coords, elements):
    """_summary_

    Parameters
    ----------
    index_frame : _type_
        _description_
    coords : _type_
        _description_
    elements : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    n_atoms = np.shape(coords)[0]
    dist_mat, dist_max, dist_min = distance_matrix(
        index_frame=index_frame, coords=coords
    )

    triu_indices = np.triu_indices(n_atoms, k=1)
    id_i, id_j = triu_indices

    for i in range(len(id_i)):
        id_is, id_js = id_i[i], id_j[i]

        if elements[id_is] == elements[id_js]:
            dist_mat[id_is, id_js] = 0
            dist_mat[id_js, id_is] = 0  # Ensure symmetry

    return dist_mat

def nearest_neighbours(
    index_frame: int,
    coords: np.ndarray,
    cut_off: float = None,
    pbc: bool = False,
    box: np.ndarray = None,
) -> list:
    """
    Computes nearest neighbors for each atom, considering periodic boundary conditions (PBC) if requested.

    Parameters
    ----------
    index_frame : int
        Number of the frame if a movie.
    coords : ndarray
        XYZ coordinates of atoms, shape (n_atoms, 3).
    cut_off : float, optional
        Cutoff distance for finding neighbors (in Å). If None, an adaptive cutoff is used.
    pbc : bool, optional
        Whether to apply periodic boundary conditions (default: False).
    box : ndarray, optional
        Simulation box size in the form (3,) for orthorhombic boxes or (3,2) for lower and upper bounds.

    Returns
    -------
    list of lists
        Each sublist contains the indices of neighboring atoms for the corresponding atom.
    """
    sqrt_2 = np.sqrt(2)

    if pbc:
        if box is None:
            raise ValueError("Box must be provided if PBC is enabled.")

        # Ensure box is correctly formatted
        if box.shape == (3, 2):  # Lower and upper bounds provided
            box_size = box[:, 1] - box[:, 0]
        elif box.shape == (3,):  # Direct box lengths
            box_size = box
        else:
            raise ValueError("Box must be of shape (3,) or (3,2)")

        # Create KD-tree with periodic boundaries
        neigh_tree = cKDTree(coords, boxsize=box_size)
    else:
        # Standard KD-tree without PBC
        neigh_tree = cKDTree(coords)

    r_cut = np.full(len(coords), cut_off if cut_off is not None else 0.0)

    if cut_off is None:
        # Compute an adaptive cutoff
        for i, atom in enumerate(coords):
            d, _ = neigh_tree.query(atom, k=12)  # 12 nearest neighbors
            d_avg = np.mean(d)
            r_cut[i] = rescale * d_avg  # Adaptive cutoff

    # Find neighbors within cutoff
    neigh = []
    for i, atom in enumerate(coords):
        neighbors = neigh_tree.query_ball_point(
            atom, r_cut[i]
        )  # Find points within radius
        neigh.append([n for n in neighbors if n != i])  # Remove self

    return neigh




import numpy as np
from scipy.spatial import cKDTree


def pair_list(
    index_frame: int,
    coords: np.ndarray,
    cut_off: float = None,
    pbc: bool = False,
    box: np.ndarray = None,
) -> list:
    """
    Generates a list of all pairs of atoms within a certain adaptive cutoff distance of each other.

    Parameters
    ----------
    index_frame : int
        Index of the frame if a movie (not used in the current version but can be useful for dynamic systems).
    coords : np.ndarray
        Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
    cut_off : float, optional
        Cutoff distance for finding pairs in angstroms. If None, an adaptive cutoff is used per atom.
    pbc : bool, optional
        Whether to apply periodic boundary conditions. Defaults to False.
    box : np.ndarray, optional
        Simulation box size (either [Lx, Ly, Lz] or [[xmin, xmax], [ymin, ymax], [zmin, zmax]]).

    Returns
    -------
    list
        List of tuples representing pairs of atoms that are within the cutoff distance.
    """
    sqrt_2 = np.sqrt(2)

    if pbc:
        if box is None:
            raise ValueError("Box must be provided if PBC is enabled.")

        # Ensure box is correctly formatted
        if box.shape == (3, 2):  # Lower and upper bounds provided
            box_size = box[:, 1] - box[:, 0]
        elif box.shape == (3,):  # Direct box lengths
            box_size = box
        else:
            raise ValueError("Box must be of shape (3,) or (3,2)")

        # Create KD-tree with periodic boundaries
        neigh_tree = cKDTree(coords, boxsize=box_size)
    else:
        # Standard KD-tree without PBC
        neigh_tree = cKDTree(coords)

    # Compute an adaptive cutoff for each atom if not provided
    r_cut = np.full(len(coords), cut_off if cut_off is not None else 0.0)

    if cut_off is None:
        for i, atom in enumerate(coords):
            d, _ = neigh_tree.query(atom, k=12)  # 12 nearest neighbors
            d_avg = np.mean(d)
            r_cut[i] = rescale * d_avg  # Adaptive cutoff per atom

    # Find atom pairs within adaptive cutoffs
    pairs = set()
    for i, atom in enumerate(coords):
        neighbors = neigh_tree.query_ball_point(atom, r_cut[i])
        for j in neighbors:
            if i < j:  # Avoid duplicates (ensures (i, j) but not (j, i))
                pairs.add((i, j))

    return list(pairs)