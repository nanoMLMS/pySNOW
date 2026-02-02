import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist, squareform

rescale = (1 + np.sqrt(2)) / 2



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
    """Computes the distance matrix for atoms with different chemical species (eg. if atom "i" is a gold atom and atom "j" a platinum atom then $M_{ij} = d_{ij}$,
    if they are both gold atoms then $M_{ij} = 0$)

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

def kl_div(func1: np.array, func2: np.array) -> float:
    """
    Calculate the Kullback-Leibler divergence between two functions.

    Arguments:
        func1 (np.array) : values taken by the first function for a given set of inputs
        func2 (np.array) : values taken by the second function for the same set of inputs

    Returns:
        kldiv (float) : KL Divergency between function 1 and function 2
    """
    kldiv = 0.0

    for y1, y2 in zip(func1, func2):
        kldiv += y1 * np.log(y1 / y2)
    return kldiv




def apply_pbc(coords, box_size):
    """
    Apply periodic boundary conditions to atom coordinates.
    Maps coordinates to be within the simulation box.

    Parameters
    ----------
    coords : np.ndarray
        Atom coordinates array, shape (n_atoms, 3).
    box_size : np.ndarray
        Box size array (3,) or (3,2) depending on box format.

    Returns
    -------
    np.ndarray
        Coordinates after applying PBC.
    """
    if box_size.shape == (3, 2):  # Lower and upper bounds provided
        lower_bounds = box_size[:, 0]
        upper_bounds = box_size[:, 1]
        # Apply modulo to map the coordinates back into the box
        coords = (
            np.mod(coords - lower_bounds, upper_bounds - lower_bounds)
            + lower_bounds
        )
    elif box_size.shape == (3,):  # Direct box lengths
        coords = np.mod(coords, box_size)
    else:
        raise ValueError("Box size must be of shape (3,) or (3,2)")

    return coords


def bounding_box(points):
    """Calculate an axis-aligned bounding box from a set of points."""
    x_coordinates, y_coordinates, z_coordinates = zip(*points)
    return np.array(
        [
            [min(x_coordinates), max(x_coordinates)],
            [min(y_coordinates), max(y_coordinates)],
            [min(z_coordinates), max(z_coordinates)],
        ]
    )





def second_neighbours(
    index_frame: int, coords: np.ndarray, cutoff: float
) -> list:
    """Generates a list of lists of atomic indeces for each atom corresponding to aotoms that are neighbours of first neighbours
    excluding those which are already first neighbours.

    Parameters
    ----------
    index_frame : int
        Index of the frame if a movie (not used in the current version but can be useful for dynamic systems).
    coords : np.ndarray
        Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
    cutoff : float
        Cutoff distance for finding pairs in angstroms. If None, an adaptive cutoff is used per atom.

    Returns
    -------
    list
        List of lists containing indeces of second neighbours for each atom
    """
    neigh = nearest_neighbours(
        index_frame=index_frame, coords=coords, cut_off=cutoff
    )
    snn_list = []
    n_atoms = np.shape(coords)[0]
    for i in range(n_atoms):
        temp_snn = []
        for j in neigh[i]:
            int_ij = np.intersect1d(neigh[i], neigh[j], assume_unique=True)
            snn_ij = np.setdiff1d(neigh[j], int_ij, assume_unique=True).tolist()
            temp_snn.extend(snn_ij)
        temp_snn = list(dict.fromkeys(temp_snn))
        snn_list.append(temp_snn)

    return snn_list



def _check_structure(coords: np.ndarray, elements: np.ndarray | None = None, *, require_elements: bool = False):
    """_summary_

    Parameters
    ----------
    coords : np.ndarray
        _description_
    elements : np.ndarray | None, optional
        _description_, by default None
    require_elements : bool, optional
        _description_, by default False
    """
    if not isinstance(coords, np.ndarray):
        raise ValueError("Coordinates must be provided as np.ndarray")
    
    #check if coordinates have the right shape and dimensions
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coordinates array must have shape (N, 3), got {coords.shape}")
    