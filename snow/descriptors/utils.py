import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

rescale = (1 + np.sqrt(2)) / 2



def distance_matrix(coords):
    """
    Computes distance between atoms and saves them in a matrix of distances

    Parameters
    ----------
    coords : np.ndarray
        array of the 3d coordinates of the N atoms

    Returns
    -------
    tuple
    distance_materix : np.ndarray
        distance matrix
    max : float
        maximum of the distance matrix
    min : float
        minimum of the distance matrix
    """
    distances = pdist(coords)
    dist_mat = squareform(distances)
    dist_min = 0.0
    dist_max = np.max(dist_mat)
    return dist_mat, dist_max, dist_min


def adjacency_matrix(coords, cutoff):
    """
    Computes the adjacency matrix Ad_ij, where the entry ij is 1 if distance r_ij<=cutoff, else is 0

    Parameters
    ----------
    coords : np.ndarray
        coordinates of the system
    cutoff : np.ndarray
        cutoff to define neighbouring atoms

    Returns
    -------
    adj_matrix : np.ndarray
        adjacency matrix

    """
    dist_mat = distance_matrix(coords)[0]
    adj_matrix = dist_mat <= cutoff
    return adj_matrix


def sparse_adjacency_matrix(coords, cutoff):
    """
    Create a sparse adjacency matrix where entries are 1 if the distance between points
    is less than or equal to the cutoff.

    Parameters
    ----------
    coords : ndarray
        Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
    cut_off : float
        Cutoff distance for finding neighbors in angstrom.

    Returns
    -------
    adjacency_matrix : np.ndarray
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


def hetero_distance_matrix(coords, elements):
    """Computes the distance matrix for only atoms with different chemical species 
    
    Only takes into account hetero pairs, e.g. if atom "i" is a gold atom and atom "j" a platinum atom
    then $M_{ij} = d_{ij}$, if they are both gold atoms then $M_{ij} = 0$)

    Parameters
    ----------
    coords : ndarray
        positions of the atoms in the system
    elements : ndarray
        array of chemical species of the atoms in the system

    Returns
    -------
    dist_mat : np.ndarray
        hetero-pairs distance matrix
        
    """
    n_atoms = np.shape(coords)[0]
    dist_mat, _, _ = distance_matrix(coords=coords)

    triu_indices = np.triu_indices(n_atoms, k=1)
    id_i, id_j = triu_indices

    for i in range(len(id_i)):
        id_is, id_js = id_i[i], id_j[i]

        if elements[id_is] == elements[id_js]:
            dist_mat[id_is, id_js] = 0
            dist_mat[id_js, id_is] = 0  # Ensure symmetry

    return dist_mat

def distance_matrix_pbc(positions, cell):
    """
    Compute pairwise distance matrix under periodic boundary conditions. 
    
    CKDTree only accepts parallelopipedal cells. This method is slower but works with any cell
    provided as a (3,3) array (3 axes defining the cell). 

    Parameters
    ----------
    positions : (N,3) array-like
        Cartesian coordinates.
    cell : (3,3) array-like
        Simulation cell matrix (rows = lattice vectors).

    Returns
    -------
    dmat : (N,N) ndarray
        Pairwise distance matrix using minimum image convention.
    """

    pos = np.asarray(positions)
    cell = np.asarray(cell)

    inv_cell = np.linalg.inv(cell)

    # fractional coordinates
    frac = pos @ inv_cell

    # pairwise fractional displacements
    df = frac[:, None, :] - frac[None, :, :]

    # minimum image convention
    df -= np.round(df)

    # back to Cartesian
    dr = df @ cell

    # distances
    dmat = np.linalg.norm(dr, axis=-1)

    return dmat, np.max(dmat), np.min(dmat) #for compatibility also return max & min distance

def nn_pbc(coords, box, cut_off):
    """
    Computes the nearest neighbour list with a slower but pbc-compatible algorithm

    Nearest neighbour calculation with KDTree can only be done with rectangular boxes. This helper function allows to get nearest neighbours in PBC
    with any kind of box, given the box as ((a1, a2, a3), (b1, b2, b3), (c1, c2, c3))
    
    Parameters
    ----------
    coords : ndarray
        XYZ coordinates of atoms, shape (n_atoms, 3).
    box : ndarray
        Simulation cell matrix with shape (3, 3), where rows are lattice vectors.
    cut_off : float
        Cutoff distance for finding neighbors (in Å).

    Returns
    -------
    neigh : list of lists
        The i-th sublist contains the indices of neighboring atoms for the i-th atom.
    """

    dmat   = distance_matrix_pbc(coords, box)[0]
    ad_mat = dmat < cut_off
    np.fill_diagonal(ad_mat, False)

    # Convert adjacency matrix to list of neighbor lists
    neigh = []
    for i in range(ad_mat.shape[0]):
        neighbors = np.where(ad_mat[i])[0].tolist()
        neigh.append(neighbors)
    
    return neigh

def nearest_neighbours(
    coords: np.ndarray,
    cut_off: float = None,
    pbc: bool = False,
    box: np.ndarray = None,
) -> list:
    """
    Computes nearest neighbors for each atom, considering periodic boundary conditions (PBC) if requested.

    Parameters
    ----------
    coords : ndarray
        XYZ coordinates of atoms, shape (n_atoms, 3).
    cut_off : float, optional
        Cutoff distance for finding neighbors (in Å). If None, an adaptive cutoff is used.
    pbc : bool, optional
        Whether to apply periodic boundary conditions (default: False).
    box : ndarray, optional
        Simulation box size in the form (3,) for orthorhombic boxes or (3,2) for lower and upper bounds or (3,3) if you pass box vectors (slower).

    Returns
    -------
    neigh : list of lists
        The i-th sublist contains the indices of neighboring atoms for the i-th atom.
    """

    if pbc:
        if box is None:
            raise ValueError("Box must be provided if PBC is enabled.")

        # Ensure box is correctly formatted
        if box.shape == (3, 2):  # Lower and upper bounds provided
            box_size = box[:, 1] - box[:, 0]
        elif box.shape == (3,):  # Direct box lengths
            box_size = box
        elif box.shape == (3,3):
            #resort to slower but more robust function
            return nn_pbc(coords, box, cut_off)
        else:
            raise ValueError("Box must be of shape (3,) or (3,2) or (3,3)")

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

def pairs_from_neighbor_list(neigh_list):
    """
    Convert neighbor list to unique pair list.
    
    Parameters
    ----------
    neigh_list : list of lists
        For each atom i, a list of its neighbors
        
    Returns
    -------
    piars : list of tuples
        Unique pairs
    """
    pairs = set()
    for i, neighbors in enumerate(neigh_list):
        for j in neighbors:
            if i < j:  # Only add if i < j to avoid duplicates
                pairs.add((i, j))
    
    return list(pairs)

def pair_list(
    coords: np.ndarray,
    cut_off: float = None,
    pbc: bool = False,
    box: np.ndarray = None,
) -> list:
    """
    Generates a list of all pairs of atoms within a certain adaptive cutoff distance of each other.

    Parameters
    ----------
    coords : np.ndarray
        Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
    cut_off : float, optional
        Cutoff distance for finding pairs in angstroms. If None, an adaptive cutoff is used per atom.
    pbc : bool, default False
        Whether to apply periodic boundary conditions.
    box : np.ndarray, optional
        Simulation box size (either [Lx, Ly, Lz] or [[xmin, xmax], [ymin, ymax], [zmin, zmax]] or 3 cell vectors (shape (3,3) - slower)). Has to provided if pbc=True

    Returns
    -------
    pairs : list
        List of tuples with indexes of pairs of atoms that are within the cutoff distance.
    """

    if pbc:
        if box is None:
            raise ValueError("Box must be provided if PBC is enabled.")

        # Ensure box is correctly formatted
        if box.shape == (3, 2):  # Lower and upper bounds provided
            box_size = box[:, 1] - box[:, 0]
        elif box.shape == (3,):  # Direct box lengths
            box_size = box
        elif box.shape == (3,3):
            neighbours_list = nn_pbc(coords, box, cut_off)
            return pairs_from_neighbor_list(neighbours_list)
        else:
            raise ValueError("Box must be of shape (3,) or (3,2) or (3,3)")

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

    Parameters
    ----------
    func1 : np.array 
        values taken by the first function for a given set of inputs
    func2 : np.array
        values taken by the second function for the same set of inputs

    Returns
    -------
    kldiv : float)
        KL divergence between function 1 and function 2
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
    coords : np.ndarray
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
    """Calculate an axis-aligned bounding box from a set of points.
    
    Parameters
    ----------
    points : np.ndarray
        coordinates of the points in the system
    
    Returns
    -------
    box : np.ndarray
        computed axis-aligned bounding box.    
    """
    x_coordinates, y_coordinates, z_coordinates = zip(*points)
    return np.array(
        [
            [min(x_coordinates), max(x_coordinates)],
            [min(y_coordinates), max(y_coordinates)],
            [min(z_coordinates), max(z_coordinates)],
        ]
    )


def second_neighbours(
        coords: np.ndarray, cutoff: float, pbc: bool = False, box = None
) -> list:
    """Generates a list of lists of atomic indices for each atom corresponding to atoms that are neighbours of first neighbours
    excluding those which are already first neighbours.

    Parameters
    ----------
    coords : np.ndarray
        Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
    cutoff : float
        Cutoff distance for finding pairs in angstroms. If None, an adaptive cutoff is used per atom.

    Returns
    -------
    snn_list : list[list]
        List of lists containing indeces of second neighbours for each atom
    """
    neigh = nearest_neighbours(
        coords=coords, cut_off=cutoff, pbc=pbc, box=box
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

def get_coords_by_element(el, coords, chosen_element):
    """
    Returns the coordinates (and the chemical element array, for consistency) filtered for a selected chemical specie.
    It can deal with both single frame and 'movie'-style coords objects in list or np.ndarray format

    Parameters
    ----------
    el: np.ndarray or list of np.ndarrays
        chemical elements of the atoms corresponding to the positions in coords 
    cooords: np.ndarray or list of np.ndarrays
        positions of the atoms in the system
    chosen_element: str
        element you want to select in your system to get the coordinates of those atoms.
    
    Returns
    -------
    Tuple
    el : list or list[list]
        list of elements of selected atoms. The return type depends on whether the input is a single frame or a movie (list of frames)
    coords : np.ndarray or list[np.ndarray]
        cooridnates of selected atoms. The return type depends on whether the input is a single frame or a movie (list of frames)
        
    """

    if type(coords) == list or ( type(coords) == np.ndarray and len(coords.shape) > 2): #movie-style
        
        selected_coords = []
        return_elements = []

        #read over frames
        for frame_el, frame_coords in zip(el, coords):

            selected_from_frame = []

            #check on each atom
            for atoms_el, atoms_coords in zip(frame_el, frame_coords):
                if atoms_el == chosen_element:
                    selected_from_frame.append(atoms_coords)
        
            #prepare el list and convert to np.ndarray before adding to final list
            return_elements.append( [chosen_element]*len(selected_from_frame) )
            selected_coords.append( np.asarray(selected_from_frame) )
            
    
    elif type(coords) == np.ndarray and len(coords.shape) == 2: #single-frame

        selected_coords = []

        #check on each atoms
        for atoms_el, atoms_coords in zip(el, coords):
            if atoms_el == chosen_element:
                selected_coords.append(atoms_coords)
        
        #prepare el list and convert to np.ndarray
        return_elements = [chosen_element]*len(selected_coords)
        selected_coords = np.asarray(selected_coords)
        
    else:
        raise ValueError('Unknown format for coords array.')    

    return return_elements, selected_coords



def _check_structure(coords: np.ndarray, elements: list | None = None, *, require_elements: bool = False):
    """sanity-checks that a structure (list of coordinates) is provided in the right format

    Parameters
    ----------
    coords : np.ndarray
        coordinates of the atoms in the system
    elements : list | None, optional
        elements list for the atoms in the system
    require_elements : bool, optional
        if elements are required for further checks. No check on elements is currently implemented.
    """
    if not isinstance(coords, np.ndarray):
        raise ValueError("Coordinates must be provided as np.ndarray")
    
    #check if coordinates have the right shape and dimensions
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coordinates array must have shape (N, 3), got {coords.shape}")

def pbc_distance(p1, p2, box):
    """
    Computes the minimum image distance between two points under periodic boundary conditions.

    Parameters
    ----------
    p1 : np.ndarray
        Position of the first point.
    p2 : np.ndarray
        Position of the second point.
    box : np.ndarray
        Cell vectors as a (3,3) array (rows are vectors a, b, c) or (3,) array (xmax, ymax, zmax).

    Returns
    -------
    distance : float
        Minimum image distance between p1 and p2.
    """

    if type(box) == list and len(box)==3 or type(box)==np.ndarray and (box.shape==(3,) or box.shape==(3,1)):
        box = np.asarray([[box[0],0.,0.], [0., box[1], 0.], [0., 0., box[2]] ])
    elif box.shape !=(3,3):
        raise Exception('Please provide the box as either a (3,3) or (3,) array or list.')


    diff    = p1-p2
    inv_box = np.linalg.inv(box)

    # fractional coordinates
    frac = diff @ inv_box

    # wrap to [-0.5, 0.5]
    frac -= np.round(frac)
    # back to cartesian
    diff_mic = frac @ box
    return np.linalg.norm(diff_mic)

def progress_bar(current, total, length=50):
    """
    Prints a nice progess bar

    Parameters
    ----------
    current : float
        current step
    total : float
        total number of steps
    length : int, default 50
        how many elements your progress bar should be made of.
    """
    percent = current / total
    filled_length = int(length * percent)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r[{bar}] {percent * 100:.2f}%', end='')
    return