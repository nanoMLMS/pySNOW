import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree, ConvexHull
from snow.misc.constants import mass
from scipy.sparse import coo_matrix

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
    adjacency_matrix = coo_matrix((len(coords), len(coords)), shape=(len(coords), len(coords)))
    adjacency_matrix = coo_matrix((np.ones(len(rows), dtype=int), (rows, cols)))

    return adjacency_matrix
    
    
def pddf_calculator(index_frame, coords, bin_precision=None, bin_count=None):
    """
    Computes the pair distance distribution function for a given set of coordinates of atoms. \n
    The user can either provide a bin precision or the numer of bins depending on wheter they are striving for a specific precision in the bins
    or on a specific number of bins for representation.

    Parameters
    ----------
    index_frame : int
        Index of the frame relative to the snapshot, primarily for reference.
    coords : ndarray
        Array of the coordinates of the atoms forming the system.
    bin_precision: float, optional
        Specify a value if you want to compute the PDDF with a given bin precision (in Angstrom)
    bin_count: int, optional
        Specify a value if you want to compute the PDDF with a given number of bins

    Returns
    -------
    tuple
        - ndarray: the values of the interatomic distances for each bin
        - ndarray: the number of atoms within a given distance for each bin

    """
    n_atoms = np.shape(coords)[0]
    dist_mat, dist_max, dist_min = distance_matrix(
        index_frame=index_frame, coords=coords
    )

    triu_indeces = np.triu_indices(n_atoms, k=1)
    distances = dist_mat[triu_indeces]
    if bin_precision:
        n_bins = int(np.ceil(dist_max / bin_precision))
    elif bin_count:
        n_bins = bin_count
        bin_precision = dist_max / n_bins
    else:
        raise ValueError("You must specify either bin_precision or bin_count.")

    bins = np.linspace(0, dist_max, n_bins + 1)
    dist_count, _ = np.histogram(distances, bins=bins)

    return bins[:-1] + bin_precision / 2, dist_count

    return dist, dist_count



def pddf_calculator_by_element(index_frame, coords, elements, element='Au', bin_precision=None, bin_count=None):
    """
    Computes the pair distance distribution function (PDDF) for a given set of coordinates,
    considering only atoms of a specified chemical element.

    Parameters
    ----------
    index_frame : int
        Index of the frame relative to the snapshot, primarily for reference.
    coords : ndarray
        Array of the coordinates of the atoms forming the system.
    elements : list
        List of atomic species corresponding to each coordinate.
    element: str, optional
        The atomic species to consider (default is 'Au').
    bin_precision: float, optional
        Bin precision in Angstroms.
    bin_count: int, optional
        Number of bins.
    Returns
    -------
    tuple
        - ndarray: the values of the interatomic distances for each bin
        - ndarray: the number of atoms within a given distance for each bin
    """
    # Select only the indices of the atoms corresponding to the specified element

    selected_indices = [i for i, el in enumerate(elements) if el == element]
    selected_coords = coords[selected_indices]

    n_atoms = len(selected_indices)
    if n_atoms < 2:
        raise ValueError("Not enough atoms of the specified element to compute PDDF.")
    
    dist_mat, dist_max, _ = distance_matrix(index_frame=index_frame, coords=selected_coords)
    
    if bin_precision:
        n_bins = dist_max / bin_precision
    else:
        n_bins = bin_count
        bin_precision = dist_max / n_bins
    
    n_bins_int = int(n_bins)
    dist_count = np.zeros(n_bins_int)
    dist = np.zeros(n_bins_int)
    
    for i in range(n_bins_int):
        for j in range(n_atoms):
            for k in range(n_atoms):
                if j != k and (dist_mat[j, k] < bin_precision * i and dist_mat[j, k] >= (bin_precision * (i - 1))):
                    dist_count[i] += 1
        dist[i] = bin_precision * i
    
    return dist, dist_count
    
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
        coords = np.mod(coords - lower_bounds, upper_bounds - lower_bounds) + lower_bounds
    elif box_size.shape == (3,):  # Direct box lengths
        coords = np.mod(coords, box_size)
    else:
        raise ValueError("Box size must be of shape (3,) or (3,2)")

    return coords


def bounding_box(points):
    """Calculate an axis-aligned bounding box from a set of points."""
    x_coordinates, y_coordinates, z_coordinates = zip(*points)
    return np.array([
        [min(x_coordinates), max(x_coordinates)],
        [min(y_coordinates), max(y_coordinates)],
        [min(z_coordinates), max(z_coordinates)],
    ])

def nearest_neighbours(index_frame: int, coords: np.ndarray, cut_off: float, pbc: bool = False, box: np.ndarray = None) -> list:
    """
    Computes nearest neighbors for each atom, considering periodic boundary conditions (PBC) if requested.

    Parameters
    ----------
    index_frame : int
        Number of the frame if a movie.
    coords : ndarray
        XYZ coordinates of atoms, shape (n_atoms, 3).
    cut_off : float
        Cutoff distance for finding neighbors (in Å).
    pbc : bool, optional
        Whether to apply periodic boundary conditions (default: False).
    box : ndarray, optional
        Simulation box size in the form (3,) for orthorhombic boxes or (3,2) for lower and upper bounds.

    Returns
    -------
    list of lists
        Each sublist contains the indices of neighboring atoms for the corresponding atom.
    """
    if pbc:
        if box is None:
            # Estimate bounding box if no box size is provided
            box = bounding_box(coords)
        coords = apply_pbc(coords, box)
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

    # Find neighbors within cutoff
    neigh = neigh_tree.query_ball_point(coords, cut_off)

    # Remove self from neighbor lists
    for i in range(len(neigh)):
        neigh[i] = [neighbor for neighbor in neigh[i] if neighbor != i]

    return neigh



def rdf_calculator(index_frame: int, coords: np.ndarray, cut_off, bin_count: int = None, bin_precision: float = None, box_volume = None):
    """Computes the RDF (Radial Distribution Function) for a system of atoms defined by their coordinates

    Parameters
    ----------
    index_frame : int
        _description_
    coords : np.ndarray
        _description_
    cut_off : _type_
        _description_
    box_size : _type_, optional
        _description_, by default None
    bin_count : int, optional
        _description_, by default None
    bin_precision : float, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    n_atoms = len(coords)

    # Determine binning parameters
    if bin_count is None:
        if bin_precision is not None:
            bin_count = int(cut_off / bin_precision)
        else:
            raise ValueError("Either bin_count or bin_precision must be specified.")
    
    bin_edges = np.linspace(0, cut_off, bin_count + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rdf = np.zeros(bin_count)

    # Build KD-tree and get all unique pairs
    tree = cKDTree(coords)
    pairs = tree.query_pairs(cut_off, output_type="ndarray")  # Unique pairs, excludes self

    # Compute distances only for valid pairs
    distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)

    # Bin distances
    counts, _ = np.histogram(distances, bins=bin_edges)
    rdf += counts * 2  # Multiply by 2 since each pair is counted once

    # Normalize RDF
    shell_volumes = (4 / 3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    if box_volume == None:
        hull = ConvexHull(coords)
        box_volume = hull.volume
    
    number_density = n_atoms / box_volume
    #rdf = rdf / (number_density * shell_volumes * n_atoms)
    rdf = rdf / (number_density * shell_volumes * n_atoms)
    return bin_centers, rdf


def partial_rdf_calculator(
    index_frame: int,
    elements: np.ndarray,
    coords: np.ndarray,
    cut_off: int,
    bin_count: int = None,
    bin_precision: float = None
):
    unique_elements, n_elements = np.unique(elements, return_counts=True)
    rdf_dict = {}
    elements = np.array(elements)
    hull = ConvexHull(coords)
    box_volume = hull.volume
    for el in unique_elements:
        print("el = {} and is of type {}".format(el, type(el)))
        print("elements is of type {}".format(type(elements)))
        idx_el = np.where(elements == str(el))[0]
        coords_el = np.array(coords[idx_el])

        if bin_count:
            d_el, rdf_el = rdf_calculator(index_frame=index_frame, coords=coords_el, cut_off = cut_off, bin_count=bin_count, box_volume = box_volume)
        elif bin_precision:
            d_el, rdf_el = rdf_calculator(index_frame=index_frame, coords=coords_el, cut_off = cut_off, bin_precision=bin_precision, box_volume = box_volume)
        else:
            raise ValueError("Either bin_count or bin_precision must be provided.")

        rdf_dict[el] = [d_el, rdf_el]
    d_tot, rdf_tot = rdf_calculator(index_frame=index_frame, coords=coords, cut_off = cut_off, bin_count=bin_count)
    rdf_dict["Tot"] = [d_tot, rdf_tot]
    return rdf_dict



   




def pair_list(index_frame: int, coords: np.ndarray, cut_off: float, pbc: bool = False, box: np.ndarray = None) -> list:
    """
    Generates list of all pairs of atoms within a certain cutoff distance of each other.

    Parameters
    ----------
    index_frame : int
        Index of the frame if a movie (not used in the current version but can be useful for dynamic systems).
    coords : np.ndarray
        Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
    cut_off : float
        Cutoff distance for finding pairs in angstroms.
    pbc : bool, optional
        Whether to apply periodic boundary conditions. Defaults to False.
    box : np.ndarray, optional
        Simulation box size (either [Lx, Ly, Lz] or [[xmin, xmax], [ymin, ymax], [zmin, zmax]]).

    Returns
    -------
    list
        List of tuples representing pairs of atoms that are within the cutoff distance.
    """
    if pbc:
        if box is None:
            # Estimate bounding box if no box size is provided
            box = bounding_box(coords)
        coords = apply_pbc(coords, box)
        
        # Ensure box is correctly formatted
        if box.shape == (3, 2):  # Lower and upper bounds provided
            box_size = box[:, 1] - box[:, 0]
        elif box.shape == (3,):  # Direct box lengths
            box_size = box
        else:
            raise ValueError("Box must be of shape (3,) or (3,2)")

        # Create KD-tree with periodic boundary conditions
        neigh_tree = cKDTree(coords, boxsize=box_size)
    else:
        # Standard KD-tree without PBC
        neigh_tree = cKDTree(coords)

    # Find all pairs of atoms within the cutoff distance
    pairs = neigh_tree.query_pairs(cut_off)
    
    # Convert pairs to a list of tuples
    pair_list = list(pairs)

    return pair_list



        

def second_neighbours(index_frame: int, coords: np.ndarray, cutoff: float) -> list:
    """
    Compute the second nearest neighbours as the first enighbours of the first neighbours of each atom which are not also first neighbours of the aotm itself. \n
    Computes as the neighbours of j neighbour of i not also in the intersection of neigh[i] and neigh[j].
    
    Parameters:
        index_frame (int): Number of the frame if a movie
        coords (array): array with the XYZ coordinates of the atoms
        cut_off (float): cut off for the neighbours in angstrom
        Returns:
        neigh (list of lists): List of lists of the indeces of the second neighbours of each atom (i-th list of the list of lists is the list of second neighbouring atom indeces of the i-th atom)
    
    """
    neigh = nearest_neighbours(index_frame=index_frame, coords=coords, cut_off=cutoff)
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
            

def coordination_number(index_frame, coords, cut_off, neigh_list = False):
    """
    Computes the coordination number (number of nearest neighbours within a cutoff) for each atom in the system,
    optionally it also returns the neighbour list
    
    Parameters
    ----------
    index_frame : int
        Index of the frame relative to the snapshot, primarily for reference.
    coords : ndarray
        Array of the coordinates of the atoms forming the system.
    cut_off : float
        The cutoff distance for determining nearest neighbors.
    neigh_list : bool, optional
        Option to return the neighbour list as well as the coordination number of each atom (defualt is False)
    
    Returns
    -------
    If neigh_list is True:
        tuple
            - list: neighbour list, the list of indeces of the neighbours of each atom
            - ndarray: the coordination numbers of each atom
    Otherwise:
        - ndarray: the coordination numbers of each atom
    """
    neigh = nearest_neighbours(index_frame=index_frame, coords=coords, cut_off=cut_off)
    n_atoms = np.shape(coords)[0]
    coord_numb = np.zeros(n_atoms)
    for i in range(n_atoms):
        coord_numb[i] = len(neigh[i])
    if neigh_list:
        return neigh, coord_numb
    else:
        return coord_numb

def agcn_calculator(index_frame, coords, cut_off, gcn_max = 12.0):
    """
    """
    neigh_list, coord_numbers = coordination_number(index_frame, coords, cut_off, neigh_list=True)
    n_atoms = len(coord_numbers)
    agcn = np.zeros(n_atoms)
    
    for i, atom_neighbors in enumerate(neigh_list):
        agcn_i = sum(coord_numbers[neigh] for neigh in atom_neighbors)
        agcn[i] = agcn_i / gcn_max
    return agcn           


def bridge_gcn(index_frame: int, coords: np.ndarray, cut_off: float, gcn_max=18.0, phantom=False):
    """
    Identifies bridge absorption sites and computes the Generalized Coordination Number (GCN) 
    for a site. The GCN is defined as the sum of the coordination numbers of the neighbors 
    of the two atoms forming the site, counted only once.

    Parameters
    ----------
    - index_frame : int
        Index of the frame relative to the snapshot, primarily for reference.
    - coords : ndarray
        Array of the coordinates of the atoms forming the system.
    - cut_off : float
        The cutoff distance for determining nearest neighbors.
    - gcn_max : float, optional
        Maximum typical coordination number in the specific system (default is 18.0).
    - phantom : bool, optional
        If True, also returns the coordinates of the midpoints between pairs for 
        representation and testing (default is False).

    Returns
    -------
    If phantom is True:
        tuple
            - ndarray: Coordinates of the midpoints.
            - list: List of pairs.
            - ndarray: Values of the bridge GCN ordered as the pairs.
    Otherwise:
        tuple
            - list: List of pairs.
            - ndarray: Values of the bridge GCN.
    """
    pairs = pair_list(index_frame=index_frame, coords=coords, cut_off=cut_off)
    neigh_list, coord_numb = coordination_number(index_frame=index_frame, coords=coords, cut_off=cut_off, neigh_list=True)
    b_gcn = np.zeros(len(pairs))
    for i, p in enumerate(pairs):
        neigh_1 = neigh_list[p[0]]
        neigh_2 = neigh_list[p[1]]
        neigh_unique_12 = np.unique(np.concatenate((neigh_1, neigh_2)))
        b_gcn_i = sum(coord_numb[neigh] for neigh in neigh_unique_12) - (coord_numb[p[0]] + coord_numb[p[1]])
        b_gcn[i] = b_gcn_i / gcn_max
    if phantom:
        phant_xyz = np.zeros((len(pairs), 3))
        for i, p in enumerate(pairs):
            pos_1 = coords[p[0]]
            pos_2 = coords[p[1]]
            phant_xyz[i] = (pos_1 + pos_2) / 2
        return phant_xyz, pairs, b_gcn
    return pairs, b_gcn
    
    


def center_of_mass(index_frame: int, coords: np.ndarray, elements):
    """
    Calculate the center of mass for a given frame of coordinates.
    
    Parameters:
        index_frame (int): Index of the frame in the trajectory.
        coords (np.ndarray): Array of atomic coordinates of shape (frames, atoms, 3).
        elements (list or np.ndarray): List of element symbols corresponding to the atoms.
    
    Returns:
        np.ndarray: The center of mass as a 3D vector.
    """

    
    # Get the masses of the elements
    masses = np.array([mass[e] for e in elements])
    
    # Compute the total mass
    total_mass = np.sum(masses)
    
    # Calculate the weighted average of coordinates
    com = np.sum(coords * masses[:, None  ], axis=0) / total_mass
    
    return com
    
def geometric_com(index_frame: int, coords: np.ndarray):
    gcom = np.mean(coords, axis = 0)
    return gcom

    
            

    
