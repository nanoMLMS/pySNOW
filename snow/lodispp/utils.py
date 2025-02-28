
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree, ConvexHull
from snow.misc.constants import mass
from scipy.sparse import coo_matrix
import os 

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

    
            

    

def LAE_xyz(index_frame: int, coords: np.ndarray, elements, cut_off):
    # Trova i vicini più prossimi per ogni atomo
    nearest_neigh = nearest_neighbours(index_frame, coords, cut_off)

    CN_list = []
    num_atom_same_species = []
    num_atom_other_species = []

    # Loop sugli atomi
    for j in range(len(elements)):
        CN = 0
        same = 0
        other = 0

        #print(elements[j])
        # Itera sui vicini di j
        for k in nearest_neigh[j]:
            CN += 1  # Incrementa il numero totale di vicini
            
            print(elements[j], elements[k])
            if elements[j] == elements[k]:
                same += 1  # Stesso elemento
            else:
                other += 1  # Elemento diverso
        
        print(same, other)
        CN_list.append(CN)
        num_atom_same_species.append(same)
        num_atom_other_species.append(other)

    return CN_list, num_atom_same_species, num_atom_other_species




def LAE_write_xyz_file(filename, elements, coords, lae_output, frame):
    if frame == 0 and os.path.exists(filename):
        os.remove(filename)

    if frame == 0:
        print("el, x, y, z, CN, num_atom_same_species, num_atom_other_species\n")

    with open(filename, "a") as f:
        num_atoms = len(elements)
        f.write(f"{num_atoms}\n\n")

        for i in range(num_atoms):
            x, y, z = coords[i]
            lae1, lae2, lae3= lae_output[0][i], lae_output[1][i], lae_output[2][i]
            f.write(f"{elements[i]} {x:.6f} {y:.6f} {z:.6f} {lae1} {lae2} {lae3}\n")


def LAE_Surf_write_xyz_file(index_frame: int, filename, elements, coords, lae_output, frame, cut_off):
    agcn_array = agcn_calculator(index_frame, coords, cut_off, gcn_max=12.0)

    if frame == 0 and os.path.exists(filename):
        os.remove(filename)

    if frame == 0:
        print("el, x, y, z, num_atom_same_species, num_atom_other_species, surface\n")

    with open(filename, "a") as f:
        num_atoms = len(elements)
        f.write(f"{num_atoms}\n\n")

        for i in range(num_atoms):
            x, y, z = coords[i]
            lae1, lae2 = lae_output[0][i], lae_output[1][i]

            # Identify surface atom
            surface = 1 if agcn_array[i] <= 8.5 else 0

            f.write(f"{elements[i]} {x:.6f} {y:.6f} {z:.6f} {lae1} {lae2} {surface}\n")

    
def LAE_count(index_frame: int, coords: np.ndarray, elements, cut_off, elem_1, elem_2, range_lower_bin, range_upper_bin, savepath):
    nearest_neigh = nearest_neighbours(index_frame, coords, cut_off)
    
    elements_nn = [[elements[el] for el in i] for i in nearest_neigh]
    
    num_atom_same_species = []
    num_atom_other_species = []
    num_NN = []
    
    for j, elem in enumerate(elements_nn):
        same = sum(1 for i in elem if i == elements[j])
        other = len(elem) - same
        num_atom_same_species.append(same)
        num_atom_other_species.append(other)
        num_NN.append(len(elem))
    
    results = {elem_1: [0] * len(range_lower_bin), elem_2: [0] * len(range_lower_bin)}
    
    for r in range(len(range_lower_bin)):
        for k, e in enumerate(elements):
            if range_lower_bin[r] <= num_atom_other_species[k] and num_atom_other_species[k]<= range_upper_bin[r]:
                results[e][r] += 1
    
    filename = f'{savepath}/LAE_count_results_{elem_1}.csv'
    with open(filename, 'a') as f:
    
            
        if index_frame==0:
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
            
        
        for elem in [elem_1, elem_2]:
            f.write(f'{elem},{index_frame}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
    
    
    filename = f'{savepath}/LAE_count_results_{elem_2}.csv'
    with open(filename, 'a') as f:
    
            
        if index_frame==0:
            f.write('Element, frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
            
        
        for elem in [elem_2]:
            f.write(f'{elem},{index_frame}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
            
            
    print(f'Results saved in {filename}')

        


def LAE_surf_count(index_frame: int, coords: np.ndarray, elements, cut_off, elem_1, elem_2, range_lower_bin, range_upper_bin, savepath):
    nearest_neigh = nearest_neighbours(index_frame, coords, cut_off)
    agcn_array = agcn_calculator(index_frame, coords, cut_off, gcn_max = 12.0)
    
    #print(agcn_array, len(agcn_array))
    elements_nn = [[elements[el] for el in i] for i in nearest_neigh]
    
    num_atom_same_species = []
    num_atom_other_species = []
    num_NN = []
    
    for j, elem in enumerate(elements_nn):
        same = sum(1 for i in elem if i == elements[j])
        other = len(elem) - same
        num_atom_same_species.append(same)
        num_atom_other_species.append(other)
        num_NN.append(len(elem))
    
    results = {elem_1: [0] * len(range_lower_bin), elem_2: [0] * len(range_lower_bin)}
    

    for r in range(len(range_lower_bin)):
        n_surf=0
        for k, e in enumerate(elements):
            if agcn_array[k] <= 8.5:  # Consider only surface atoms
                n_surf=n_surf+1
                
                #print('array: ', agcn_array[k] )
                if range_lower_bin[r] <= num_atom_other_species[k] <= range_upper_bin[r]:
                    results[e][r] += 1
                    
        #print('n_surf ', n_surf)
            
    filename = f'{savepath}/LAE_count_results_surface_{elem_1}.csv'
    with open(filename, 'a') as f:
        if index_frame==0:
                
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
        
        for elem in [elem_1]:
            f.write(f'{elem},{index_frame}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
    
    
    
    filename = f'{savepath}/LAE_count_results_surface_{elem_2}.csv'
    with open(filename, 'a') as f:
        if index_frame==0:
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
        
        for elem in [elem_2]:
            f.write(f'{elem},{index_frame+1}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
            
            
    print(f'Surface results saved in {filename}')



def LAE_inner_count(index_frame: int, coords: np.ndarray, elements, cut_off, elem_1, elem_2, range_lower_bin, range_upper_bin, savepath):
    nearest_neigh = nearest_neighbours(index_frame, coords, cut_off)
    agcn_array = agcn_calculator(index_frame, coords, cut_off, gcn_max = 12.0)
    
    #print(agcn_array, len(agcn_array))
    elements_nn = [[elements[el] for el in i] for i in nearest_neigh]
    
    num_atom_same_species = []
    num_atom_other_species = []
    num_NN = []
    
    for j, elem in enumerate(elements_nn):
        same = sum(1 for i in elem if i == elements[j])
        other = len(elem) - same
        num_atom_same_species.append(same)
        num_atom_other_species.append(other)
        num_NN.append(len(elem))
    
    results = {elem_1: [0] * len(range_lower_bin), elem_2: [0] * len(range_lower_bin)}
    

    for r in range(len(range_lower_bin)):
        n_inner=0
        for k, e in enumerate(elements):
            if agcn_array[k] > 8.5:  # Consider only surface atoms
                n_inner=n_inner+1
                
                #print('array: ', agcn_array[k] )
                if range_lower_bin[r] <= num_atom_other_species[k] <= range_upper_bin[r]:
                    results[e][r] += 1
                    
        #print('n_inner ', n_inner)
            
    filename = f'{savepath}/LAE_count_results_inner_{elem_1}.csv'
    with open(filename, 'a') as f:
        if index_frame==0:
                
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
        
        for elem in [elem_1]:
            f.write(f'{elem},{index_frame}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
    
    
    
    filename = f'{savepath}/LAE_count_results_inner_{elem_2}.csv'
    with open(filename, 'a') as f:
        if index_frame==0:
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
        
        for elem in [elem_2]:
            f.write(f'{elem},{index_frame+1}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
            
            
    #print(f'Inner results saved in {filename}')




def progress_bar(current, total, length=50):
    """
    Prints a nice progess bar to make look like it is doing something
    """
    percent = current / total
    filled_length = int(length * percent)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r[{bar}] {percent * 100:.2f}%', end='')
    return



def three_hollow_gcn(index_frame: int, coords: np.ndarray, cut_off: float, thr_cn: int, dbulk: list[float],
                     gcn_max: float = 22.0, strained: bool = False) -> list:
    """
    Finds the location of three-hollow sites and returns their location and GCN
    Parameters
    ----------
        index_frame : int
            Number of the frame if a movie.
        coords: ndarray
            Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
        cut_off : float
            Cutoff distance for finding neighbors in angstrom.
        thr_cn : int
            An atom is considered in the surface if its CN < thr_cn
        gcn_max:
            GCN max in the formula for computing the GCN
        strained:
        dbulk

    Returns
    -------
        sites : list
            Midpoint of triplets that form a three hollow site
        th_gcn: list
            GCN of the three hollow sites

    """
    triplets = []
    sites = []
    th_gcn = []
    pairs = pair_list(index_frame=index_frame, coords=coords, cut_off=cut_off)
    # neighbor list and coordination number not compatible!
    neigh_list, coord_numb = coordination_number(index_frame=index_frame, coords=coords, cut_off=cut_off,
                                                 neigh_list=True)
    for i, p in enumerate(pairs):
        progress_bar(i, len(pairs) - 1, 50)
        # we check that both are at the surface
        # thr_cn is a threshold on cn to check if we are at surface
        if not (coord_numb[p[0]] < thr_cn and coord_numb[p[1]] < thr_cn):
            continue
        neigh_1 = neigh_list[p[0]]
        neigh_2 = neigh_list[p[1]]
        neigh_unique_12 = np.intersect1d(neigh_1, neigh_2, assume_unique=True)
        for cn in neigh_unique_12:
            if cn != p[0] and cn != p[1] and coord_numb[cn] < thr_cn:
                new_triplet = sorted([p[0], p[1], cn])
                if not new_triplet in triplets:  # this is a new surface triplet
                    triplets.append(new_triplet)
                    sites.append((coords[p[0]] + coords[p[1]] + coords[cn]) / 3)
                    neigh_unique_triplet = []
                    for idx in new_triplet:
                        neigh_unique_triplet += neigh_list[idx]
                    neigh_unique_triplet = np.unique(neigh_unique_triplet)
                    if strained:
                        sgcn = 0
                        for nb in neigh_unique_triplet:
                            for nnb in neigh_list[nb]:
                                d_nb_nnb = np.linalg.norm(coords[nb] - coords[nnb])
                                sgcn += dbulk / d_nb_nnb
                        self_sgcn = 0
                        for nb in new_triplet:
                            for nnb in neigh_list[nb]:
                                d_nb_nnb = np.linalg.norm(coords[nb] - coords[nnb])
                                self_sgcn += dbulk / d_nb_nnb
                        th_gcn.append((sgcn - self_sgcn) / gcn_max)
                    else:
                        th_gcn_i = sum(coord_numb[neigh] for neigh in neigh_unique_triplet) - sum(
                            coord_numb[neigh] for neigh in new_triplet)
                        th_gcn.append(th_gcn_i / gcn_max)

    print("\nDone three hollow")
    return sites, th_gcn


def four_hollow_gcn(index_frame: int, coords: np.ndarray, cut_off: float, thr_cn: int, dbulk: list[float],
                    gcn_max: float = 26.0, strained: bool = False) -> list:
    """
    Finds the location of four-hollow sites and returns their location and GCN
    Parameters
    ----------
        index_frame : int
            Number of the frame if a movie.
        coords: ndarray
            Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
        cut_off : float
            Cutoff distance for finding neighbors in angstrom.
        thr_cn : int
            An atom is considered in the surface if its CN < thr_cn
        gcn_max:
            GCN max in the formula for computing the GCN

    Returns
    -------
        sites : list
            Midpoint of triplets that form a four hollow site
        th_gcn: list
            GCN of the four hollow sites

    """
    fours = []
    sites = []
    fh_gcn = []
    pairs = pair_list(index_frame=index_frame, coords=coords, cut_off=cut_off)
    # neighbor list and coordination number not compatible!
    neigh_list, coord_numb = coordination_number(index_frame=index_frame, coords=coords, cut_off=cut_off,
                                                 neigh_list=True)
    snb, _ = coordination_number(index_frame=index_frame, coords=coords, cut_off=cut_off * 1.3, neigh_list=True)
    current = 0
    for j in range(len(pairs)):
        for k in range(j):
            progress_bar(current, len(pairs) * (len(pairs) - 1) / 2)
            current += 1
            indices = [pairs[j][0], pairs[j][1], pairs[k][0], pairs[k][1]]
            check = False
            for idx in indices:  # check if all atoms are in surface
                if coord_numb[idx] >= thr_cn:
                    check = True
                # print(coord_numb[idx],thr_cn,check)
            if check:
                continue
            check = len(set(indices))
            if check != 4:
                continue  # got a shared atom
            # check if pairs are common atmostsecond neighbors
            common_nb_j = np.intersect1d(snb[indices[0]], snb[indices[1]], assume_unique=True)
            if (indices[2] in common_nb_j) and (indices[3] in common_nb_j):
                new_fours = sorted(indices)
                if not new_fours in fours:
                    fours.append(new_fours)
                    newsite = np.array([0., 0., 0.])
                    for at in new_fours:
                        newsite += coords[at]
                    sites.append(newsite / 4.0)
                    neigh_unique_four = []
                    for idx in new_fours:
                        neigh_unique_four += neigh_list[idx]
                    neigh_unique_four = np.unique(neigh_unique_four)
                    if strained:
                        sgcn = 0
                        for nb in neigh_unique_four:
                            for nnb in neigh_list[nb]:
                                d_nb_nnb = np.linalg.norm(coords[nb] - coords[nnb])
                                sgcn += dbulk / d_nb_nnb
                        self_sgcn = 0
                        for nb in new_fours:
                            for nnb in neigh_list[nb]:
                                d_nb_nnb = np.linalg.norm(coords[nb] - coords[nnb])
                                self_sgcn += dbulk / d_nb_nnb
                        fh_gcn.append((sgcn - self_sgcn) / gcn_max)
                    else:
                        fh_gcn_i = sum(coord_numb[neigh] for neigh in neigh_unique_four) - sum(
                            coord_numb[neigh] for neigh in new_fours)
                        fh_gcn.append(fh_gcn_i / gcn_max)
    # for f in fours[:5]:
    # print(coord_numb[f[0]],coord_numb[f[1]],coord_numb[f[2]],coord_numb[f[3]])
    print("\nDone four hollow")
    return sites, fh_gcn



