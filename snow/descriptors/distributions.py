import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from snow.descriptors.utils import distance_matrix, _check_structure
from snow.descriptors.shape_descriptors import center_of_mass
from snow.misc.rototranslation import align_axis_to_z

    
def pddf_calculator(coords, bin_width: float, use_lattice_units: bool, lattice : float = None):
    """
    Computes the pair distance distribution function for a given set of coordinates of atoms. 
    Please note that this function will count each pair once e.g. will consider (i,j) but not (j,i)

    If use_lattice_units=True, bin_width should be provided in lattice units (alat) and the pddf 
    is returned in lattice units.
    If use_lattice_units=False, the bin width should be provided in the coordinates units.

    Parameters
    ----------
    coords : np.ndarray
        Array of the coordinates of the atoms forming the system.
    bin_width : float
        width of the bins to bin the distances in the system. It should be provided in lattice units if
        use_lattice_units==True, and in the same units as coords if use_lattice==False
    use_lattice_units : bool
        If True, the PDDF is returned in units of the lattice constant (passed as the 'lattice' argument) and
        the bin_width should be given in units of the lattice constant.
        If False, the PDDF is returned in the units of coords, and the bin_width should be given 
        in the same units as coords.
    lattice : float, optional
        Specify a value for the lattice parameter of your structure in the same units as coords.
        Only needed if use_lattice_units is True

    Returns
    -------
    bin_centers : np.ndarray
        the values of the interatomic distances corresponding to each bin
    dist_count : np.ndarray
        the count of distances for the each bin

    """

    if use_lattice_units:

        if lattice is None:
            raise ValueError('If use_lattice_units==True, you should provide a value for the lattice constant to use')
        
        coords = coords/lattice #creates a copy instead of modifying in-place the array passed as argument
    
    #bin_precision=bin_size_lattice*lattice #convert in \AA units the lattice dimension
    _check_structure(coords=coords)
    n_atoms = np.shape(coords)[0]
    
    dist_mat = distance_matrix(coords=coords)
    dist_max = np.max(dist_mat)

    triu_indeces = np.triu_indices(n_atoms, k=1)
    distances = dist_mat[triu_indeces]

    n_bins = int(np.ceil(dist_max / bin_width))

    bins = np.linspace(0, n_bins*bin_width, n_bins + 1)
    dist_count, _ = np.histogram(distances, bins=bins)

    return (bins[:-1] + bin_width/2.), dist_count


def pddf_calculator_by_elements(
        species: list,
        coords: np.ndarray,
        elements: list,
        bin_width: float,
        use_lattice_units: bool,
        lattice : float = None,
        cutoff : float = None,
):
    """
    Computes the chemical element-wise pair distance distribution function (PDDF) for a given set of coordinates.
    Please note that this function will count each pair once e.g. will consider (i,j) but not (j,i)

    This function only considers distances between atoms of specified chemical elements (A-A, A-B, or B-B).
    It can be decided whether to use lattice units or not. Histogram counting is used for efficiency.

    Parameters
    ----------
    species : list[str]
        List of atomic species corresponding to each coordinate.
    coords : ndarray
        Array of the coordinates of the atoms forming the system.
    elements : list[str]
        The elements of which to consider the pairs 
        (i.e. [A,A], or [A,B], or [B,B], given A and B two chemical species in your system)
    bin_width : float
        width of the bins to bin the distances in the system. It should be provided in lattice units if
        use_lattice_units==True, and in the same units as coords if use_lattice==False
    use_lattice_units : bool
        If True, the PDDF is computed and returned in units of the lattice constant (passed as the 'lattice' argument) and
        the bin_width should be given in units of the lattice constant.
        If False, the PPDF is returned in the units of coords, and the bin_width should be given in the same units
        as coords.
    lattice: float, optional
        Specify a value for the lattice parameter of your structure in the same units as coords. Only needed if use_lattice_units is set to True
    cutoff: float, optional
        If specified, only distances up to this value are taken into account for the histogram calculation

    Returns
    -------
    bin_centers : np.ndarray
        the values of the interatomic distances corresponding to each bin
    dist_count : np.ndarray
        the count of distances for the each bin
    """


    #some sanity checks
    if use_lattice_units:
        coords = coords / lattice

        if lattice is None:
            raise ValueError('If use_lattice_units==True, you should provide a value for the lattice constant to use')

    if elements[0] == elements[1]:
        #Specialized distance matrix method
        
        element = elements[0]
        # Check structure and select only atoms of the given element
        #_check_structure(coords=coords, species=species)
        selected_indices = [i for i, el in enumerate(species) if el == element]
        selected_coords = coords[selected_indices]

        n_atoms = len(selected_indices)

        if n_atoms < 2:
            raise ValueError(
                f"Not enough atoms of element '{element}' to compute PDDF."
            )

        # Compute distance matrix
        dist_mat = distance_matrix(coords=selected_coords)
        dist_max = np.max(dist_mat)
        if cutoff:
            n_bins = int(np.ceil(cutoff/ bin_width))
        else:
            n_bins = int(np.ceil(dist_max / bin_width))

        # Extract upper triangle (j < k)
        triu_indices = np.triu_indices(n_atoms, k=1)
        distances = dist_mat[triu_indices]

        # Compute histogram
        bins = np.linspace(0, n_bins*bin_width, n_bins + 1)
        dist_count, _ = np.histogram(distances, bins=bins)

        # Bin midpoints
        bin_centers = (bins[:-1] + bins[1:])/2.

        return bin_centers, dist_count
    else:
        # Heteroelemental
        # Only take distances of given pair
        dist_mat = distance_matrix(coords)

        #The Mask will zero distances that are not the ones we are looking for
        mask = np.zeros((len(coords),len(coords)))
        for i in range(len(coords)):
            for j in range(len(coords)):
                elements_ij = [species[i],species[j]]
                #Now check that the pair is of the two elements, no matter the order
                if (elements[0] in elements_ij) and (elements[1] in elements_ij):
                    mask[i,j] = 1

        dist_mat *= mask
        dist_max = np.max(dist_mat)

        if cutoff:
            n_bins = int(np.ceil(cutoff/ bin_width))
        else:
            n_bins = int(np.ceil(dist_max / bin_width))
        # Extract upper triangle (j < k)
        triu_indices = np.triu_indices(len(coords), k=0)
        distances = dist_mat[triu_indices]

        # Compute histogram
        bins = np.linspace(0, n_bins*bin_width, n_bins + 1)
        dist_count, _ = np.histogram(distances[distances > 0], bins=bins)

        # Bin midpoints
        bin_centers = (bins[:-1] + bins[1:])/2.

        return bin_centers, dist_count
    return

        


def gdr_notnorm_calculator(
    coords: np.ndarray,
    cut_off: float,
    bin_count: int = None,
    bin_precision: float = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the (unnormalized) Radial Distribution Function as defined in "Understanding Molecular Simulation" by Frenkel and Smit, for each atom concentric shells with
    a certain bin precision (or number of bins) are constructed and the density of atoms found in each shell is computed. 
    
    Parameters
    ----------
    coords : np.ndarray
        XYZ coordinates of atoms, shape (n_atoms, 3).
    cut_off : float
        Cutoff distance for finding pairs in angstroms.
    bin_count : int, optional
        Number of bins, by default None. Either bin_count or bin_precision should be specified.
    bin_precision : float, optional
        Bin precision, by default None. Either bin_count or bin_precision should be specified.

    Returns
    -------
    bin_centers : np.ndarray
        the values of the interatomic distances corresponding to each bin
    rdf : np.ndarray
        unnormalized g(r) values

    Raises
    ------
    ValueError
        If neither bin_count nor bin_precision was specified.
    """
    _check_structure(coords=coords)

    n_atoms = len(coords)

    # Determine binning parameters
    if bin_count is None:
        if bin_precision is not None:
            bin_count = int(cut_off / bin_precision)
        else:
            raise ValueError(
                "Either bin_count or bin_precision must be specified."
            )

    bin_edges = np.linspace(0, cut_off, bin_count + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rdf = np.zeros(bin_count)

    # Build KD-tree and get all unique pairs
    tree = cKDTree(coords)
    pairs = tree.query_pairs(
        cut_off, output_type="ndarray"
    )  # Unique pairs, excludes self

    # Compute distances only for valid pairs
    distances = np.linalg.norm(
        coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1
    )

    # Bin distances
    counts, _ = np.histogram(distances, bins=bin_edges)
    rdf += counts * 2  # Multiply by 2 since each pair is counted once

    # Normalize RDF
    shell_volumes = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    # if box_volume == None:
    #     hull = ConvexHull(coords)
    #     box_volume = hull.volume
    hull = ConvexHull(coords)
    box_volume = hull.volume

    number_density = n_atoms / box_volume
    rdf = rdf / (number_density * shell_volumes * n_atoms)
    return bin_centers, rdf



def com_rdf_calculator(coords : np.ndarray, 
                       bin_width : float, 
                       com : np.ndarray = None, 
                       elements : list = None):
    """
    Compute the Radial Distribution Function: a distribution of all the distances wrt to the center
    of mass of the system. The com can be provided as an argument or computed by the function (in this case, 
    pass the list of chemical elements in your system as an argument)

    Parameters
    ----------
    coords : np.ndarray
        coordinates of atoms in the system
    bin_width : float
        bin width for binning of the distribution
    com : np.ndarray, optional
        center of mass of the system (as a three-elements coordinates array). If None (default), it is computed
    elements : list[str], optional
        chemical species of the atoms in the system used in the center of mass calculation. If None,
        provide the com as an argument to the function

    Returns
    -------
    bin_centers : np.ndarray
        the values of the interatomic distances corresponding to each bin
    dist_count : np.ndarray
        the count of distances for the each bin

    Raises
    ------
    ValueError
        If neither the list of elements nor the center of mass was specified.
    """

    #compute com if not provided
    if elements is None and com is None:
        raise ValueError('Provide either the list of elements or the center of mass of your system as an argument to the RDF function')

    if com is None:
        com = center_of_mass(elements, coords)
    
    #obtain the list of distances of each atom to the com
    com_dists = np.zeros(len(coords))
    for i, pos in enumerate(coords):
        com_dists[i] = np.linalg.norm(pos - com)
    
    dist_max = np.max(com_dists)

    n_bins = int(np.ceil(dist_max / bin_width))

    bins = np.linspace(0, n_bins*bin_width, n_bins + 1)
    dist_count, _ = np.histogram(com_dists, bins=bins)

    return (bins[:-1] + bin_width/2.), dist_count


def cut_layers(
    elements: np.ndarray,
    coords_frame: np.ndarray,
    layer_height: float,
    cutting_ax = 'z',
    species_A: str = None,
    species_B: str = None,
    
):
    """
    Cuts a single frame into layers and compute the distribution of atoms in the layers.

    Computes the distribution of atoms per layer of width `layer_height`. The axis along which (perpendicular)
    planes are cut can be specified as either 'z' (default), 'x', 'y', or a user-defined np.ndarray

    Parameters
    ----------
    elements : np.ndarray
        chemical symbols of the atoms provided - Shape (n_atoms,)
    coords_frame : np.ndarray
        coordinates of the atoms provided - Shape (n_atoms, 3)
    cutting_ax : str or np.ndarray
        either 'x', 'y', 'z', or a (3, ) np.ndarray such as (1,1,0)
    species_A : str (optional)
        chemical specie 1 to filter the coords and get a chemical specie-wise count of atoms per layer
    species_B : str (optional)
        chemical specie 2 to filter the coords and get a chemical specie-wise count of atoms per layer
        
    Returns
    -------
    layer_number : np.ndarray
        Layer indices, shape (n_layers,).
    layer_ntot : np.ndarray
        Total atom count per layer, shape (n_layers,).
    layer_na : np.ndarray
        Atom count per layer for species_A, shape (n_layers,).
        Only returned if species_A is not None.
    layer_nb : np.ndarray
        Atom count per layer for species_B, shape (n_layers,).
        Only returned if species_B is not None.
    """

    if cutting_ax == 'x':
        cutting_ax = np.asarray([1.,0.,0.])
    elif cutting_ax == 'y':
        cutting_ax = np.asarray([0.,1.,0.])
    elif cutting_ax == 'z':
        cutting_ax = np.asarray([0.,0.,1.])

    elements = np.array(elements, dtype=str)  # dtype+np.ndarray conversion
    elements = np.char.strip(elements)        # remove whitespaces
            
    #align selected axis to z
    if not np.array_equal(cutting_ax, np.array([0., 0., 1.])):
        cc = align_axis_to_z(coords_frame, axis=cutting_ax)
    else:
        cc = coords_frame

    z = cc[:,2]
    
    min_z = z.min()
    max_z = z.max()
    
    n_layers = int((max_z - min_z) / layer_height) + 1

    layer_number = np.zeros(n_layers, dtype=int) #or layer bin center?
    layer_ntot   = np.zeros(n_layers, dtype=int)
    if species_A is not None:
        layer_na     = np.zeros(n_layers, dtype=int)
    if species_B is not None:
        layer_nb     = np.zeros(n_layers, dtype=int)

    for i in range(n_layers):
    
        z_min_bin = min_z + i * layer_height
        z_max_bin = min_z + (i + 1) * layer_height

        mask = (z >= z_min_bin) & (z < z_max_bin)
        
        tot = np.count_nonzero(mask)
        if species_A is not None:
            n_A = np.count_nonzero(mask & (elements == species_A))
        if species_B is not None:
            n_B = np.count_nonzero(mask & (elements == species_B))
        
        layer_number[i] = i
        layer_ntot[i]   = tot
        if species_A is not None:
            layer_na[i]     = n_A
        if species_B is not None:
            layer_nb[i]     = n_B

    if species_A is not None and species_B is not None:
        return layer_number, layer_ntot, layer_na, layer_nb
    elif species_A is not None:
        return layer_number, layer_ntot, layer_na
    elif species_B is not None:
        return layer_number, layer_ntot, layer_nb
    else:
        return layer_number, layer_ntot

