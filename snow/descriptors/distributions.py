import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist, squareform


from snow.descriptors.utils import distance_matrix, hetero_distance_matrix, _check_structure
from snow.descriptors.shape_descriptors import center_of_mass, geometric_com

    
def pddf_calculator(coords, bin_width: float, use_lattice_units: bool, lattice=None):
    """
    Computes the pair distance distribution function for a given set of coordinates of atoms. \n
    If use_lattice_units=True, bin_width should be provided in lattice units (alat) and the pddf 
    is returned in lattice units.
    If use_lattice=False, the bin width should be provided in the coordinates units.
    Parameters
    ----------
    coords : ndarray
        Array of the coordinates of the atoms forming the system.
    bin_width: float
        width of the bins to bin the distances in the system. It should be provided in lattice units if
        use_lattice_units==True, and in the same units as coords if use_lattice==False
    use_lattice_units: bool
        If True, the PDDF is returned in units of the lattice consatnt (passed as the 'lattice' argument) and
        the bin_width should be given in units of the lattice constant.
        If False, the PPDF is returned in the units of coords, and the bin_width should be given in the same units
        as coords.
    lattice: float,
        Specify a value for the lattice parameter of your structure in the same units as coords.

    Returns
    -------
    tuple
        - ndarray: the values of the interatomic distances for each bin
        - ndarray: the number of atoms within a given distance for each bin

    """

    if use_lattice_units:

        if lattice is None:
            raise ValueError('If use_lattice_units==True, you should provide a value for the lattice constant to use')
        
        coords = coords/lattice #creates a copy instead of modifying in-place the array passed as argument
    
    #bin_precision=bin_size_lattice*lattice #convert in \AA units the lattice dimension
    _check_structure(coords=coords)
    n_atoms = np.shape(coords)[0]
    
    dist_mat, dist_max, dist_min = distance_matrix(coords=coords)

    triu_indeces = np.triu_indices(n_atoms, k=1)
    distances = dist_mat[triu_indeces]

    n_bins = int(np.ceil(dist_max / bin_width))

    bins = np.linspace(0, n_bins*bin_width, n_bins + 1)
    dist_count, _ = np.histogram(distances, bins=bins)

    return (bins[:-1] + bin_width/2.), dist_count


def pddf_calculator_by_elements(
        coords: np.ndarray,
        species: list,
        elements: list,
        bin_width: float,
        use_lattice_units: bool,
        lattice=None,
        cutoff : float = None,
):
    """
    Computes the pair distance distribution function (PDDF) for a given set of coordinates,
    considering only atoms of a specified chemical element. Uses histogram counting for efficiency.

    Parameters
    ----------
    coords : ndarray
        Array of the coordinates of the atoms forming the system.
    species : list[str]
        List of atomic species corresponding to each coordinate.
    elements: list[str]
        The elements of which to consider the pairs
    bin_width: float
        width of the bins to bin the distances in the system. It should be provided in lattice units if
        use_lattice_units==True, and in the same units as coords if use_lattice==False
    use_lattice_units: bool
        If True, the PDDF is returned in units of the lattice consatnt (passed as the 'lattice' argument) and
        the bin_width should be given in units of the lattice constant.
        If False, the PPDF is returned in the units of coords, and the bin_width should be given in the same units
        as coords.
    lattice: float,
        Specify a value for the lattice parameter of your structure in the same units as coords.
    cutoff: float,
        If specified, only 

    Returns
    -------
    tuple
        - ndarray: the midpoints of the bins (in lattice units)
        - ndarray: the histogram counts of distances
    """


    #some sanity checks
    if use_lattice_units:

        coords /= lattice

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
        dist_mat, dist_max, dist_min = distance_matrix(coords=selected_coords)
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
        dist_mat,_,_ = distance_matrix(coords)

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
    bin_precision: float = None,
    box_volume=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Radial Distribution Function as defined in "Understanding Molecular Simulation" by Frankel and Smit, for each atom concentric shells with
    a certain bin precision (or number of bins) are constructed and the density of atoms found in each shell is computed. 
    
    Note that technically we implemented a periodic version as well but it still is not wokring properly.

    Parameters
    ----------
    elements : np.ndarray
        _description_
    coords : np.ndarray
        XYZ coordinates of atoms, shape (n_atoms, 3).
    cut_off : float
        Cutoff distance for finding pairs in angstroms. If None, an adaptive cutoff is used per atom.
    bin_count : int, optional
        Number of bins, by default None
    bin_precision : float, optional
        Bin precision, by default None
    box_volume = None: float, optional
        Box dimension for PBC (WIP), by default None (no PBC, good for isolated systems)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Binned distance and rdf

    Raises
    ------
    ValueError
        _description_
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
    if box_volume == None:
        hull = ConvexHull(coords)
        box_volume = hull.volume

    number_density = n_atoms / box_volume
    # rdf = rdf / (number_density * shell_volumes * n_atoms)
    rdf = rdf / (number_density * shell_volumes * n_atoms)
    return bin_centers, rdf



def com_rdf_calculator(coords :np.ndarray, 
                       bin_width :float, 
                       com=None, 
                       elements:list = None):
    """
    Compute the Radial Distribution Function: a distribution of all the distances wrt to the center
    of mass of the system. The com can be provided as an argument or computed by the function

    ----------
    Parameters
    coords: ndarray
        coordinates of atoms in the system
    bin_width: float
        bin width for binning of the distribution
    elements: ndarray, optional
        chemical species of the atoms in the system used in the center of mass calculation. If None,
        provide the com as an argument to the function
    com: ndarray, optional
        center of mass of the system. If None (default), it is computed
    """

    #compute com if not provided
    if elements is None and com is None:
        raise ValueError('Provide either the list of elements or the center of mass of your system as an argument to the RDF function')

    if com is None:
        com = center_of_mass(coords, elements)
    
    #obtain the list of distances of each atom to the com
    com_dists = np.zeros(len(coords))
    for i, pos in enumerate(coords):
        com_dists[i] = np.linalg.norm(pos - com)
    
    dist_max = np.max(com_dists)

    n_bins = int(np.ceil(dist_max / bin_width))

    bins = np.linspace(0, n_bins*bin_width, n_bins + 1)
    dist_count, _ = np.histogram(com_dists, bins=bins)

    return (bins[:-1] + bin_width/2.), dist_count
