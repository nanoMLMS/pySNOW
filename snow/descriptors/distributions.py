import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist, squareform


from snow.descriptors.utils import distance_matrix, hetero_distance_matrix, _check_structure
from snow.descriptors.shape import center_of_mass, geometric_com

    
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


def pddf_calculator_by_element(
    coords,
    elements,
    element,
    bin_width,
    use_lattice_units: bool,
    lattice=None,
):
    """
    Computes the pair distance distribution function (PDDF) for a given set of coordinates,
    considering only atoms of a specified chemical element. Uses histogram counting for efficiency.

    Parameters
    ----------
    coords : ndarray
        Array of the coordinates of the atoms forming the system.
    elements : list
        List of atomic species corresponding to each coordinate.
    element: str
        The atomic species to consider.
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
        - ndarray: the midpoints of the bins (in lattice units)
        - ndarray: the histogram counts of distances
    """

    # Check structure and select only atoms of the given element
    _check_structure(coords=coords, elements=elements)
    selected_indices = [i for i, el in enumerate(elements) if el == element]
    selected_coords = coords[selected_indices]

    n_atoms = len(selected_indices)

    #some sanity checks
    if use_lattice_units:

        if lattice is None:
            raise ValueError('If use_lattice_units==True, you should provide a value for the lattice constant to use')
        
        selected_coords /= lattice

    if n_atoms < 2:
        raise ValueError(
            f"Not enough atoms of element '{element}' to compute PDDF."
        )

    # Compute distance matrix
    dist_mat, dist_max, dist_min = distance_matrix(coords=selected_coords)

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
    

#TODO
def pddf_as_window_function(coords, lattice, bin_size_lattice, d0):
    """
    Computes the WF order parameter as defined by Pavan et al. (2015)

    Parameters
    ----------
    coords : ndarray
        Atomic coordinates
    lattice: float,
        Specify a value for the Bulk lattice constant of your structure (in Angstrom) 
    bin_size_lattice: int
        Specify a value for the size of PDDF binning in units of the lattice parameter.
    d0 : float
        Characteristic distance of the window (Å)

    Returns
    -------
    float
        Window function order parameter
    """

    r0 = bin_size_lattice * lattice  # window width

    n_atoms = coords.shape[1]
    dist_mat, _, _ = distance_matrix(coords=coords)

    triu_indices = np.triu_indices(n_atoms, k=1)
    rij = dist_mat[triu_indices]

    x = (rij - d0) / r0

    wf = np.sum((1 - x**6) / (1 - x**12))

    return wf
    
    

# def pddf_calculator_old(coords, bin_precision=None, bin_count=None):
#     """
#     Computes the pair distance distribution function for a given set of coordinates of atoms. \n
#     The user can either provide a bin precision or the numer of bins depending on wheter they are striving for a specific precision in the bins
#     or on a specific number of bins for representation.

#     Parameters
#     ----------
#     coords : ndarray
#         Array of the coordinates of the atoms forming the system.
#     bin_precision: float, optional
#         Specify a value if you want to compute the PDDF with a given bin precision (in Angstrom)
#     bin_count: int, optional
#         Specify a value if you want to compute the PDDF with a given number of bins

#     Returns
#     -------
#     tuple
#         - ndarray: the values of the interatomic distances for each bin
#         - ndarray: the number of atoms within a given distance for each bin

#     """
#     _check_structure(coords=coords)
#     n_atoms = np.shape(coords)[0]
#     dist_mat, dist_max, dist_min = distance_matrix(coords=coords)

#     triu_indeces = np.triu_indices(n_atoms, k=1)
#     distances = dist_mat[triu_indeces]
#     if bin_precision:
#         n_bins = int(np.ceil(dist_max / bin_precision))
#     elif bin_count:
#         n_bins = bin_count
#         bin_precision = dist_max / n_bins
#     else:
#         raise ValueError("You must specify either bin_precision or bin_count.")

#     bins = np.linspace(0, dist_max, n_bins + 1)
#     dist_count, _ = np.histogram(distances, bins=bins)

#     return bins[:-1] + bin_precision / 2, dist_count


def hetero_pddf_calculator(
    coords: np.ndarray, elements: np.ndarray, bin_precision: float | None = None, bin_count: int | None = None
):
    """Computes the PDDF only for atoms with different chemical species.

    Parameters
    ----------
    coords : ndarray
        Array of the coordinates of the atoms forming the system.
    elements: ndarray
        Array of elements (must have same order for atoms as they appear in coords array)
    bin_precision : float, optional
        Specify a value if you want to compute the PDDF with a given bin precision (in Angstrom)
    bin_count : int, optional
        Specify a value if you want to compute the PDDF with a given number of bins


    Returns
    -------
    r_centers : ndarray
        Bin center positions.
    dist_count : ndarray
        Counts per bin.

    Raises
    ------
    ValueError
        If input shapes are inconsistent or binning is ill-defined.
    """

    _check_structure(coords=coords, elements=elements)
    n_atoms = np.shape(coords)[0]

    dist_mat = hetero_distance_matrix(
        elements=elements, coords=coords
    )

    #retrieve only upper triangular to avoid double counting automatically
    triu_indices = np.triu_indices(n_atoms, k=1)
    distances = dist_mat[triu_indices]

    # Exclude zero distances
    distances = distances[distances > 0]

    if distances.size == 0:
        raise ValueError(
            "No valid nonzero distances found. Check your input data."
        )

    dist_max = np.max(distances)

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


def chemical_pddf_calculator( #update to new pddf style
    coords, elements, bin_width: float, use_lattice_units: bool, lattice=None
):
    """
    Computes the chemical element-wise pair distance distribution function for a given set of coordinates of atoms. \n
    The user can either provide a bin precision or the numer of bins depending on wheter they are striving for a specific precision in the bins
    or on a specific number of bins for representation.
    The function returns a dict witht the pddf for each chemical specie.

    Parameters
    ----------
    coords : ndarray
        Array of the coordinates of the atoms forming the system.
    elements: ndarray
        Array of chemical elements corresponding to the coords.
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
    _check_structure(coords=coords, elements=elements)

    unique_elements, n_elements = np.unique(elements, return_counts=True)

    pddf_dict = {}
    elements = np.array(elements)

    d_tot, pddf_tot = pddf_calculator(coords, bin_width, use_lattice_units, lattice)

    pddf_dict["Tot"] = [d_tot, pddf_tot]

    for el in unique_elements:

        idx_el = np.where(elements == str(el))[0]
        coords_el = np.array(coords[idx_el])

        d_el, pddf_el = pddf_calculator(coords, bin_width, use_lattice_units, lattice)

        pddf_dict[el] = [d_el, pddf_el]

    return pddf_dict


# def pddf_calculator_by_element_old(
#     coords,
#     elements,
#     element,
#     bin_precision=None,
#     bin_count=None,
# ):
#     """
#     Computes the pair distance distribution function (PDDF) for a given set of coordinates,
#     considering only atoms of a specified chemical element.

#     Parameters
#     ----------
#     coords : ndarray
#         Array of the coordinates of the atoms forming the system.
#     elements : list
#         List of atomic species corresponding to each coordinate.

#     bin_precision: float, optional
#         Bin precision in Angstroms.
#     bin_count: int, optional
#         Number of bins.
#     Returns
#     -------
#     tuple
#         - ndarray: the values of the interatomic distances for each bin
#         - ndarray: the number of atoms within a given distance for each bin
#     """
#     # Select only the indices of the atoms corresponding to the specified element
#     _check_structure(coords=coords, elements=elements)

#     selected_indices = [i for i, el in enumerate(elements) if el == element]
#     selected_coords = coords[selected_indices]

#     n_atoms = len(selected_indices)
#     if n_atoms < 2:
#         raise ValueError(
#             "Not enough atoms of the specified element to compute PDDF."
#         )

#     dist_mat, dist_max, _ = distance_matrix(
#         coords=selected_coords
#     )

#     if bin_precision:
#         n_bins = dist_max / bin_precision
#     else:
#         n_bins = bin_count
#         bin_precision = dist_max / n_bins

#     n_bins_int = int(n_bins)
#     dist_count = np.zeros(n_bins_int)
#     dist = np.zeros(n_bins_int)

#     for i in range(n_bins_int):
#         for j in range(n_atoms):
#             for k in range(n_atoms):
#                 if j != k and (
#                     dist_mat[j, k] < bin_precision * i
#                     and dist_mat[j, k] >= (bin_precision * (i - 1))
#                 ):
#                     dist_count[i] += 1
#         dist[i] = bin_precision * i

#     return dist, dist_count

def rdf_calculator(
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


def partial_rdf_calculator(
    elements: np.ndarray,
    coords: np.ndarray,
    cut_off: int,
    bin_count: int = None,
    bin_precision: float = None,
) -> dict:
    """Computes the radial distribution function for the system computing both the total and elemnt separated function, that is,
    if there are two atomic species (say Au and Pd) it computes the Au-Au, Pd-Pd and "chemical blind" RDF.

    To bin the distribution the user can decide wether they want to specify the bin precision (in Angstrom) or the number of bins. One of
    the two has to be chosen.

    To facilitate the representation and use, the cmputed RDF is returned as a dictionary.

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

    Returns
    -------
    dict
        Dictionary containing as keys the element for which the rdf has been computed and as value two arrays, one continaing the bineed distance
        the other containg the computed RDF.

    Raises
    ------
    ValueError
        Raised if the user does not specify either the bin_count or the bin_precision.
    """
    _check_structure(coords=coords, elements=elements)

    unique_elements, n_elements = np.unique(elements, return_counts=True)
    rdf_dict = {}
    elements = np.array(elements)
    hull = ConvexHull(coords)
    box_volume = hull.volume
    for el in unique_elements:

        idx_el = np.where(elements == str(el))[0]
        coords_el = np.array(coords[idx_el])

        if bin_count:
            d_el, rdf_el = rdf_calculator(
                coords=coords_el,
                cut_off=cut_off,
                bin_count=bin_count,
                box_volume=box_volume,
            )
        elif bin_precision:
            d_el, rdf_el = rdf_calculator(
                coords=coords_el,
                cut_off=cut_off,
                bin_precision=bin_precision,
                box_volume=box_volume,
            )
        else:
            raise ValueError(
                "Either bin_count or bin_precision must be provided."
            )

        rdf_dict[el] = [d_el, rdf_el]
    d_tot, rdf_tot = rdf_calculator(
        coords=coords,
        cut_off=cut_off,
        bin_count=bin_count,
    )
    rdf_dict["Tot"] = [d_tot, rdf_tot]
    return rdf_dict

def com_rdf_calculator(coords, bin_width, elements, com=None):
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