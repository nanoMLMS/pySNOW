import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist, squareform


from snow.descriptors.utils import distance_matrix, hetero_distance_matrix

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


def hetero_pddf_calculator(
    index_frame: int, coords: np.ndarray, elements: np.ndarray, bin_precision: float | None = None, bin_count: int | None = None
):
    """Computes the PDDF only for atoms with different chemical species.

    Parameters
    ----------
    index_frame : int
        Index of the frame relative to the snapshot, primarily for reference.
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
    n_atoms = np.shape(coords)[0]
    dist_mat = hetero_distance_matrix(
        index_frame=index_frame, elements=elements, coords=coords
    )

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


def chemical_pddf_calculator(
    index_frame, coords, elements, bin_precision=None, bin_count=None
):
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
    elements: ndarray
        Array of elements
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
    unique_elements, n_elements = np.unique(elements, return_counts=True)

    pddf_dict = {}
    elements = np.array(elements)
    if bin_count:
        d_tot, pddf_tot = pddf_calculator(
            index_frame=index_frame, coords=coords, bin_count=bin_count
        )
    elif bin_precision:
        d_tot, pddf_tot = pddf_calculator(
            index_frame=index_frame, coords=coords, bin_precision=bin_precision
        )

    pddf_dict["Tot"] = [d_tot, pddf_tot]
    for el in unique_elements:

        idx_el = np.where(elements == str(el))[0]
        coords_el = np.array(coords[idx_el])

        if bin_count:
            d_el, pddf_el = pddf_calculator(
                index_frame=index_frame, coords=coords_el, bin_count=bin_count
            )
        elif bin_precision:
            d_el, pddf_el = pddf_calculator(
                index_frame=index_frame,
                coords=coords_el,
                bin_precision=bin_precision,
            )
        else:
            raise ValueError(
                "Either bin_count or bin_precision must be provided."
            )

        pddf_dict[el] = [d_el, pddf_el]

    return pddf_dict


def pddf_calculator_by_element(
    index_frame,
    coords,
    elements,
    element="Au",
    bin_precision=None,
    bin_count=None,
):
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
        raise ValueError(
            "Not enough atoms of the specified element to compute PDDF."
        )

    dist_mat, dist_max, _ = distance_matrix(
        index_frame=index_frame, coords=selected_coords
    )

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
                if j != k and (
                    dist_mat[j, k] < bin_precision * i
                    and dist_mat[j, k] >= (bin_precision * (i - 1))
                ):
                    dist_count[i] += 1
        dist[i] = bin_precision * i

    return dist, dist_count

def rdf_calculator(
    index_frame: int,
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
    index_frame : int
        _description_
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
    index_frame: int,
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
    index_frame : int
        _description_
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
                index_frame=index_frame,
                coords=coords_el,
                cut_off=cut_off,
                bin_count=bin_count,
                box_volume=box_volume,
            )
        elif bin_precision:
            d_el, rdf_el = rdf_calculator(
                index_frame=index_frame,
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
        index_frame=index_frame,
        coords=coords,
        cut_off=cut_off,
        bin_count=bin_count,
    )
    rdf_dict["Tot"] = [d_tot, rdf_tot]
    return rdf_dict
