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
                       elements : list = None,
                       dist_max: float = None):
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
    dist_max : float, default to None
        Maximum distance up to which the distribution is computed. This is useful
        when comparing distributions from different configurations, as it fixes the
        histogram range (i.e. the bin edges). Default to None, for which the maximum distance from the com is used
        to define the bins limit.

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
    com_dists = np.linalg.norm(coords - com, axis=1)
    
    if dist_max is None:
        dist_max = np.max(com_dists)
    else:
        if np.max(com_dists) > dist_max:
            print(f"Warning: selected max distance for com_rdf_calculator is smaller than system's max distance from com ({np.max(com_dists)}, {dist_max})")

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
    length_max: float = None
    
):
    """
    Cuts a single frame into layers and compute the distribution of atoms in the layers.

    Computes the distribution of atoms per layer of width `layer_height`. The axis along which (perpendicular)
    planes are cut can be specified as either 'z' (default), 'x', 'y', or a user-defined np.ndarray.
    the distribution goes from the minimum coordinate along the given axis to either the maximum coordinate
    along that direction or length_max, if provided by the user

    Parameters
    ----------
    elements : np.ndarray
        chemical symbols of the atoms provided - Shape (n_atoms,)
    coords_frame : np.ndarray
        coordinates of the atoms provided - Shape (n_atoms, 3)
    layer_height : float
        width for the layers (bins) of the distributions
    cutting_ax : str or np.ndarray
        either 'x', 'y', 'z', or a (3, ) np.ndarray such as (1,1,0)
    species_A : str (optional)
        chemical specie 1 to filter the coords and get a chemical specie-wise count of atoms per layer
    species_B : str (optional)
        chemical specie 2 to filter the coords and get a chemical specie-wise count of atoms per layer
    length_max : float, default to None
        Maximum distance up to which the distribution is computed. This is useful
        when comparing distributions from different configurations, as it fixes the
        histogram range (i.e. the bin edges). Default to None, for which the maximum distance 
        between atoms along the given axis is computed.
        
    Returns
    -------
    layer_center : np.ndarray
        Layer coordinates (bin centers), shape (n_layers,).
    layer_ntot : np.ndarray
        Total atom count per layer, shape (n_layers,).
    layer_na : np.ndarray
        Atom count per layer for species_A, shape (n_layers,).
        Only returned if species_A is not None.
    layer_nb : np.ndarray
        Atom count per layer for species_B, shape (n_layers,).
        Only returned if species_B is not None.
    """

    #get splicing axis
    if isinstance(cutting_ax, str):
        if cutting_ax == 'x':
            cutting_ax = np.asarray([1.,0.,0.])
        elif cutting_ax == 'y':
            cutting_ax = np.asarray([0.,1.,0.])
        elif cutting_ax == 'z':
            cutting_ax = np.asarray([0.,0.,1.])
    else:
        cutting_ax = np.asarray(cutting_ax, dtype=float)

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

    # #shift local copy of coordinates to (0, zmax) for better manipulation
    # z = z - min_z
    
    #range across which the histogram is computed
    if length_max is not None:
        length_range = length_max
        if length_range < max_z - min_z:
            print(f'Warning: selected length_range is smaller than length of the system across given axis ({length_range}, {max_z-min_z})')
    else:
        length_range = max_z - min_z

    n_layers = int(np.ceil(length_range / layer_height))

    #layer_number  = np.zeros(n_layers, dtype=int)
    layer_ntot    = np.zeros(n_layers, dtype=int)
    layer_centers = np.zeros(n_layers, dtype=float)
    if species_A is not None:
        layer_na     = np.zeros(n_layers, dtype=int)
    if species_B is not None:
        layer_nb     = np.zeros(n_layers, dtype=int)

    for i in range(n_layers):
    
        #bin edges
        z_min_bin = i * layer_height + min_z
        z_max_bin = (i + 1) * layer_height + min_z
        layer_centers[i] = (z_min_bin + z_max_bin)/2.

        mask = (z >= z_min_bin) & (z < z_max_bin) if i < n_layers - 1 else (z >= z_min_bin) & (z <= z_max_bin) #dealing with last atom on last bin case
        
        tot = np.count_nonzero(mask)
        if species_A is not None:
            n_A = np.count_nonzero(mask & (elements == species_A))
        if species_B is not None:
            n_B = np.count_nonzero(mask & (elements == species_B))
        
        #layer_number[i] = i
        layer_ntot[i]   = tot
        if species_A is not None:
            layer_na[i]     = n_A
        if species_B is not None:
            layer_nb[i]     = n_B

    if species_A is not None and species_B is not None:
        return layer_centers, layer_ntot, layer_na, layer_nb
    elif species_A is not None:
        return layer_centers, layer_ntot, layer_na
    elif species_B is not None:
        return layer_centers, layer_ntot, layer_nb
    else:
        return layer_centers, layer_ntot


def cylindrical_distribution(el, coords, bin_width, ax, com=True, center=None):
    """
    Compute the distribution of atomic positions in the system in a cylindrical-wise fashion.

    A principal axis is considered (identifying the cylindrical symmetry axis, or in general,
    the z/planar coordinate for the cylindrical system), and from there, atomic coordinates are binned
    in 'slice-of-cake'-like bins, where the bin size is given as an angular width identifying a range 
    of radial directions spanning from the central cylindrical axis. By default, binning is computed across
    the entire (2*pi) angular domain.

    Parameters
    ----------
    el : np.ndarray
        chemical symbols of the atoms provided - Shape (n_atoms,)
    coords : np.ndarray
        coordinates of the atoms provided - Shape (n_atoms, 3)
    ax : ndarray - shape (3,)
        an array identifying the main axis of the cylindrical coordinates system
    bin_width : float
        width for the distributions bins - in *radians*. This is an angular width for bins 
        spanning radial directions starting from the axis
    com : bool, deafult True
        if True, the center (origin) of the system of coordinates is in the center of mass of the system.
        If False, you should provide the desired center with the center argument.
    center : ndarray, shape (3,), default to None
        the xyz positions of the center (origin) of the system, if com is set to False and thus you want to 
        specify your own origin for the system

    Returns
    -------
    bin_centers : array[float]
        bin centers of the bins of the distribution - in radians
    bin_count : array[float]
        number of atoms in each bin.

    """

    if not com and center is None:
        raise ValueError('if com is False, you should provide your own origin for the coordinates system.')
    

    #align z axis with provided axis
    if not np.array_equal(ax, np.array([0., 0., 1.])):
        cc = align_axis_to_z(coords, axis=ax)
        if not com:
            center = align_axis_to_z(center[None, :], axis=ax)[0] #handles correctly np.ndarray shapes 
    else:
        cc = coords

    #shift to desired origin
    if com:
        center = center_of_mass(el, cc)
    
    shifted_coords = cc - center

    #transform x', y' to angles
    angles = np.arctan2(shifted_coords[:,1], shifted_coords[:,0]) % (2*np.pi)
    
    n_bins = int(np.ceil(2*np.pi / bin_width))
    bins = np.linspace(0, n_bins*bin_width, n_bins + 1)
    angles_count, _ = np.histogram(angles, bins=bins)
    
    return (bins[:-1] + bin_width/2.), angles_count

def solid_angle_distribution(el, coords, bin_width_theta, bin_width_phi, com=True, center=None):
    """
    Compute the distribution of atomic positions in the system in a spherical-wise fashion.

    The z-axis is used as the polar axis of the spherical coordinate system. Atomic coordinates
    are binned over solid angle bins defined by the polar angle theta (0 to pi) and the azimuthal
    angle phi (0 to 2*pi). Each bin covers a range of solid angles identified by
    (bin_width_theta, bin_width_phi).

    Parameters
    ----------
    el : np.ndarray
        chemical symbols of the atoms provided - Shape (n_atoms,)
    coords : np.ndarray
        coordinates of the atoms provided - Shape (n_atoms, 3)
    bin_width_theta : float
        width for the bins along the polar angle theta - in *radians*. The polar angle ranges
        from 0 (north pole) to pi (south pole).
    bin_width_phi : float
        width for the bins along the azimuthal angle phi - in *radians*. The azimuthal angle ranges
        from 0 to 2*pi.
    com : bool, default True
        if True, the center (origin) of the system of coordinates is the center of mass of the system.
        If False, you should provide the desired center with the center argument.
    center : ndarray, shape (3,), default to None
        the xyz positions of the center (origin) of the system, if com is set to False and thus you want to
        specify your own origin for the system

    Returns
    -------
    bin_centers : np.ndarray
        array of shape (n_bins_theta, n_bins_phi, 2) containing the (theta, phi) coordinates
        of each bin center - theta in radians, phi in radians
    counts : np.ndarray
        array of shape (n_bins_theta, n_bins_phi) containing the number of atoms in each bin

    Raises
    ------
    ValueError
        If com is False and no center is provided.
    """

    if not com and center is None:
        raise ValueError('if com is False, you should provide your own origin for the coordinates system.')

    # shift to desired origin
    if com:
        center = center_of_mass(el, coords)

    shifted_coords = coords - center

    # transform to spherical angles
    r = np.linalg.norm(shifted_coords, axis=1)
    # avoid division by zero for atoms exactly at the origin
    r_safe = np.where(r > 0, r, 1.0)

    # polar angle theta: arccos(z/r), ranges [0, pi]
    theta = np.arccos(np.clip(shifted_coords[:, 2] / r_safe, -1.0, 1.0))
    # azimuthal angle phi: arctan2(y, x), ranges [0, 2*pi]
    phi = np.arctan2(shifted_coords[:, 1], shifted_coords[:, 0]) % (2 * np.pi)

    # remove atoms exactly at the origin (they have undefined angles)
    valid = r > 0
    theta = theta[valid]
    phi = phi[valid]

    n_bins_theta = int(np.ceil(np.pi / bin_width_theta))
    n_bins_phi = int(np.ceil(2 * np.pi / bin_width_phi))

    # build 2D histogram
    theta_edges = np.linspace(0, n_bins_theta * bin_width_theta, n_bins_theta + 1)
    phi_edges = np.linspace(0, n_bins_phi * bin_width_phi, n_bins_phi + 1)

    counts, _, _ = np.histogram2d(theta, phi, bins=[theta_edges, phi_edges])

    # compute bin centers
    theta_centers = theta_edges[:-1] + bin_width_theta / 2.0
    phi_centers = phi_edges[:-1] + bin_width_phi / 2.0
    bin_centers = np.stack(np.meshgrid(theta_centers, phi_centers, indexing='ij'), axis=-1)

    return bin_centers, counts

def columns_distribution(coords, bin_width_x, bin_width_y, use_lattice_units, lattice=None, ax='z', max_x=None, max_y=None):
    """
    Compute distribution of atomic positions in a per-column fashion (counts the number of atoms in
    each (x, x+dx) x (y, y+dy) rectangular bin).

    The z-axis is used as the column axis by default. The x and y axes define the plane across which
    atoms are binned into columns. If a different column axis is specified via `ax`, the coordinates
    are first rotated to align that axis to z before binning.

    If use_lattice_units=True, bin_width_x and bin_width_y should be provided in lattice units (alat)
    and the distribution is returned in lattice units.
    If use_lattice_units=False, the bin widths should be provided in the same units as coords.

    Parameters
    ----------
    coords : np.ndarray
        coordinates of the atoms provided - Shape (n_atoms, 3)
    bin_width_x : float
        width of the bins along the x axis. It should be provided in lattice units if
        use_lattice_units==True, and in the same units as coords if use_lattice_units==False.
    bin_width_y : float
        width of the bins along the y axis. It should be provided in lattice units if
        use_lattice_units==True, and in the same units as coords if use_lattice_units==False.
    use_lattice_units : bool
        If True, the distribution is computed and returned in units of the lattice constant (passed
        as the 'lattice' argument) and the bin widths should be given in units of the lattice constant.
        If False, the distribution is returned in the units of coords, and the bin widths should be
        given in the same units as coords.
    lattice : float, optional
        Specify a value for the lattice parameter of your structure in the same units as coords.
        Only needed if use_lattice_units is True
    ax : str or np.ndarray, default 'z'
        the column axis. Either 'x', 'y', 'z', or a (3,) np.ndarray such as (1,1,0).
    max_x : float, default to None
        Maximum extent of the distribution along the x axis. This is useful
        when comparing distributions from different configurations, as it fixes the
        histogram range (i.e. the bin edges). Default to None, for which the maximum
        x coordinate of the system is used.
    max_y : float, default to None
        Maximum extent of the distribution along the y axis. This is useful
        when comparing distributions from different configurations, as it fixes the
        histogram range (i.e. the bin edges). Default to None, for which the maximum
        y coordinate of the system is used.

    Returns
    -------
    bin_centers : np.ndarray
        array of shape (n_bins_x, n_bins_y, 2) containing the (x, y) coordinates
        of each bin center
    counts : np.ndarray
        array of shape (n_bins_x, n_bins_y) containing the number of atoms in each bin

    Raises
    ------
    ValueError
        If use_lattice_units is True and no lattice constant is provided.
    """

    if use_lattice_units:
        if lattice is None:
            raise ValueError('If use_lattice_units==True, you should provide a value for the lattice constant to use')
        coords = coords / lattice
        if max_x is not None:
            max_x = max_x / lattice
        if max_y is not None:
            max_y = max_y / lattice

    # resolve ax string to array
    if isinstance(ax, str):
        if ax == 'x':
            ax = np.asarray([1., 0., 0.])
        elif ax == 'y':
            ax = np.asarray([0., 1., 0.])
        elif ax == 'z':
            ax = np.asarray([0., 0., 1.])
    else:
        ax = np.asarray(ax, dtype=float)

    # align selected axis to z
    if not np.array_equal(ax, np.array([0., 0., 1.])):
        cc = align_axis_to_z(coords, axis=ax)
    else:
        cc = coords

    x = cc[:, 0]
    y = cc[:, 1]

    min_x = x.min()
    min_y = y.min()

    # x range
    if max_x is not None:
        range_x = max_x
        if range_x < x.max() - min_x:
            print(f'Warning: selected max_x is smaller than the system extent along x ({range_x}, {x.max() - min_x})')
    else:
        range_x = x.max() - min_x

    # y range
    if max_y is not None:
        range_y = max_y
        if range_y < y.max() - min_y:
            print(f'Warning: selected max_y is smaller than the system extent along y ({range_y}, {y.max() - min_y})')
    else:
        range_y = y.max() - min_y

    n_bins_x = int(np.ceil(range_x / bin_width_x))
    n_bins_y = int(np.ceil(range_y / bin_width_y))

    x_edges = np.linspace(min_x, min_x + range_x, n_bins_x + 1)
    y_edges = np.linspace(min_y, min_y + range_y, n_bins_y + 1)

    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    x_centers = x_edges[:-1] + bin_width_x / 2.0
    y_centers = y_edges[:-1] + bin_width_y / 2.0
    bin_centers = np.stack(np.meshgrid(x_centers, y_centers, indexing='ij'), axis=-1)

    return bin_centers, counts