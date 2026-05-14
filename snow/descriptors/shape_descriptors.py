# implementation of functions to compute
# the gyration tensor, inertia tensor, 
# and a series of descriptors for the shape of the cluster
# see e.g. https://en.wikipedia.org/wiki/Gyration_tensor
# https://en.wikipedia.org/wiki/Moment_of_inertia
# https://docs.lammps.org/compute_gyration_shape.html
# https://doi.org/10.1002/adts.201900013

import numpy as np
from snow.misc.constants import mass

def eccentricity(coords, round_step=None):
    """
    Compute the eccentricity (smallest/largest bounding box dimension), a measure of shape anisotropy.

    Parameters
    -----------
    coords: ndarray(n_atoms, 3) 
        Coordinates of the atoms in your systems.
    round_step: float, optional
        precision for rounding - optional
    
    Returns
    -------
    ratio : float
        computed eccentricity
    """

    frame_coords = np.atleast_2d(coords)  # ensures array 2D

    if frame_coords.shape[0] < 2:
        # too few atoms to compute the bounding box
        return np.nan

    min_values = frame_coords.min(axis=0)
    max_values = frame_coords.max(axis=0)

    length_x = max_values[0] - min_values[0]
    length_y = max_values[1] - min_values[1]
    length_z = max_values[2] - min_values[2]

    largest_dimension = max(length_x, length_y, length_z)
    smallest_dimension = min(length_x, length_y, length_z)

    ratio = smallest_dimension / largest_dimension

    if round_step is not None:
        ratio = round(ratio / round_step) * round_step

    return ratio
    


def center_of_mass(elements: list[str], coords: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass for a given frame of coordinates.

    Atomic masses are read from `snow.misc.constants` and used as weights for the 
    weighted average of positions.

    Parameters
    ----------
    elements : list or np.ndarray
        List of element symbols corresponding to the atoms, for weighted sum on masses
    coords : np.ndarray
        Array of atomic coordinates of shape ( n_atoms, 3).

    Returns
    -------
    com : np.ndarray 
        The center of mass as a 3D vector (x_{com}, y_{com}, z_{com}).
    """

    # Get the masses of the elements
    masses = np.array([mass[e] for e in elements])

    # Compute the total mass
    total_mass = np.sum(masses)

    # Calculate the weighted average of coordinates
    com = np.sum(coords * masses[:, None], axis=0) / total_mass

    return com


def geometric_com(coords: np.ndarray):
    """
    Computes the geometric center of mass (average of the coordinates)

    Computes the average of the coordinates, which is the geometrical center of mass
    (thus, not considering different weights due to different atomic masses)

    Parameters
    ----------
    coords : np.ndarray
        Array of atomic coordinates of shape ( n_atoms, 3).

    Returns
    -------
    gcom : np.ndarray 
        The geometric center of mass as a 3D vector (x_{com}, y_{com}, z_{com}).
    """

    gcom = np.mean(coords, axis=0)
    return gcom


def gyr_tensor(positions):
    """ 
    Computes the gyration tensor for a given set of coordinates.
    This is done in the center of positions reference system.

    Parameters
    ----------
    positions : ndarray
        (n,3) Array of the coordinates of the atoms forming the system.

    Returns
    -------
    np.ndarray
        3x3 gyration tensor
    """

    centered = positions - geometric_com(positions)

    Sxx = np.average(centered[:,0]*centered[:,0])
    Syy = np.average(centered[:,1]*centered[:,1])
    Szz = np.average(centered[:,2]*centered[:,2])
    Sxy = np.average(centered[:,0]*centered[:,1])
    Sxz = np.average(centered[:,0]*centered[:,2])
    Syz = np.average(centered[:,1]*centered[:,2])
        
    return np.array([[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]])


def gyr_desc_from_tensor(gyration_tensor):
    """ 
    Computes general shape descriptors (asphericity, acylindricity, relative shape anisotropy) from the gyration tensor.
    
    Parameters
    ----------
    gyration_tensor : ndarray
        3x3 gyration tensor

    Returns
    -------
    b : float
        asphericity
    c : float
        acylindricity
    k : float
        relative shape anisotropy
    """

    eigenvalues, eigenvectors = np.linalg.eig(gyration_tensor)
    eigenvalues = np.sort(eigenvalues)

    l1 = eigenvalues[0]
    l2 = eigenvalues[1]
    l3 = eigenvalues[2]

    b = l3 - (l1 + l2)/2.   #asphericity
    c = l2 - l1             #acilindricity
    k = 3.*(l1*l1 + l2*l2 + l3*l3)/2./((l1 + l2 + l3)**2.) -0.5   #relative shape anisotropy

    return b, c, k


def gyr_desc(positions):
    """ 
    Computes general shape descriptors obtained from the gyration tensor.

    Computes asphericity, acylindricity, and relative shape anisotropy from the provided atomic positions.
    
    Parameters
    ----------
    positions : ndarray
        (n,3) Array of the coordinates of the atoms forming the system.

    Returns
    -------
    b : float
        asphericity
    c : float
        acylindricity
    k : float
        relative shape anisotropy
    """

    return gyr_desc_from_tensor(gyr_tensor(positions))



def inertia_tensor(positions, masses=None, COM=True):

    """ 
    Computes the inertia tensor for a given set of coordinates.

    This can be done in the center of mass reference system or in the raw provided 
    coordinates - switch with the boolean `COM` variable

    Parameters
    ----------
    positions : ndarray
        Nx3 Array of the coordinates of the atoms in the system.
    masses : ndarray, optional
        Masses of the atoms, shape (n_atoms,). If None (default), all masses
        are set to 1 (equal masses).
    COM : bool
        if True, the calculation is performed in the center of mass reference system. Otherwise, it 
        is performed with the raw coordinates provided by the user

    Returns
    -------
    np.ndarray
        3x3 inertia tensor
    """

    if masses is None:
        masses = np.ones(positions.shape[0])

    if COM:
        centered = positions - np.average(positions, axis=0, weights=masses)
    else:
        centered = positions
    
    Ixx = np.sum(masses * (centered[:, 1]**2 + centered[:, 2]**2))
    Iyy = np.sum(masses * (centered[:, 0]**2 + centered[:, 2]**2))
    Izz = np.sum(masses * (centered[:, 0]**2 + centered[:, 1]**2))
    Ixy = np.sum(-masses*centered[:,0]*centered[:,1])
    Ixz = np.sum(-masses*centered[:,0]*centered[:,2])
    Iyz = np.sum(-masses*centered[:,1]*centered[:,2])

    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def aspect_ratio_from_tensor(inertia_tensor):
    """
    Computes the aspect ratio, a shape descriptor obtained from the inertia tensor.

    Parameters
    ----------
    inertia_tensor : ndarray
        3x3 inertia tensor
    
    Returns
    -------
    ar : float
        aspect ratio
    """

    eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)
    eigenvalues = np.sort(eigenvalues)

    return  eigenvalues[0]/eigenvalues[2]



def aspect_ratio(positions, masses=None, COM=True):
    """ 
    Computes the aspect ratio, a shape descriptor derived from the inertia tensor, for a given set of coordinates. \n

    This can be done in the center of mass reference system or in the raw provided coordinates (switch with the COM
    boolean variable).

    Parameters
    ----------
    positions : ndarray
        Nx3 Array of the coordinates of the atoms in the system.
    masses : ndarray
        array of the masses of the atoms in the system. Default to all equal masses.
    COM : bool
        if True, the calculation is performed in the center of mass reference system. Otherwise, it 
        is performed with the raw coordinates provided by the user

    Returns
    -------
    float
        aspect ratio
    """

    return aspect_ratio_from_tensor(inertia_tensor(positions, masses, COM))

def gyr_rad(positions, masses=None):
    """
    Computes the gyration radius, which corresponds to the average <r^2> weighted by the masses of\n
    atoms in the system.

    Parameters
    ----------
    positions : ndarray
        Nx3 Array of the coordinates of the atoms in the system.
    masses : ndarray
        array of the masses of the atoms in the system. Default to all equal masses
    
    Returns
    -------
    float
        Gyration radius
    """

    if(masses is None):
        masses=np.ones(positions.shape[0])
    
    centered = positions - np.average(positions, axis=0, weights=masses)

    return np.sqrt( np.average( np.sum(centered**2, axis=1), weights=masses ) )



#experimental fast pbc com/gcom calculators
#further testing needed.
# def _parse_box_triclinic(box: np.ndarray):
#     """
#     Returns (H, origin) where H is the (3,3) matrix of lattice vectors as columns.
#     Accepts:
#       (3,)   — orthorhombic lengths, origin at 0
#       (3,2)  — lower/upper bounds per axis, orthorhombic
#       (3,3)  — full lattice matrix (vectors as rows, standard convention)
#     """
#     box = np.asarray(box, dtype=float)
#     if box.shape == (3,):
#         return np.diag(box), np.zeros(3)
#     elif box.shape == (3, 2):
#         lengths = box[:, 1] - box[:, 0]
#         return np.diag(lengths), box[:, 0]
#     elif box.shape == (3, 3):
#         # convention: rows are lattice vectors → transpose to get columns
#         return box.T, np.zeros(3)
#     else:
#         raise ValueError("Box must be shape (3,), (3,2), or (3,3).")


# def center_of_mass_pbc(
#     coords: np.ndarray,
#     elements,
#     box: np.ndarray,
# ) -> np.ndarray:
#     masses = np.array([mass[e] for e in elements])
#     H, origin = _parse_box_triclinic(box)

#     # Cartesian → fractional
#     H_inv = np.linalg.inv(H)
#     r_frac = (coords - origin) @ H_inv.T          # (n, 3), each row in [0,1)

#     # Angle method in fractional space (L=1)
#     theta = 2 * np.pi * r_frac                    # (n, 3)
#     xi    = np.average(np.cos(theta), axis=0, weights=masses)
#     zeta  = np.average(np.sin(theta), axis=0, weights=masses)

#     frac_com = (np.arctan2(-zeta, -xi) + np.pi) / (2 * np.pi)

#     # Fractional → Cartesian
#     return H @ frac_com + origin


# def geometric_com_pbc(
#     coords: np.ndarray,
#     box: np.ndarray,
# ) -> np.ndarray:
#     H, origin = _parse_box_triclinic(box)
#     H_inv = np.linalg.inv(H)
#     r_frac = (coords - origin) @ H_inv.T

#     theta = 2 * np.pi * r_frac
#     xi    = np.mean(np.cos(theta), axis=0)
#     zeta  = np.mean(np.sin(theta), axis=0)

#     frac_com = (np.arctan2(-zeta, -xi) + np.pi) / (2 * np.pi)
#     return H @ frac_com + origin