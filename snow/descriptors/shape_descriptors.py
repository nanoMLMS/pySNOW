# implementation of functions to compute
# the gyration tensor, inertia tensor, 
# and a series of descriptors for the shape of the cluster
# see e.g. https://en.wikipedia.org/wiki/Gyration_tensor
# https://en.wikipedia.org/wiki/Moment_of_inertia
# https://docs.lammps.org/compute_gyration_shape.html
# https://doi.org/10.1002/adts.201900013

# date: Jul 16th, 2025 - Mar 9th, 2026
# author: gilberto.nardi@unimi.it

import numpy as np
from snow.misc.constants import mass



def eccentricity(el, coords, round_step=0.05):
    """
    Calcola l'eccentricità (smallest/largest bounding box dimension) 
    per ogni frame.
    
    coords: array (n_frames, n_atoms, 3)
    el: non usato, solo per compatibilità
    round_step: passo per arrotondamento opzionale
    """
    n_frames = coords.shape[0]
    ratios = np.full(n_frames, np.nan)  # default NaN

    for t in range(n_frames):
        frame_coords = np.atleast_2d(coords[t])  # assicura array 2D

        if frame_coords.shape[0] < 2:
            # troppo pochi atomi per calcolare bounding box
            continue

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

        ratios[t] = ratio

    return ratios
    


def center_of_mass(
    index_frame: int, coords: np.ndarray, elements
) -> np.ndarray:
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
    com = np.sum(coords * masses[:, None], axis=0) / total_mass

    return com


def geometric_com(index_frame: int, coords: np.ndarray):
    gcom = np.mean(coords, axis=0)
    return gcom

def gyr_tensor(positions):
    """ 
    Computes the gyration tensor for a given set of coordinates. \n
    This is done in the center of positions reference system.

    Parameters
    ----------
    positions : ndarray
        (n,3) Array of the coordinates of the atoms forming the system.

    Returns
    -------
    ndarray
        3x3 gyration tensor
    """

    centered = positions - np.average(positions, axis=0)

    Sxx = np.average(centered[:,0]*centered[:,0])
    Syy = np.average(centered[:,1]*centered[:,1])
    Szz = np.average(centered[:,2]*centered[:,2])
    Sxy = np.average(centered[:,0]*centered[:,1])
    Sxz = np.average(centered[:,0]*centered[:,2])
    Syz = np.average(centered[:,1]*centered[:,2])
        
    return np.array([[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]])


def gyr_desc_from_tensor(gyration_tensor):
    """ 
    Computes general shape descriptors (asphericity, acylindricity, relative shape anisotropy) from the gyration tensor
    
    Parameters
    ----------
    gyration_tensor : ndarray
        3x3 gyration tensor

    Returns
    -------
    float
        asphericity
    float
        acylindricity
    float
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


def gyr_desc(index_frame: int, positions):
    """ 
    Computes general shape descriptors obtained from the gyration tensor \n
    (gyration radius, asphericity, acylindricity, relative shape anisotropy) directly from the provided atomic positions. \n
    
    Parameters
    ----------
    index_frame (int):
        Index of the frame in the trajectory (for reference only - not used now).
    positions : ndarray
        (n,3) Array of the coordinates of the atoms forming the system.

    Returns
    -------
    float
        asphericity
    float
        acylindricity
    float
        relative shape anisotropy
    """

    return gyr_desc_from_tensor(gyr_tensor(positions))



def inertia_tensor(positions, masses=None, COM=True):

    """ 
    Computes the inertia tensor for a given set of coordinates. \n
    This can be done in the center of mass reference system or in the raw provided coordinates

    Parameters
    ----------
    positions : ndarray
        Nx3 Array of the coordinates of the atoms in the system.
    masses : ndarray
        array of the masses of the atoms in the system. Default to np.ones(n)
    COM : bool
        if True, the calculation is performed in the center of mass reference system. Otherwise, it 
        is performed with the raw coordinates provided by the user

    Returns
    -------
    ndarray
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
    """

    eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)
    eigenvalues = np.sort(eigenvalues)

    return  eigenvalues[0]/eigenvalues[2]



def aspect_ratio(index_frame: int, positions, masses=None, COM=True):
    """ 
    Computes the aspect ratio, a shape descriptor derived from the inertia tensor, for a given set of coordinates. \n
    This can be done in the center of mass reference system or in the raw provided coordinates

    Parameters
    ----------
    index_frame (int):
        Index of the frame in the trajectory (for reference only - not used now).
    positions : ndarray
        Nx3 Array of the coordinates of the atoms in the system.
    masses : ndarray
        array of the masses of the atoms in the system. Default to np.ones(n)
    COM : bool
        if True, the calculation is performed in the center of mass reference system. Otherwise, it 
        is performed with the raw coordinates provided by the user

    Returns
    -------
    float
        aspect ratio
    """

    return aspect_ratio_from_tensor(inertia_tensor(positions, masses, COM))

def gyr_rad(index_frame: int, positions, masses=None):
    """
    Computes the gyration radius, which corresponds to the average <r^2> weighted by the masses of\n
    atoms in the system.

    Parameters
    ----------
    index_frame (int):
        Index of the frame in the trajectory (for reference only - not used now).
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
