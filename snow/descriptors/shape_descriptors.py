# implementation of functions to compute
# the gyration tensor, inertia tensor, 
# and a series of descriptors for the shape of the cluster
# see e.g. https://en.wikipedia.org/wiki/Gyration_tensor
# https://en.wikipedia.org/wiki/Moment_of_inertia
# https://docs.lammps.org/compute_gyration_shape.html
# https://doi.org/10.1002/adts.201900013

# date: Jul 16th, 2025
# author: gilberto.nardi@studenti.unimi.it

# update: add masses as weights everywhere and default masses (all masses=1), add gyration radius also from the gyration tensor

import numpy as np

def compute_gyration_tensor(positions, masses=None, COM=True):
    """ 
    Computes the gyration tensor for a given set of coordinates. \n
    This can be done in the center of mass reference system or in the raw provided coordinates.
    In averages, positions are weighted by their mass. The default is: all masses are the same (np.ones(n))

    Parameters
    ----------
    positions : ndarray
        (n,3) Array of the coordinates of the atoms forming the system.
    masses : ndarray
        (n,) array of the masses of the atoms in the system. Default to np.ones(n)
    COM : bool
        if True, the calculation is performed in the center of mass reference system. Otherwise, it 
        is performed with the raw coordinates provided by the user

    Returns
    -------
    ndarray
        3x3 gyration tensor
    """

    if masses is None:
        masses = np.ones(positions.shape[0])

    if COM:
        centered = positions - np.average(positions, axis=0, weights=masses)
    else:
        centered = positions
    
    Sxx = np.average(centered[:,0]*centered[:,0], weights=masses)
    Syy = np.average(centered[:,1]*centered[:,1], weights=masses)
    Szz = np.average(centered[:,2]*centered[:,2], weights=masses)
    Sxy = np.average(centered[:,0]*centered[:,1], weights=masses)
    Sxz = np.average(centered[:,0]*centered[:,2], weights=masses)
    Syz = np.average(centered[:,1]*centered[:,2], weights=masses)
        
    return np.array([[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]])


def compute_gyration_desc_from_tensor(gyration_tensor):
    """ 
    Computes general shape descriptors (gyration radius, asphericity, acylindricity, relative shape anisotropy) from the gyration tensor
    
    Parameters
    ----------
    gyration_tensor : ndarray
        3x3 gyration tensor

    Returns
    -------
    float
        gyration radius
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

    rg = np.sqrt(l1+l2+l3)  #gyration radius
    b = l3 - (l1 + l2)/2.   #asphericity
    c = l2 - l1             #acilindricity
    k = 3.*(l1*l1 + l2*l2 + l3*l3)/2./((l1 + l2 + l3)**2.) -0.5   #relative shape anisotropy

    return rg, b, c, k


def compute_gyration_descriptors(positions, masses=None, COM=True):
    """ 
    Computes general shape descriptors obtained from the gyration tensor \n
    (gyration radius, asphericity, acylindricity, relative shape anisotropy) directly from the provided atomic positions. \n
    The positions can be weighted by their masses, but if you are only interested in the characterization of \n
    the shape, you can decide to use the default masses, whichs is np.ones(n) - equivalent to a non-weighted average when \n
    computing the gyration tensor
    
    Parameters
    ----------
    positions : ndarray
        (n,3) Array of the coordinates of the atoms forming the system.
    masses : ndarray
        (n,) array of the masses of the atoms in the system. Default to np.ones(n)
    COM : bool
        if True, the calculation is performed in the center of mass reference system. Otherwise, it 
        is performed with the raw coordinates provided by the user

    Returns
    -------
    float
        asphericity
    float
        acylindricity
    float
        relative shape anisotropy
    """

    return compute_gyration_desc_from_tensor(compute_gyration_tensor(positions, masses, COM))



def compute_inertia_tensor(positions, masses=None, COM=True):

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


def compute_aspect_ratio_from_tensor(inertia_tensor):
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



def compute_aspect_ratio(positions, masses=None, COM=True):
    """ 
    Computes the aspect ratio, a shape descriptor derived from the inertia tensor, for a given set of coordinates. \n
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
    float
        aspect ratio
    """

    return compute_aspect_ratio_from_tensor(compute_inertia_tensor(positions, masses, COM))

def compute_gyr_rad(positions, masses=None):
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