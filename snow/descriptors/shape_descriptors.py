# implementation of functions to compute
# the gyration tensor, inertia tensor, 
# and a series of descriptors for the shape of the cluster
# see e.g. https://en.wikipedia.org/wiki/Gyration_tensor
# https://en.wikipedia.org/wiki/Moment_of_inertia
# https://docs.lammps.org/compute_gyration_shape.html
# https://doi.org/10.1002/adts.201900013

# date: Jul 16th, 2025
# author: gilberto.nardi@studenti.unimi.it

import numpy as np

def compute_gyration_tensor(positions, COM=True):
    """ 
    Computes the gyration tensor for a given set of coordinates. \n
    This can be done in the center of mass reference system or in the raw provided coordinates

    Parameters
    ----------
    positions : ndarray
        (n,3) Array of the coordinates of the atoms forming the system.
    COM : bool
        if True, the calculation is performed in the center of mass reference system. Otherwise, it 
        is performed with the raw coordinates provided by the user

    Returns
    -------
    ndarray
        3x3 gyration tensor
    """

    if COM:
        centered = positions - positions.mean(axis=0)
    else:
        centered = positions
    
    Sxx = np.average(centered[:,0]*centered[:,0])
    Syy = np.average(centered[:,1]*centered[:,1])
    Szz = np.average(centered[:,2]*centered[:,2])
    Sxy = np.average(centered[:,0]*centered[:,1])
    Sxz = np.average(centered[:,0]*centered[:,2])
    Syz = np.average(centered[:,1]*centered[:,2])
        
    return np.array([[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]])


def compute_gyration_desc_from_tensor(gyration_tensor):
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

    b = l3 - (l1 + l2)/2.  #asphericity
    c = l2 - l1            #acilindricity
    k = 3.*(l1*l1 + l2*l2 + l3*l3)/2./((l1 + l2 + l3)**2.) -0.5   #relative shape anisotropy

    return b, c, k


def compute_gyration_descriptors(positions, COM=True):
    """ 
    Computes general shape descriptors obtained from the gyration tensor \n
    (asphericity, acylindricity, relative shape anisotropy) directly from the provided atomic positions \n
    
    Parameters
    ----------
    positions : ndarray
        (n,3) Array of the coordinates of the atoms forming the system.
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

    return compute_gyration_desc_from_tensor(compute_gyration_tensor(positions, COM))



def compute_inertia_tensor(positions, masses, COM=True):

    """ 
    Computes the inertia tensor for a given set of coordinates. \n
    This can be done in the center of mass reference system or in the raw provided coordinates

    Parameters
    ----------
    positions : ndarray
        Nx3 Array of the coordinates of the atoms in the system.
    masses : ndarray
        array of the masses of the atoms in the system
    COM : bool
        if True, the calculation is performed in the center of mass reference system. Otherwise, it 
        is performed with the raw coordinates provided by the user

    Returns
    -------
    ndarray
        3x3 inertia tensor
    """

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



def compute_aspect_ratio(positions, masses, COM=True):
    """ 
    Computes the aspect ratio, a shape descriptor derived from the inertia tensor, for a given set of coordinates. \n
    This can be done in the center of mass reference system or in the raw provided coordinates

    Parameters
    ----------
    positions : ndarray
        Nx3 Array of the coordinates of the atoms in the system.
    masses : ndarray
        array of the masses of the atoms in the system
    COM : bool
        if True, the calculation is performed in the center of mass reference system. Otherwise, it 
        is performed with the raw coordinates provided by the user

    Returns
    -------
    float
        aspect ratio
    """

    return compute_aspect_ratio_from_tensor(compute_inertia_tensor(positions, masses, COM))


