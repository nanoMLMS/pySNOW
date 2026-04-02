import numpy as np
from snow.descriptors.shape_descriptors import center_of_mass as com, geometric_com as gcom

def ax_from_two_points(coord_pt_1, coord_pt_2):

    x_ax = coord_pt_2[0] - coord_pt_1[0]
    y_ax = coord_pt_2[1] - coord_pt_1[1]
    z_ax = coord_pt_2[2] - coord_pt_1[2]
    
    ax_connecting = np.asarray([x_ax, y_ax, z_ax])
    
    return ax_connecting

def translate_com_to_origin(coords : np.ndarray, elements=None) -> np.ndarray:
    """
    Shifts the positions to the center of mass reference system (so that the center of mass is in the origin). 
    If elements are provided, a mass-weighted average of positions is performed, otherwise (elements=None), a simple
    geometrical average is used.

    Parameters:
        index_frame (int): Index of the frame in the trajectory - for reference only.
        coords (np.ndarray): Array of atomic coordinates of shape (frames, atoms, 3).
        elements (list or np.ndarray): List of element symbols corresponding to the atoms. Default to None.
          If None, all positions will have the same weight in the calculation of the center of mass
    
    Returns:
        np.ndarray: shifted coords
    """

    #check type
    coords = np.asarray(coords, dtype = float)

    if elements is not None:
        return coords - com(coords, elements)
    else:
        return coords - gcom(coords)

def rotate_around_ax(coords, axis, angle):
    """
    Rotate coordinates around a given axis by a given angle.

    Parameters
    ----------
    coords : array-like, shape (..., 3)
        Coordinates to rotate.
    axis : array-like, shape (3,)
        Rotation axis (will be normalized).
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Rotated coordinates, same shape as input.
    """
    coords = np.asarray(coords, dtype=float)
    axis = np.asarray(axis, dtype=float)

    n = np.linalg.norm(axis)
    if n == 0:
        raise ValueError("Rotation axis must be non-zero.")
    axis = axis / n

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    # Rodrigues' rotation matrix
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])

    # Apply rotation (works for shape (N,3) or (...,3))
    return coords @ R.T


def align_axis_to_z(coords: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """ 
    Rotates the system so that the provided symmetry_axis is aligned with the z=(0,0,1) axis

    Parameters
    ----------
    coords : np.ndarray
        Array of the atomic coordinates
    axis : np.ndarray
        axis to become the new z-axis of the coordinates

    Returns
    -------
    np.ndarray
        The transformed system of coordinates
    """

    #two possibly bad cases
    if np.allclose(axis, [0., 0., 1.]):
        return coords
    elif np.allclose(axis, [0., 0., -1.]):
        return rotate_around_ax(coords, [1., 0., 0.], np.pi)

    axis = np.asarray(axis, dtype = float)
    axis /= np.linalg.norm(axis) 
    rotation_axis = np.cross(axis, np.array([0, 0, 1]))
    
    #angle of rotation
    cos_theta = np.dot(axis, np.array([0, 0, 1]))
    sin_theta = np.linalg.norm(rotation_axis)
    
    angle = np.arctan2(sin_theta, cos_theta)
    
    #normalizing the rot axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    return rotate_around_ax(coords, rotation_axis, angle)