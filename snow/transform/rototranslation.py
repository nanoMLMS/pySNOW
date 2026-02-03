import numpy as np
from snow.descriptors.shape_descriptors import center_of_mass as com, geometric_com as gcom


def align_to_axis(index_frame: int, coords: np.ndarray, symmetry_axis: np.ndarray) -> np.ndarray:
    """ Rotates the system so that the provided symmetry_axis is aligned with the z=(0,0,1) axis
    and the first atom of the list of coordinates is set at the origin

    Parameters
    ----------
    index_frame : int
        Index of the frame if going through a trajectory, mostly for reference
    coords : np.ndarray
        Array of the atomic coordinates
    symmetry_axis : np.ndarray
        Symmetry axis along which structure will be aligned

    Returns
    -------
    np.ndarray
        The transformed system of coordinates
    """

    symmetry_axis /= np.linalg.norm(symmetry_axis) 
    rotation_axis = np.cross(symmetry_axis, np.array([0, 0, 1]))
    
    #angle of rotation
    cos_theta = np.dot(symmetry_axis, np.array([0, 0, 1]))
    sin_theta = np.linalg.norm(rotation_axis)
    
    angle = np.arctan2(sin_theta, cos_theta)
    
    #normalizing the rot axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                               [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                               [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])
    rotated_coords = np.dot(rotation_matrix, coords.T).T
    translation_vector = -rotated_coords[0]  # reference atom (the first one in the list) is moved to the origin
    translated_coords = rotated_coords + translation_vector

    return translated_coords
    
def center_com(index_frame : int, coords : np.ndarray, elements=None) -> np.ndarray:
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

    coords = np.asarray(coords)

    if elements is not None:
        return coords - com(index_frame, coords, elements)
    else:
        return coords - gcom(index_frame, coords)

def rotate(coords, axis, angle):
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