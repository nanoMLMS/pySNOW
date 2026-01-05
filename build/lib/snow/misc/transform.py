import numpy as np


def rotate_translate_system(index_frame: int, coords: np.ndarray, symmetry_axis: np.ndarray) -> np.ndarray:
    """ ROtates the system around the origin along a given axis

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
    
    rotation_axis = np.cross(symmetry_axis, np.array([0, 0, 1]))
    
    #angle of rotation
    cos_theta = np.dot(symmetry_axis, np.array([0, 0, 1]))
    sin_theta = np.linalg.norm(rotation_axis)
    
    angle = np.arctan2(sin_theta, cos_theta)
    
    #normalizing the rot axis
    print(np.linalg.norm(rotation_axis))
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
    translation_vector = -rotated_coords[0]  # Porta l'atomo di riferimento all'origine
    translated_coords = rotated_coords + translation_vector

    return translated_coords
    
