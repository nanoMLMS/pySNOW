import numpy as np
from snow.lodispp.pp_io import read_xyz
from snow.lodispp.utils import center_of_mass

def rotate_translate_system(index_frame: int, coords: np.ndarray, symmetry_axis: np.ndarray):
    """
    
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
    
