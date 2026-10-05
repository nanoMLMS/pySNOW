import numpy as np
from scipy.spatial.transform import Rotation

from snow.descriptors.shape_descriptors import center_of_mass as com, geometric_com as gcom
from snow.misc.constants import mass

def ax_from_two_points(coord_pt_1, coord_pt_2):
    """
    get the vector connecting two points, oriented from the first to the second.
    
    Parameters
    ----------
    coord_pt_1 : np.ndarray or list
        coordinates of the first point
    coord_pt_2 : np.ndarray or list
        coordinates of the second point
    
    Returns
    -------
    ax_connecting : np.ndarray
        vector connecting the two points
    
    """

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

    Parameters
    ----------
    coords : np.ndarray
        Array of atomic coordinates
    elements : list
        List of element symbols corresponding to the atoms. Default to None.
        If None, all positions will have the same weight in the calculation of the center of mass
    
    Returns
    -------
    new_coords : np.ndarray 
        shifted coords
    """

    if elements is not None:
        return coords - com(elements, coords)
    else:
        return coords - gcom(coords)

def rotate_around_ax(coords, axis, angle):
    """
    Rotate coordinates around a given axis by a given angle (in radians).

    Parameters
    ----------
    coords : array-like, shape (..., 3)
        Coordinates to rotate.
    axis : array-like, shape (3,)
        Rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    new_coords : np.ndarray
        Rotated coordinates, same shape as input.
    """
    
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
    Rotates the system so that the provided axis is aligned with the z=(0,0,1) axis

    Parameters
    ----------
    coords : np.ndarray
        Array of the atomic coordinates
    axis : np.ndarray
        axis to become the new z-axis of the coordinates

    Returns
    -------
    new_coords : np.ndarray
        The transformed coordinates
    """

    #two possible bad cases
    if np.allclose(axis, [0., 0., 1.]):
        return coords
    elif np.allclose(axis, [0., 0., -1.]):
        return rotate_around_ax(coords, [1., 0., 0.], np.pi)

    axis = np.asarray(axis, dtype = float)
    axis = axis / np.linalg.norm(axis) 
    rotation_axis = np.cross(axis, np.array([0, 0, 1]))
    
    #angle of rotation
    cos_theta = np.dot(axis, np.array([0, 0, 1]))
    sin_theta = np.linalg.norm(rotation_axis)
    
    angle = np.arctan2(sin_theta, cos_theta)
    
    #normalizing the rot axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    return rotate_around_ax(coords, rotation_axis, angle)

def align_z_to_axis(coords: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """ 
    Rotates the system so that the original z-axis of the system 
    is aligned with the provided axis

    Parameters
    ----------
    coords : np.ndarray
        Array of the atomic coordinates
    axis : np.ndarray
        axis to become the new z-axis of the coordinates

    Returns
    -------
    new_coords : np.ndarray
        The transformed coordinates
    """

    #two possible bad cases
    if np.allclose(axis, [0., 0., 1.]):
        return coords
    elif np.allclose(axis, [0., 0., -1.]):
        return rotate_around_ax(coords, [1., 0., 0.], np.pi)

    axis = np.asarray(axis, dtype = float)
    axis /= np.linalg.norm(axis) 
    rotation_axis = np.cross(np.array([0, 0, 1]), axis)
    
    #angle of rotation
    cos_theta = np.dot(axis, np.array([0, 0, 1]))
    sin_theta = np.linalg.norm(rotation_axis)
    
    angle = np.arctan2(sin_theta, cos_theta)
    
    #normalizing the rot axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    return rotate_around_ax(coords, rotation_axis, angle)

def eckart_frame(el, coords, ref_coords):
    """
    Applies Eckart conditions to the provided frame, removing translations and 
    minimizing rigid rotations with respect to a reference frame.

    Parameters
    ----------
    el : list[str]
        list of chemical symbols of atoms in the system
    coords : ndarray
        xyz coordinates of atoms in the system for the current frame
    ref_coords : ndarray
        xyz coordinates of atoms in the system for the reference frame (assuming the elements list remains consistent)

    Returns
    -------
    new_coords : ndarray
        xyz coordinates of the provided frame with applied eckart conditions.

    """
    
    #subtract com
    coords = coords - com(el, coords)
    ref_coords = ref_coords - com(el, ref_coords)

    masses = np.array([mass[e] for e in el])
    
    #find the best rotation that matches the two frames
    R, _ = Rotation.align_vectors(ref_coords, coords, weights=masses)
    new_coords = R.apply(coords)
    return new_coords


def rotation_matrix_x(angle_rad):
    """
    Create a rotation matrix around the X axis.

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    return np.array(
        [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
    )


def rotation_matrix_y(angle_rad):
    """
    Create a rotation matrix around the Y axis.

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    return np.array(
        [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
    )


def rotation_matrix_z(angle_rad):
    """
    Create a rotation matrix around the Z axis.

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    return np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )


def create_rotation_matrix(rx, ry, rz):
    """
    Create a combined rotation matrix from rotations around X, Y, and Z axes.

    The rotations are applied in the order: X, Y, Z (so the combined matrix
    is R = Rz * Ry * Rx, meaning Rx is applied first, then Ry, then Rz).

    Parameters
    ----------
    rx : float
        Rotation angle around X axis in radians.
    ry : float
        Rotation angle around Y axis in radians.
    rz : float
        Rotation angle around Z axis in radians.

    Returns
    -------
    np.ndarray
        3x3 combined rotation matrix.
    """
    Rx = rotation_matrix_x(rx)
    Ry = rotation_matrix_y(ry)
    Rz = rotation_matrix_z(rz)
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def create_transformation_matrix(dx, dy, dz, rx, ry, rz):
    """
    Create a 4x4 homogeneous transformation matrix combining rotation and translation.

    Parameters
    ----------
    dx : float
        Translation along X axis.
    dy : float
        Translation along Y axis.
    dz : float
        Translation along Z axis.
    rx : float
        Rotation angle around X axis in radians.
    ry : float
        Rotation angle around Y axis in radians.
    rz : float
        Rotation angle around Z axis in radians.

    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix.
    """
    R = create_rotation_matrix(rx, ry, rz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [dx, dy, dz]
    return T


def apply_transformation(points, transformation_matrix):
    """
    Apply a transformation matrix to a set of 3D points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 3) containing n 3D points.
    transformation_matrix : np.ndarray
        4x4 homogeneous transformation matrix.

    Returns
    -------
    np.ndarray
        Transformed points of shape (n, 3).
    """
    n_points = len(points)
    homogeneous_points = np.ones((n_points, 4))
    homogeneous_points[:, :3] = points
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)
    return transformed_points[:, :3]


def transform_points(points, dx=0, dy=0, dz=0, rx=0, ry=0, rz=0):
    """
    Transform points using translation and rotation parameters.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 3) containing n 3D points.
    dx : float, default 0
        Translation along X axis.
    dy : float, default 0
        Translation along Y axis.
    dz : float, default 0
        Translation along Z axis.
    rx : float, default 0
        Rotation angle around X axis in radians.
    ry : float, default 0
        Rotation angle around Y axis in radians.
    rz : float, default 0
        Rotation angle around Z axis in radians.

    Returns
    -------
    np.ndarray
        Transformed points of shape (n, 3).
    """
    T = create_transformation_matrix(dx, dy, dz, rx, ry, rz)
    return apply_transformation(points, T)


def degrees_to_radians(degrees):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    degrees : float or array-like
        Angle(s) in degrees.

    Returns
    -------
    float or np.ndarray
        Angle(s) in radians.
    """
    return np.radians(degrees)
