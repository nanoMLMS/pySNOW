from collections import Counter

import numpy as np
from scipy.optimize import minimize

from snow.transform.rototranslation import align_z_to_axis
from snow.descriptors.shape_descriptors import geometric_com as gcom


def prepare_data(data_list):
    return Counter([round(val, 2) for val in data_list if val != 0])


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector v returns a new vector of unit length

    Parameters
    ----------
    v : np.ndarray
        A vector defined as a numpy array

    Returns
    -------
    np.ndarray
        A vecotr of unit length.
    """
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Generates a rotation matrix for a rotation of a specific angle in radians around an axis

    Parameters
    ----------
    axis : np.ndarray
        Vecotr describing the axis along which the rotation matrix is constructed
    angle_rad : float
        Angle of rotation in radians

    Returns
    -------
    np.ndarray
        3x3 rotation matrix for the considered rotation
    """
    axis = normalize(axis)
    ux, uy, uz = axis
    ct = np.cos(angle_rad)
    st = np.sin(angle_rad)
    return np.array(
        [
            [
                ct + ux**2 * (1 - ct),
                ux * uy * (1 - ct) - uz * st,
                ux * uz * (1 - ct) + uy * st,
            ],
            [
                uy * ux * (1 - ct) + uz * st,
                ct + uy**2 * (1 - ct),
                uy * uz * (1 - ct) - ux * st,
            ],
            [
                uz * ux * (1 - ct) - uy * st,
                uz * uy * (1 - ct) + ux * st,
                ct + uz**2 * (1 - ct),
            ],
        ]
    )


def center_of_mass(coords):
    return np.mean(coords, axis=0)


def generate_geometry(point, height, angle_deg, bond_length, center, ref_vec):
    normal = normalize(point - center)
    pt_n = point + height * normal
    angle_rad = np.deg2rad(angle_deg)
    rotation = rotation_matrix(np.cross(ref_vec, normal), angle_rad)
    vec_o = rotation.dot(-normal)
    pt_o = pt_n + bond_length * vec_o
    return pt_n, pt_o


def add_diatomic_molecule_from_gcn_list(
    pairs,
    nanoparticle,
    height,
    angle,
    bond_length,
    center,
    ref_vec,
    atom1="N",
    atom2="O",
):
    adsorbates = []
    for gcn_val, coords in pairs:
        pt_n, pt_o = generate_geometry(
            coords, height, angle, bond_length, center, ref_vec
        )
        adsorbates.append((atom1, pt_n, gcn_val))
        adsorbates.append((atom2, pt_o, gcn_val))
    return adsorbates


def add_monatomic_adsorbate_from_gcn_list(pairs, nanoparticle, height, atom1="N"):
    center = center_of_mass(nanoparticle)
    adsorbates = []
    for gcn_val, coords in pairs:
        normal = normalize(coords - center)
        pt = coords + height * normal
        adsorbates.append((atom1, pt, gcn_val))
    return adsorbates


def get_unique_gcn_coords(sites, gcn_vals):
    gcn_data = sorted(
        [(g, s) for g, s in zip(gcn_vals, sites) if g > 0], key=lambda x: round(x[0], 2)
    )
    gcn_coords = {}
    for g, coords in gcn_data:
        g = round(g, 2)
        if g not in gcn_coords:
            gcn_coords[g] = coords
    return gcn_coords


def get_all_gcn_coords_in_range(site_lists, gcn_lists, gcn_min, gcn_max):
    result = []
    for sites, gcns in zip(site_lists, gcn_lists):
        for gcn, coord in zip(gcns, sites):
            if gcn_min <= gcn <= gcn_max:
                result.append((round(gcn, 2), coord))
    return result


def apply_occupation(pairs, occ):
    if not pairs:
        return []
    if occ == 0:
        return [pairs[0]]
    n = max(1, int(len(pairs) * occ / 100))
    return pairs[:n]


#gibo stuff

def add_molecule(el: list[str],
                coords: np.ndarray, 
                site: np.ndarray, 
                direction: np.ndarray, 
                distance: float, 
                el_molecule: list[str], 
                coords_molecule: np.ndarray):
    """
    add a molecule at a distance from a given site and along a given direction. 
    The molecule will be taken as provided to the function and aligned to the direction vector (basically 
    rotate of an angle that aligns the (0,0,1) vector with direction). It is thus good practice to 
    leave the anchor atom of the molecule in the origin, and orient it as you want it to be placed.
    Future implementation: add the angle at which the molecule is placed wrt direction.
    
    Parameters
    ----------
    
    
    
    
    Returns
    -------
    
    """

    norm_direction = np.asarray(direction, dtype=float)
    norm_direction /= np.linalg.norm(norm_direction)

    shift = distance*norm_direction

    coords_molecule = align_z_to_axis(coords_molecule, norm_direction)
    coords_adsorbed_molecule = np.array([coord_mol + site + shift for coord_mol in coords_molecule])

    el = el + el_molecule
    coords = np.vstack([coords, coords_adsorbed_molecule])

    return el, coords


def locally_normal_direction(coords: np.ndarray, site: np.ndarray, cutoff: float):
    """
    get a direction to place adsorbed molecules on nanoparticles and surfaces
    by computing the line connecting the adsorption site 
    and the (geometric) center of mass of the local environment of the site."""

    neighbours = np.array([coord for coord in coords if (np.linalg.norm(coord-site) < cutoff) ])
    #capire come togliere i due siti da bridge
    if neighbours is None:
        raise Exception("cutoff is too small in locally_normal_direction, no neighbours found to define a locally normal direction.")
    
    direction = site - gcom(neighbours)
    if np.all( direction == np.array([0.,0.,0.]) ):
        raise Exception("cutoff is too small in locally_normal_direction, not enough neighbours found to define a locally normal direction.")
    direction = direction / np.linalg.norm(direction)

    return direction 







# #some helper function for the optimization process
# def angles_to_unit_vector(theta_phi):
#     theta, phi = theta_phi #need a single array of variables for optimization
#     return np.array([
#         np.sin(theta) * np.cos(phi),
#         np.sin(theta) * np.sin(phi),
#         np.cos(theta)
#     ])

# #the main function to minimize: sum of distances
# def objective(theta_phi, neigh_coords):
#     p = angles_to_unit_vector(theta_phi)
#     return -np.sum(np.linalg.norm(neigh_coords - p, axis=1))

# def gradient(theta_phi):
#     "analytic gradient of the objective wrt theta, phi"

#     theta, phi = theta_phi
#     st, ct = np.sin(theta), np.cos(theta)
#     sp, cp = np.sin(phi),   np.cos(phi)

# def site_direction_gcn(el, coords, index, site, cutoff):
#     """get the direction to adsorb molecules on a gcn (atomic) site by 
#     maximizing the distance of the adsorbed atom from atoms in proximity of the site.
#     """

#     if index is not None:
#         site = coords[index]

#     #implict constraint: we only care about a direction (so an angle)
#     initial_direction =  np.array(site - gcom(coords))
#     initial_direction /= np.linalg.norm(initial_direction)
#     shifted_coords = coords-site

#     #select only neighbours in the cutoff radius
#     distances_to_site = np.linalg.norm(shifted_coords, axis=1)
#     mask = distances_to_site < cutoff
#     # exclude the site atom itself (distance == 0)
#     if index is not None:
#         mask[index] = False
#     neighbour_coords = shifted_coords[mask]

#     if len(neighbour_coords) == 0:
#         # no neighbours: fall back to the bulk-away direction
#         return initial_direction
    
#     #convert initial direction to angles:
#     d = initial_direction
#     theta_0 = np.arccos(d[2])
#     phi_0   = np.arctan2(d[1], d[0])

#     #optimize
#     result = minimize(
#         objective, [theta_0, phi_0],
#         jac=gradient,
#         method='BFGS',
#         options={'gtol': 1e-10, 'maxiter': 500}
#     )