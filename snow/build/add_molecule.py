from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist

from snow.transform.rototranslation import align_z_to_axis, rotate_around_ax
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
#TODO: extend to pbc

def add_molecule(el: list[str],
                coords: np.ndarray, 
                site: np.ndarray, 
                direction: np.ndarray, 
                distance: float, 
                el_molecule: list[str], 
                coords_molecule: np.ndarray,
                theta: float=0.,
                phi: float=0.):
    """
    Add a molecule at a distance from a given site and along a given direction. 
    Regarding the final orientation: the molecule will be taken as provided to the function, 
    rotated by theta (angle wrt direction vector around the x axis) and phi (angle around the direction vector),
    and placed at a distance from the adsorption site along the given direction. 
    
    Parameters
    ----------
    el: list[str]
        list of chemical symbols of atoms in the system.
    coords: np.ndarray
        positions of atoms in the system
    site: np.ndarray
        coordinates of the adsorption site. The anchor atom will be placed at a distance=distance from this site
    direction: np.ndarray
        direction (3-d vector) to place the adsorbed molecule in
    distance: float 
        distance at which to place the (anchor atom of the) molecule
    el_molecule: list[str]
        list of chemical symbols of the atoms in the molecule
    coords_molecule: np.ndarray
        cooridnates of the atoms making up the molecule
    theta: float
        angle in radians with respect to the direction vector. The molecule will be rotated around the original
        x axis of an angle theta, resulting in an adsorbed configuration with theta being the angle between the
        direction vector and the initial z axis of your molecule
    phi: float
        angle in radians to rotate the molecule around the direction vector.
    

    Returns
    -------
        Tuple[list, np.ndarray]
        The list of chemical symbols and the np.ndarray for the positions of the atoms in the new system comprising the adsorbed molecule.
    
    """

    #rotate the molecule 
    if theta != 0.:
        coords_molecule = rotate_around_ax(coords_molecule, [1.,0.,0.], theta)
    if phi != 0.:
        coords_molecule = rotate_around_ax(coords_molecule, [0.,0.,1.], phi)

    #normalize direction
    norm_direction = np.asarray(direction, dtype=float)
    norm_direction /= np.linalg.norm(norm_direction)

    #compute shift
    shift = distance*norm_direction

    #place molecule
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


def triplet_normal(coords: np.ndarray, triplet: list[int]):
    """
    get the normal direction wrt to a plane defined by the three atoms making up a triplet.
    """

    p1, p2, p3 = coords[triplet[0]], coords[triplet[1]], coords[triplet[2]]

    ax1 = p2-p1
    ax2 = p3-p1

    normal = np.cross(ax1, ax2)
    normal /= np.linalg.norm(normal)

    #a primitive check on orientation:
    #this probably only works with convex nanoparticles.
    site = (p1+p2+p3)/3.
    if np.dot(normal, site-gcom(coords)) >= 0:
        return normal
    else:
        return normal*-1.


def fourplet_normal(coords: np.ndarray, fourplet: list[int]):
    """
    get the normal direction wrt to the surface defined by the atoms in a fourplet.
    If the four atoms are not on a single plane, an average over the planes defined 
    by the possible triplets is returned.
    """

    p1, p2, p3, p4 = coords[fourplet[0]], coords[fourplet[1]], coords[fourplet[2]], coords[fourplet[3]]
    
    n1 = triplet_normal(coords, [fourplet[0], fourplet[1], fourplet[2]])
    n2 = triplet_normal(coords, [fourplet[0], fourplet[1], fourplet[3]])
    n3 = triplet_normal(coords, [fourplet[1], fourplet[2], fourplet[3]])
    n4 = triplet_normal(coords, [fourplet[0], fourplet[2], fourplet[3]])

    normal = n1+n2+n3+n4
    return normal / np.linalg.norm(normal)

def cover_surface(el, coords, thr_cn, ratio=1.0, sites: str = 'atop', ):

    return

def check_overlapping(el: list[str], coords: np.ndarray, atomic_radii: dict):
    """
    Check if any atom in coords is overlapping with any other atom.
    """

    radii = np.array([atomic_radii[s] for s in el])          # (N,)
    radii_sum = radii[:, None] + radii[None, :]              # (N, N) pairwise sums

    dist_matrix = cdist(coords, coords)                      # (N, N) pairwise distances

    # ignore self-pairs by setting diagonal to infinity
    np.fill_diagonal(dist_matrix, np.inf)

    return np.any(dist_matrix < radii_sum)#, np.where(dist_matrix < radii_sum)