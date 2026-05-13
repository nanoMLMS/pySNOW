from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist
from snow.descriptors.utils import pbc_distance

from snow.misc.rototranslation import align_z_to_axis, rotate_around_ax
from snow.descriptors.shape_descriptors import geometric_com as gcom, geometric_com_pbc as gcom_pbc
from snow.descriptors.coordination import coordination_number, bridge_gcn, three_hollow_gcn, four_hollow_gcn


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
                phi: float=0.,
                molecule_only: bool=False):
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
    molecule_only: bool
        Only return the elements list and coords array of the molecule rather than those of the entire system. Default to False
    

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

    if molecule_only:
        return el_molecule, coords_adsorbed_molecule

    else:
        el = el + el_molecule
        coords = np.vstack([coords, coords_adsorbed_molecule])

        return el, coords


def get_local_neighbours(coords, center, cutoff, pbc=False, box=None):
    """get neighbours of a specific inside inside a given cutoff"""


    if pbc:
        print('pbc are not implemented yet')
        neighbours = None
        # if box is None:
        #     raise ValueError('A box must be provided if pbc are enabled')

        # neighbours = np.array([coord for coord in coords if pbc_distance(coord, center, box) < cutoff])

    else:
        neighbours = np.array([coord for coord in coords if (np.linalg.norm(coord-center) < cutoff) ])

    if neighbours is None:
        raise Exception("cutoff is too small in locally_normal_direction, no neighbours found to define a locally normal direction.")

    return neighbours

def locally_normal_direction(coords: np.ndarray, site: np.ndarray, cutoff: float, pbc=False, box=None):
    """
    get a direction to place adsorbed molecules on nanoparticles and surfaces
    by computing the line connecting the adsorption site 
    and the (geometric) center of mass of the local environment of the site."""

    #should we remove the two atoms from birdge sites neighbour calculations?

    #get neighbours of the site
    neighbours = get_local_neighbours(coords, site, cutoff, pbc, box)

    #compute a locally normal direction
    if pbc:
        direction = site - gcom_pbc(neighbours, box)

    else:
        direction = site - gcom(neighbours)

    if np.all( direction == np.array([0.,0.,0.]) ):
        raise Exception("cutoff is too small in locally_normal_direction, not enough neighbours found to define a locally normal direction (the adsorption site coincides with the geometric center of mass of the neighbourhood).")

    direction = direction / np.linalg.norm(direction)

    return direction 


def triplet_normal(coords: np.ndarray, triplet: list[int], cutoff, pbc=False, box=None):
    """
    get the normal direction wrt to a plane defined by the three atoms making up a triplet.
    You still need a cutoff to define a lcoal atomic environmoment - this is used to compute
    the 'outside' orientation with respect to the local surface.
    """

    p1, p2, p3 = coords[triplet[0]], coords[triplet[1]], coords[triplet[2]]

    ax1 = p2-p1
    ax2 = p3-p1

    if pbc:

        print('pbc not implemented yet')
        pass

        # if type(box) == list and len(box)==3 or type(box)==np.ndarray and (box.shape==(3,) or box.shape==(3,1)):
        #     box = np.asarray([[box[0],0.,0.], [0., box[1], 0.], [0., 0., box[2]] ])
        # elif box.shape !=(3,3):
        #     raise Exception('Please provide the box as either a (3,3) or (3,) array or list.')

        # inv_box = np.linalg.inv(box)
        # frac_1 = ax1 @ inv_box
        # frac_2 = ax2 @ inv_box
        # frac_1 -= np.round(frac_1)
        # frac_2 -= np.round(frac_2)
        # ax1 = frac_1 @ box
        # ax2 = frac_2 @ box


    normal = np.cross(ax1, ax2)
    normal /= np.linalg.norm(normal)

    #a primitive check on orientation:
    site = (p1+p2+p3)/3.

    neighs = get_local_neighbours(coords, site, cutoff, pbc, box)

    if pbc:
        ref_gcom = gcom_pbc(coords, box)
    else:
        ref_gcom = gcom(coords)

    if np.dot(normal, site-gcom(coords)) >= 0:
        return normal
    else:
        return normal*-1.


def fourplet_normal(coords: np.ndarray, fourplet: list[int], cutoff, pbc=False, box=None):
    """
    get the normal direction wrt to the surface defined by the atoms in a fourplet.
    If the four atoms are not on a single plane, an average over the planes defined 
    by the possible triplets is returned.
    You still need a cutoff to define a lcoal atomic environmoment - this is used to compute
    the 'outside' orientation with respect to the local surface.
    """

    p1, p2, p3, p4 = coords[fourplet[0]], coords[fourplet[1]], coords[fourplet[2]], coords[fourplet[3]]
    
    n1 = triplet_normal(coords, [fourplet[0], fourplet[1], fourplet[2]], cutoff, pbc, box)
    n2 = triplet_normal(coords, [fourplet[0], fourplet[1], fourplet[3]], cutoff, pbc, box)
    n3 = triplet_normal(coords, [fourplet[1], fourplet[2], fourplet[3]], cutoff, pbc, box)
    n4 = triplet_normal(coords, [fourplet[0], fourplet[2], fourplet[3]], cutoff, pbc, box)

    normal = n1+n2+n3+n4
    return normal / np.linalg.norm(normal)



def check_overlapping(el: list[str], coords: np.ndarray, atomic_radii: dict):
    """
    Check if any atom in coords is overlapping with any other atom.
    TODO: add pbc
    """

    radii = np.array([atomic_radii[s] for s in el])          # (N,)
    radii_sum = radii[:, None] + radii[None, :]              # (N, N) pairwise sums

    dist_matrix = cdist(coords, coords)                      # (N, N) pairwise distances

    # ignore self-pairs by setting diagonal to infinity
    np.fill_diagonal(dist_matrix, np.inf)

    return np.any(dist_matrix < radii_sum)#, np.where(dist_matrix < radii_sum)

def cover_surface(el, coords, cutoff, thr_cn, el_adsorbate, coords_adsorbate, distance, atomic_radii, ratio=1.0, sites_type: str = 'atop', theta=0., phi=0., pbc=False, box=None):
    """cover as much as possible a system with molecules while avoiding overlapping. Eventually you can decide to
    only keep a given fraction of all the molecules with the ratio argument. The order in which sites will tentatively be covered
    by a molecule is random. The orientation of the molecule can be random or fixed (only fixed for now). The sites can be chosen as atop, bridge, 
    three-hollow or four-hollow."""

    assert 0.0 <= ratio <= 1.0

    #get sites and normal directions
    if sites_type == 'atop':
        cns = coordination_number(coords, cutoff)
        sites = np.asarray([coord for coord, cn in zip(coords, cns) if cn<thr_cn ])
        directions = np.asarray([ locally_normal_direction(coords, site, cutoff, pbc, box) for site in sites ])
    elif sites_type == 'bridge':
        sites, pairs, bgcns = bridge_gcn(coords, cutoff, thr_cn)
        directions = np.asarray([ locally_normal_direction(coords, site, cutoff, pbc, box) for site in sites ])
    elif sites_type == 'three-hollow':
        sites, triplets, tgcns = three_hollow_gcn(coords, cutoff, thr_cn)
        directions = np.asarray([ triplet_normal(coords, triplet, cutoff, pbc, box) for triplet in triplets ])
    elif sites_type == 'four-hollow':
        sites, fourplets, fgcns = four_hollow_gcn(coords, cutoff, thr_cn)
        directions = np.asarray([ fourplet_normal(coords, fourplet, cutoff, pbc, box) for fourplet in fourplets ])
    else:
        raise Exception(f'sites_type {sites_type} not recognized.')

    #iterate randomly over sites and try to place a molecule there
    indexes = np.random.permutation(len(sites))

    test_el     = el
    test_coords = coords

    el_ads_list = []
    coords_ads_list = []

    overlap_count = 0

    #add adsorbates to full coverage
    for i in indexes:
        site = sites[i]
        direction = directions[i]
        #for now fixed orientation
        el_ads, coords_ads = add_molecule(test_el, test_coords, site, direction, distance, el_adsorbate, coords_adsorbate, theta, phi, molecule_only=True)

        if not check_overlapping(test_el+el_ads, np.vstack((test_coords,coords_ads)), atomic_radii):
            test_el = test_el + el_ads
            test_coords = np.vstack((test_coords,coords_ads))
            el_ads_list.append(el_ads)
            coords_ads_list.append(coords_ads)
        
        else:
            overlap_count +=1
    
    print('tries that resulted in overlapping atoms:',overlap_count)


    #only keep a (random) ratio fraction of the full coverage
    if ratio < 1.:
        test_el = el
        test_coords = coords
        keep_indexes = np.random.choice( len(el_ads_list), int(np.round(ratio*len(el_ads_list))), replace=False )
        for i in keep_indexes:
            test_el = test_el + el_ads_list[i]
            test_coords = np.vstack((test_coords, coords_ads_list[i]))
    
        print('only kept',len(keep_indexes),'adsorbates out of',len(el_ads_list),'possible ones ')

    return test_el, test_coords
