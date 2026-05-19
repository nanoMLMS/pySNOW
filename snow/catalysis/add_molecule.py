import numpy as np
from scipy.spatial.distance import cdist

from snow.misc.rototranslation import align_z_to_axis, rotate_around_ax
from snow.descriptors.shape_descriptors import geometric_com as gcom
from snow.descriptors.coordination import coordination_number, bridge_gcn, three_hollow_gcn, four_hollow_gcn

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
    and placed at a distance from the adsorption site along the given direction. Provide its coordinates 
    so that the anchor atom/site is in the origin.
    
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
    theta: float, deafult 0.
        angle in radians with respect to the direction vector. The molecule will be rotated around the original
        x axis of an angle theta, resulting in an adsorbed configuration with theta being the angle between the
        direction vector and the initial z axis of your molecule
    phi: float, default 0.
        angle in radians to rotate the molecule around the direction vector.
    molecule_only: bool, default False
        Only return the elements list and coords array of the molecule rather than those of the entire system.
    

    Returns
    -------
    tuple
        - list[str] : the list of chemnical symbols of atoms in the system with the adsorbed molecule
        - np.ndarray : coordinates of atoms in the system with the adsorbed molecule
    
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


def get_local_neighbours(coords, site, cutoff):
    """get coordinates of atoms inside a given cutoff wrt to a given site
    
    Parameters
    ----------
    coords : np.ndarray
        coordinates of the atoms in the system
    site : np.ndarray
        site around which the cutoff sphere is deifned
    cutoff : float
        cutoff value to distinguish atoms inside or outside the bubble
    
    Returns
    -------
    neighbours : list
        list of coordinates of atoms inside the cutoff sphere wrt to center """


    neighbours = np.array([coord for coord in coords if (np.linalg.norm(coord-site) < cutoff) ])

    if neighbours is None:
        raise Exception("cutoff is too small in locally_normal_direction, no neighbours found to define a locally normal direction.")

    return neighbours

def locally_normal_direction(coords: np.ndarray, site: np.ndarray, cutoff: float):
    """
    get a locally normal orinetation with respect to a site on a surface.

    get a direction to place adsorbed molecules on nanoparticles and surfaces
    by computing the line connecting the adsorption site 
    and the (geometric) center of mass of the local environment of the site.
    
    Parameters
    ----------
    coords : np.ndarray
        coordinates of the atoms in the system
    site : np.ndarray
        cordinates of the site
    cutoff : float
        cutoff value to define the local atomic environment of the site 

    Returns
    -------
    np.ndarray
        the vector pointing in the locally normal direction.
    """

    #should we remove the two atoms from birdge sites neighbour calculations?

    #get neighbours of the site
    neighbours = get_local_neighbours(coords, site, cutoff)

    #compute a locally normal direction

    direction = site - gcom(neighbours)

    if np.all( direction == np.array([0.,0.,0.]) ):
        raise Exception("cutoff is too small in locally_normal_direction, not enough neighbours found to define a locally normal direction (the adsorption site coincides with the geometric center of mass of the neighbourhood).")

    direction = direction / np.linalg.norm(direction)

    return direction 


def triplet_normal(coords: np.ndarray, triplet: list[int], cutoff):
    """
    get the normal direction wrt to a plane defined by the three atoms making up a triplet.

    You still need a cutoff to define a lcoal atomic environmoment - this is used to compute
    the 'outside' orientation with respect to the local surface.

    Parameters
    ----------
    coords : np.ndarray
        coordinates of the atoms in the system
    triplet : list[int]
        list of the ids of the atoms making up the triplet
    cutoff : float
        cutoff value to define the local atomic environment of the atom to correctly point outwards wrt to the surface

    Returns
    -------
    np.ndarray
        the vector pointing in the locally normal direction.

    """

    p1, p2, p3 = coords[triplet[0]], coords[triplet[1]], coords[triplet[2]]

    ax1 = p2-p1
    ax2 = p3-p1

    normal = np.cross(ax1, ax2)
    normal /= np.linalg.norm(normal)

    #a check on orientation:
    site = (p1+p2+p3)/3.
    neighs = get_local_neighbours(coords, site, cutoff)

    if np.dot(normal, site-gcom(neighs)) >= 0:
        return normal
    else:
        return normal*-1.


def fourplet_normal(coords: np.ndarray, fourplet: list[int], cutoff):
    """
    get the local normal direction for a fourplet of atoms.

    get the normal direction wrt to the surface defined by the atoms in a fourplet.
    If the four atoms are not on a single plane, an average over the planes defined 
    by the possible triplets is returned.
    You still need a cutoff to define a local atomic environmoment - this is used to compute
    the 'outside' orientation with respect to the local surface.

    Parameters
    ----------
    coords : np.ndarray
        coordinates of the atoms in the system
    fourplet : list[int]
        list of the ids of the atoms making up the fourplet
    cutoff : float
        cutoff value to define the local atomic environment of the atom to correctly point outwards wrt to the surface

    Returns
    -------
    np.ndarray
        the vector pointing in the locally normal direction.
    """

    p1, p2, p3, p4 = coords[fourplet[0]], coords[fourplet[1]], coords[fourplet[2]], coords[fourplet[3]]
    
    n1 = triplet_normal(coords, [fourplet[0], fourplet[1], fourplet[2]], cutoff)
    n2 = triplet_normal(coords, [fourplet[0], fourplet[1], fourplet[3]], cutoff)
    n3 = triplet_normal(coords, [fourplet[1], fourplet[2], fourplet[3]], cutoff)
    n4 = triplet_normal(coords, [fourplet[0], fourplet[2], fourplet[3]], cutoff)

    normal = n1+n2+n3+n4
    return normal / np.linalg.norm(normal)



def check_overlapping(el: list[str], coords: np.ndarray, atomic_radii: dict):
    """
    Check if any atom in the system is overlapping with any other atom.

    Parameters
    ----------
    el : list[str]
        list of chemical symbols of atoms in the system
    coords : np.ndarray
        coordinates of atoms in the system
    atomic_radii : dict
        a dictionary in the form { element : radius } to check that generated
        geometries do not have overlapping atoms
    
    Returns
    -------
    bool
        wether atoms are overlapping in this configuration or not
    
    """

    radii = np.array([atomic_radii[s] for s in el])          # (N,)
    radii_sum = radii[:, None] + radii[None, :]              # (N, N) pairwise sums

    dist_matrix = cdist(coords, coords)                      # (N, N) pairwise distances

    # ignore self-pairs by setting diagonal to infinity
    np.fill_diagonal(dist_matrix, np.inf)

    return np.any(dist_matrix < radii_sum)#, np.where(dist_matrix < radii_sum)

def cover_surface(el, coords, cutoff, thr_cn, el_adsorbate, coords_adsorbate, distance, atomic_radii, ratio=1.0, sites_type: str = 'atop', theta=0., phi=0.):
    """
    Cover as much as possible the surface of a system with molecules while avoiding overlapping.

    Eventually you can decide to only keep a given fraction of all the molecules with the ratio argument. 
    The order in which sites will tentatively be covered by a molecule is random. The orientation of the molecule can be specified. 
    The sites can be chosen as atop, bridge, three-hollow or four-hollow.
    
    Parameters
    ----------
    el : list
        list of chemical symbols of atoms in the original system
    coords : np.ndarray
        coordinates of atoms in the original system
    cutoff : float
        cutoff to compute nearest neighbours and (generalized) coordination numbers
    el_adsorbate : list
        list of chemical symbols of atoms in the molecule to be added
    coords_adsorbate : np.ndarray
        coordinates of atoms in the molecule to be added
    distance : float
        distance at which the molecule should be placed from the adsorption site
    atomic_radii : dict
        a dictionary in the form { element : radius } to check that generated
        geometries do not have overlapping atoms
    ratio : float, default 1.0
        ratio of molecules to keep on the surface. Default to 1.0, which means that all molecules
        that were placed stayed in place. If ratio is < 1., only an according fraction of the molecules that
        were placed on the surface are eventually kept and returned by the function.
    sites_type : str, default 'atop'
        decide where the adsorbed molecules should be placed (either 'atop', 'bridge', 'three-hollow', or 'four-hollow')
    theta: float
        angle in radians with respect to the direction vector. The molecule will be rotated around the original
        x axis of an angle theta, resulting in an adsorbed configuration with theta being the angle between the
        direction vector and the initial z axis of your molecule
    phi: float
        angle in radians to rotate the molecule around the direction vector.

    Returns
    -------
    test_el : list
        list of chemical symbols of atoms in the system with the appended molecules
    test_coords : np.ndarray
        coordinates of atoms in the system with the appended molecules
    
    """

    assert 0.0 <= ratio <= 1.0

    #get sites and normal directions
    if sites_type == 'atop':
        cns = coordination_number(coords, cutoff)
        sites = np.asarray([coord for coord, cn in zip(coords, cns) if cn<thr_cn ])
        directions = np.asarray([ locally_normal_direction(coords, site, cutoff) for site in sites ])
    elif sites_type == 'bridge':
        sites, pairs, bgcns = bridge_gcn(coords, cutoff, thr_cn)
        directions = np.asarray([ locally_normal_direction(coords, site, cutoff) for site in sites ])
    elif sites_type == 'three-hollow':
        sites, triplets, tgcns = three_hollow_gcn(coords, cutoff, thr_cn)
        directions = np.asarray([ triplet_normal(coords, triplet, cutoff) for triplet in triplets ])
    elif sites_type == 'four-hollow':
        sites, fourplets, fgcns = four_hollow_gcn(coords, cutoff, thr_cn)
        directions = np.asarray([ fourplet_normal(coords, fourplet, cutoff) for fourplet in fourplets ])
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