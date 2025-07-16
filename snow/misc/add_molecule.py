import numpy as np
from collections import Counter

from snow.descriptors.gcn import *
from snow.lodispp.pp_io import *

def prepare_data(data_list):
    return Counter([round(val, 2) for val in data_list if val != 0])

def normalize(v):
    """Normalizes a vector v

    Parameters
    ----------
    v : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    norm = np.linalg.norm(v)
    return v / norm if norm else v

def rotation_matrix(axis, angle_rad):
    """Generates a rotation matrix for a rotation of a specific angle in radians around an axis

    Parameters
    ----------
    axis : _type_
        _description_
    angle_rad : float
        Angle of rotation

    Returns
    -------
    _type_
        _description_
    """
    axis = normalize(axis)
    ux, uy, uz = axis
    ct = np.cos(angle_rad)
    st = np.sin(angle_rad)
    return np.array([
        [ct + ux**2*(1-ct),    ux*uy*(1-ct)-uz*st, ux*uz*(1-ct)+uy*st],
        [uy*ux*(1-ct)+uz*st, ct + uy**2*(1-ct),    uy*uz*(1-ct)-ux*st],
        [uz*ux*(1-ct)-uy*st, uz*uy*(1-ct)+ux*st, ct + uz**2*(1-ct)]
    ])

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

def add_diatomic_molecule_from_gcn_list(pairs, nanoparticle, height, angle, bond_length, center, ref_vec, atom1='N', atom2='O'):
    adsorbates = []
    for gcn_val, coords in pairs:
        pt_n, pt_o = generate_geometry(coords, height, angle, bond_length, center, ref_vec)
        adsorbates.append((atom1, pt_n, gcn_val))
        adsorbates.append((atom2, pt_o, gcn_val))
    return adsorbates

def add_monatomic_adsorbate_from_gcn_list(pairs, nanoparticle, height, atom1='N'):
    center = center_of_mass(nanoparticle)
    adsorbates = []
    for gcn_val, coords in pairs:
        normal = normalize(coords - center)
        pt = coords + height * normal
        adsorbates.append((atom1, pt, gcn_val))
    return adsorbates

    
def get_unique_gcn_coords(sites, gcn_vals):
    gcn_data = sorted(
        [(g, s) for g, s in zip(gcn_vals, sites) if g > 0],
        key=lambda x: round(x[0], 2)
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