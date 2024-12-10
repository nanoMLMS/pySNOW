"""
Contains functions to ocmpute physical properties for the system, such as pressure, potential energy etc.
"""

import numpy as np
from snow.lodispp.pp_io import read_rgl, read_eam
from tqdm import tqdm
from scipy.spatial.distance import pdist
from snow.lodispp.utils import nearest_neighbours
def properties_rgl(coords: np.ndarray, elements: np.ndarray, pot_file: str, dist_mat: np.ndarray, l_pressure: bool):
    """Calculate energy, density and pressure for a given atomic configuration given an interatomic potential parameter file.


    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates (n_atoms x 3).
    elements : np.ndarray
        List of element symbols for each atom.
    pot_file : str
        Path to the Rosato Gupta potential parameter file.
    dist_mat : np.ndarray
        Distance matrix (n_atoms x n_atoms).
    l_pressure : bool
        Flag to calculate pressures.

    Returns
    -------
    tuple
        (density, potential energy, atomic pressures)
    """

    n_atoms = np.shape(coords)[0]
    density = np.zeros(n_atoms)
    pot_energy = np.zeros(n_atoms)
    press_atoms = np.zeros(n_atoms)
    
    # Read potential parameters
    pot_params = read_rgl(filepot=pot_file)  # Assumes this function is implemented elsewhere
    
    at_vol = np.zeros(3)
    
    # Precompute constants
    four_thirds_pi = 4.0 / 3.0 * np.pi
    print("Computing physical properties")
    for i in tqdm(range(n_atoms)):
        eb_i = 0
        er_i = 0
        presr_i = 0
        presb_i = 0
        prsr = 0
        prsb = 0
        for k in range(n_atoms):
            if k != i:
                # Determine interaction type
                if elements[i] == elements[k]:
                    if elements[i] == pot_params["el_a"]:
                        itypik = 0  # A-A interaction
                        at_vol[0] = four_thirds_pi * pot_params["dist"][0]**3
                    elif elements[i] == pot_params["el_b"]:
                        itypik = 1  # B-B interaction
                        at_vol[1] = four_thirds_pi * pot_params["dist"][1]**3
                else:
                    itypik = 2  # A-B interaction
                    at_vol[2] = four_thirds_pi * pot_params["dist"][2]**3
                
                d_ik = dist_mat[i, k]
                d_ik_0 = pot_params["dist"][itypik]
                
                if d_ik <= pot_params["cut_start"]:
                    # Calculate potential and repulsion terms
                    espo = d_ik / d_ik_0 - 1.0
                    pexp = np.exp(-pot_params["p"][itypik] * espo)
                    qqexp = np.exp(-2.0 * pot_params["q"][itypik] * espo)
                    
                    qsi2 = pot_params["qsi"][itypik]**2
                    density[i] += qsi2 * qqexp
                    er_i += pot_params["a"][itypik] * pexp  # Repulsion term
                    
                    prsr = pot_params["p"][itypik] * pot_params["a"][itypik] * pexp
                    
                    prsb = -pot_params["q"][itypik] * qsi2 * qqexp

                elif pot_params["cut_start"] < d_ik <= pot_params["cut_end"]:
                    # Analytical extension for distances in the cutoff range
                    d_ik_m = d_ik - pot_params["cut_end"]
                    d_ik_m2 = d_ik_m**2
                    d_ik_m3 = d_ik_m2 * d_ik_m
                    d_ik_m4 = d_ik_m3 * d_ik_m
                    d_ik_m5 = d_ik_m4 * d_ik_m
                    
                    density[i] += (
                        pot_params["x5"][itypik] * d_ik_m5 +
                        pot_params["x4"][itypik] * d_ik_m4 +
                        pot_params["x3"][itypik] * d_ik_m3
                    )**2
                    er_i += (
                        pot_params["a5"][itypik] * d_ik_m5 +
                        pot_params["a4"][itypik] * d_ik_m4 +
                        pot_params["a3"][itypik] * d_ik_m3
                    )
                    prsr = (
                        5.0 * pot_params["a5"][itypik] * d_ik_m4 +
                        4.0 * pot_params["a4"][itypik] * d_ik_m3 +
                        3.0 * pot_params["a3"][itypik] * d_ik_m2
                    )
                    prsb = (
                        pot_params["x5"][itypik] * d_ik_m5 +
                        pot_params["x4"][itypik] * d_ik_m4 +
                        pot_params["x3"][itypik] * d_ik_m3
                    ) * (
                        5.0 * pot_params["x5"][itypik] * d_ik_m4 +
                        4.0 * pot_params["x4"][itypik] * d_ik_m3 +
                        3.0 * pot_params["x3"][itypik] * d_ik_m2
                    )
                
                if l_pressure:
                    presr_i += prsr * d_ik
                    presb_i += prsb * d_ik

        eb_i = -np.sqrt(density[i])
        density[i] = -1.0 / eb_i
        ener_i = eb_i + er_i
        if l_pressure:
            presb_i *= density[i]

        pot_energy[i] = ener_i
        if l_pressure:
            press_atoms[i] = (presr_i + presb_i) / at_vol[0]
    
    return density, -pot_energy, press_atoms
             
                    

def pair_energy_eam(pot_file: str, coords: np.ndarray) -> float:
    """Computes the energy associated with a pair of atoms

    Parameters
    ----------
    pot_file : str
        Path to the EAM potential file
    coords : np.ndarray
        Coordinates of two atoms of a pair

    Returns
    -------
    float
        Potential energy of a pair
    """
    
    r_ij = pdist(coords)[0]
    
    potential = read_eam(pot_file)
    
    idx_closer_r = np.searchsorted(potential["r"], r_ij)
    
    rho_r = potential["rho_r"][idx_closer_r]
    
    Z_r = potential["Z_r"][idx_closer_r]
    phi_r = 27.2 * 0.529 * Z_r * Z_r / r_ij
    
    idx_closer_rho = np.searchsorted(potential["rho_r"], rho_r)
    
    
    F_rho = potential["F_rho"][idx_closer_rho]
    
    return F_rho + 0.5 * phi_r

def energy_eam(coords: np.ndarray, pot_file: str) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    coords : np.ndarray
        _description_
    pot_file : str
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    
    N_atoms = np.shape(coords)[0]
    
    potential = read_eam(pot_file)
    cut_off = potential["cut_off"]
    
    neigh_list = nearest_neighbours(1, coords = coords, cut_off = cut_off)
    
    pot_en = np.zeros(N_atoms)
    print("Computing energies for each atom")
    for i in tqdm(range (N_atoms)):
        en_i = 0
        for neigh in neigh_list[i]:
            if neigh != i:
                coords_ij = np.array([coords[i], coords[neigh]])
                en_ij = pair_energy_eam(pot_file=pot_file, coords=coords_ij)
                en_i += en_ij
        pot_en[i] = en_i
    return pot_en
        
        
        
        
    
    