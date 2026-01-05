import numpy as np

from snow.lodispp.pp_io import read_rgl
from tqdm import tqdm


def properties(coords: np.ndarray, elements: np.ndarray, pot_file: str, dist_mat: np.ndarray, l_pressure: bool):
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
    
    return density, pot_energy, press_atoms
             
                    

