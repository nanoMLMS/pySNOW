import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist
from snow.io.xyz import read_xyz
from snow.descriptors.coordination import agcn_calculator, nearest_neighbours
from snow.misc.constants import *

"""
Utilities for estimating ORR electrocatalytic activity of Pt nanoparticles.

The module provides routines to compute geometrical descriptors,
surface properties, and electrochemical activity metrics
(specific activity, mass activity, and overpotential)
from atomistic nanoparticle structures.

References
----------
[1] Rossi, K.; Asara, G. G.; Baletto, F.
       "A genomic characterisation of monometallic nanoparticles".
       Phys. Chem. Chem. Phys. 2019, 21, 4888–4898.
       DOI: https://doi.org/10.1039/C8CP05720F
       
       
"""



KB_T = 8.6173303e-5 * 298     # eV
V_REVERSIBLE = 1.23           # V (Theorical Potential ORR)


def get_geometry_properties(coords):
    """
    Compute basic geometrical properties of the nanoparticle.

    Parameters
    ----------
    coords : ndarray of shape (N, 3)
        Cartesian coordinates of the atoms in Angstrom.

    Returns
    -------
    dict
        Dictionary containing:

        - ``natoms`` : int
            Total number of atoms.
        - ``rij_max`` : float
            Maximum interatomic distance in Angstrom.
    """
    
    natoms = len(coords)
    rij_max = np.max(pdist(coords))
    return {"natoms": natoms, "rij_max": rij_max}

def get_physical_and_surface_props(agcns, num_nn, natoms, lattice, mass_pt_mg=None):
    """
    Compute physical and surface-related properties of a Pt nanoparticle.

    Parameters
    ----------
    agcns : ndarray
        Array of generalized coordination numbers (GCN/AGCN)
        for each atom.

    num_nn : ndarray
        Number of nearest neighbours for each atom.

    natoms : int
        Total number of atoms in the nanoparticle.

    lattice : float
        FCC lattice parameter in Angstrom.

    mass_pt_mg : float, optional
        Mass of a single Pt atom in milligrams.
        If not provided, the value is computed from the
        atomic mass of Pt.

    Returns
    -------
    dict
        Dictionary containing:

        - ``mass_np_mg`` : float
            Total nanoparticle mass in milligrams.
        - ``active_surf_m2`` : float
            Electrochemically active surface area in m².
        - ``active_surf_cm2`` : float
            Electrochemically active surface area in cm².
        - ``counter_surf`` : int
            Number of surface atoms.
        - ``surf_mask`` : ndarray of bool
            Boolean mask identifying surface atoms
            (atoms with coordination number < 11).

    Notes
    -----
    Surface atoms are identified using a nearest-neighbour
    criterion (coordination number < 11).
    """
    
    atomic_r_AA = lattice/(2 * np.sqrt(2))
    atomic_r_m=atomic_r_AA* 1e-10
    
    print(atomic_r_m)
    if mass_pt_mg is not None:
        mass_Pt_mg=mass_pt_mg
    else:
        mass_Pt_amu=mass["Pt"]
        mass_Pt_mg = mass_Pt_amu * 1.66053906660e-21
    
    print(mass_Pt_mg)
    mass_np_mg = mass_Pt_mg * natoms
    surf_mask = (num_nn < 11)
    counter_surf = np.sum(surf_mask)
    
    # Area Attiva (m^2)
    active_surf_m2 = np.sum(4 * np.pi * (atomic_r_m**2) * ((12.0 - agcns[surf_mask]) / 12.0))
    
    return {
        "mass_np_mg": mass_np_mg,
        "active_surf_m2": active_surf_m2,
        "active_surf_cm2": active_surf_m2 * 1e4,
        "counter_surf": counter_surf,
        "surf_mask": surf_mask
    }

                    

def calculate_volcano_dg_fortran(gcn_unique):
    """
    Compute adsorption free-energy corrections from the
    volcano relationship as in 

    Parameters
    ----------
    gcn_unique : ndarray
        Array of unique generalized coordination numbers.

    Returns
    -------
    ndarray
        Free-energy descriptor values (eV) associated
        with each GCN value.
    """


    return np.where(gcn_unique < 8.33,
                    0.192 * (gcn_unique) - 0.724,
                    -0.178 * (gcn_unique) + 2.345)


def perform_activity_analysis(unique_gcns, occurrences, phys_props):
    """
    Compute electrochemical activity metrics for a Pt nanoparticle.

    Parameters
    ----------
    unique_gcns : ndarray
        Unique generalized coordination number values.

    occurrences : ndarray
        Number of occurrences associated with each GCN value.

    phys_props : dict
        Dictionary containing physical and surface properties
        returned by ``get_physical_and_surface_props``.

    Returns
    -------
    sa_09 : float
        Specific activity at 0.9 V.

    ma_09 : float
        Mass activity at 0.9 V in mA/mg.

    eta1 : float
        Overpotential corresponding to a total current density
        larger than 1 mA/cm².

    """
    
    u_bins = np.arange(1, 1299) * 0.001 # j*0.001
    total_current = np.zeros_like(u_bins)

    dg_values = calculate_volcano_dg_fortran(unique_gcns)
    
    freq = occurrences / np.sum(occurrences)
    
    for j_idx, u in enumerate(u_bins):

        delta = (dg_values - u) / KB_T
        j_lim = unique_gcns * 12.65 #mA cm^-2
        
        site_currents = np.exp(delta) * freq * j_lim
        total_current[j_idx] = np.sum(site_currents)
    
    # SA a 0.9V (index is 899 because u = 900 * 0.001)
    sa_09 = total_current[899]
    
    # Change the bin if you want change the applied potential SA a 0.88V (index is 879 because u = 880 * 0.001)
    #sa_09 = total_current[879]
    
    ma_09 = sa_09 * (phys_props['active_surf_cm2']) / phys_props['mass_np_mg']
    
    eta1 = 0.0
    for j_idx in range(len(u_bins)-1, -1, -1):
        if total_current[j_idx] > 1.0:
            eta1 = V_REVERSIBLE - u_bins[j_idx]
            break
            
    return sa_09, ma_09, eta1
    
    
