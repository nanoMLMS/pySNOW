import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist
from snow.io.xyz import read_xyz
from snow.descriptors.coordination import agcn_calculator, nearest_neighbours
from snow.misc.constants import *


KB_T = 8.6173303e-5 * 298     # eV
V_REVERSIBLE = 1.23           # V (Theorical Potential ORR)


def get_geometry_properties(coords):
    natoms = len(coords)
    rij_max = np.max(pdist(coords))
    return {"natoms": natoms, "rij_max": rij_max}

def get_physical_and_surface_props(agcns, num_nn, natoms, lattice, mass_pt_mg=None):

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
    return np.where(gcn_unique < 8.33,
                    0.192 * (gcn_unique) - 0.724,
                    -0.178 * (gcn_unique) + 2.345)

def perform_activity_analysis(unique_gcns, occurrences, phys_props):
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
    
    
