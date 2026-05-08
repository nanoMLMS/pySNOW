import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist
from snow.io.xyz import read_xyz
from snow.descriptors.coordination import agcn_calculator, nearest_neighbours
from snow.catalysis.Pt_ORR import *
from math import sqrt



lattice= 3.9173715677734737 #in AA
cutoff = 0.85 * lattice

file_xyz = 'traj.xyz'
el, coords = read_xyz(file_xyz)

sites, agcns = agcn_calculator(coords, cutoff, thr_gcn=11)
neigh_list = nearest_neighbours(coords=coords, cut_off=cutoff)
num_nn = np.array([len(n) for n in neigh_list])

geo = get_geometry_properties(coords)
phys = get_physical_and_surface_props(agcns, num_nn, geo['natoms'], lattice)

counts = Counter(np.round(agcns[phys['surf_mask']], 3)) #rounded
unique_gcns = np.array(list(counts.keys()))
occurrences = np.array(list(counts.values()))

print(unique_gcns)

sa_09, ma_09, eta_val = perform_activity_analysis(unique_gcns, occurrences, phys)


print(sa_09, ma_09, eta_val)



# ============================================================
# Saving data
# ============================================================

import numpy as np
from collections import Counter


# Saving in eta.dat
with open('eta.dat', 'w') as f:
    f.write(
        "# natoms \t diameter [nm] \t gcn area [m2] \t "
        "gcn area [Ang 2] \t mass NP [mg] \t eta1 [V] \t "
        "SA@0.9V (mA/cm2) \t MA@0.9V (mA/mg) (gcn area based)\n"
    )

    # Conversions area
    area_m2 = phys['active_surf_cm2'] / 1e4
    area_ang2 = phys['active_surf_cm2'] * 1e16

    line = (
        f"{geo['natoms']}\t"
        f"{geo['rij_max']/10.0:.16f}\t"
        f"{area_m2:.18e}\t"
        f"{area_ang2:.16f}\t"
        f"{phys['mass_np_mg']:.6e}\t"
        f"{eta_val:.16f}\t"
        f"{sa_09:.16f}\t"
        f"{ma_09:.16f}\n"
    )

    f.write(line)


# Saving in gcn_pure.dat
sorted_pure_gcns = sorted(counts.items())

with open('gcn_pure.dat', 'w') as f:
    f.write("# GCN_value (pure)\tOccurrence\n")

    for gcn_val, count in sorted_pure_gcns:
        f.write(f"{gcn_val:.3f}\t{count}\n")
