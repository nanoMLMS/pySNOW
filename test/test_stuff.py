from snow.lodispp.utils import *
import matplotlib.pyplot as plt
from snow.lodispp.pp_io import *
from snow.strain.strain_mono import *
from snow.descriptors.steinhardt import *

import numpy as np




# Read XYZ file
elements, coords = read_xyz("sus.xyz")
com = center_of_mass(1, coords=coords, elements=elements)

# Define parameters
cutoff_radius = 4.079 * 0.95  # Cutoff radius for neighbor detection
dist_0 = 1.66 * 2       # Reference distance for strain calculation

# Compute coordination number, strain, and other properties
cn = coordination_number(1, coords=coords, cut_off=cutoff_radius)
strain_syst = strain_mono(index_frame=1, coords=coords, dist_0=dist_0, cut_off=cutoff_radius)
fnn = nearest_neighbours(1, coords, cutoff_radius)
snn = second_neighbours(1, coords, cutoff_radius)
agcn = agcn_calculator(index_frame=1, coords=coords, cutoff=cutoff_radius, gcn_max=12)

# Determine surface atoms
is_surface = agcn < 10.0

# Compute Steinhardt parameters
stein = peratom_steinhardt(1, coords, [4, 6, 8, 12], cutoff_radius)

# Prepare additional data
additional_data = np.column_stack((cn, agcn, strain_syst, is_surface, *stein))
print("q12_avg = {:.3f} +/- {:.3f}".format(np.mean(stein[3]), np.std(stein[3])))

# Write updated XYZ file
write_xyz("output_test.xyz", elements=elements, coords=coords, additional_data=additional_data)

# Phantom atoms and bridge GCN
phant_xyz, pairs, b_gcn = bridge_gcn(1, coords=coords, cut_off=cutoff_radius, gcn_max=18, phantom=True)

# Add phantom elements
for p in pairs:
    elements.append("H")

# Combine AGCN and phantom GCN
a_b_gcn = np.concatenate((agcn, b_gcn))
a_b_gcn = a_b_gcn.reshape(-1, 1)  # Shape will be (843, 1)

rnd = np.random.uniform(len(a_b_gcn))
# Concatenate coordinates with phantom coordinates
out_xyz = np.concatenate((coords, phant_xyz))    

# Write updated XYZ file with phantom data
write_xyz("phantom.xyz", elements, out_xyz, additional_data=a_b_gcn)


print("Center of mass \t ({} \t {} \t {})".format(*com))


# Plot histogram of phantom GCN values
plt.hist(b_gcn, bins=20, label="Bridge GCN")
plt.xlabel("GCN")
plt.ylabel("Frequency")
plt.title("Distribution of Bridge GCN Values")
plt.legend()
plt.show()
