#compute locally normal directions for top, bridge, three- and four-hollow sites
#for a user-provided xyz file
#and write them as fictitious forces in a new xyz file.
#the sites positions are represented by fictitious Hydrogen atoms.
from snow.io.xyz import read_xyz, write_xyz
from snow.descriptors.coordination import *
from snow.build.add_molecule import *

import sys

if len(sys.argv)<3:
    sys.exit(f'usage: {sys.argv[0]} <file.xyz> <cutoff_radius>')

file_in  = sys.argv[1]
file_out = file_in.strip('.xyz') + '_directions.xyz'
cutoff   = float(sys.argv[2]) 

el, coords = read_xyz(file_in)

vectors = []
thr_cn = 10


cns = coordination_number(coords, cutoff)
print('computed cn')
b_sites, pairs, bgcns = bridge_gcn(coords, cutoff, thr_cn)
print('\ncomputed bridge gcn')
t_sites, triplets, tgcns = three_hollow_gcn(coords, cutoff, thr_cn)
print('\ncomputed three-hollow gcn')
f_sites, fourplets, fgcns = four_hollow_gcn(coords, cutoff, thr_cn)
print('\ncomputed four-hollow gcn')


#atop
for coord, cn in zip(coords, cns):
    if cn<thr_cn:
        vec = locally_normal_direction(coords, coord, cutoff)
    else:
        vec = np.asarray([0.,0.,0.])
    vectors.append(vec)


write_xyz(file_out, el, coords, additional_data=np.asarray(vectors), mode='w' )

#bridge
vectors = []
for site in b_sites:
    vec = locally_normal_direction(coords, site, cutoff)
    vectors.append(vec)

vectors = np.asarray(vectors)
zeroes  = np.zeros( (len(el), 3))

bridge_el = el + ['H']*len(b_sites)
bridge_coords = np.vstack((coords, b_sites))
vectors = np.vstack( ( zeroes, vectors ) )
write_xyz(file_out, bridge_el, bridge_coords, additional_data=vectors, mode='a' )

#3hollow
vectors = []
for site, triplet in zip(t_sites, triplets):
    vec = triplet_normal(coords, triplet)
    vectors.append(vec)

vectors = np.asarray(vectors)
zeroes  = np.zeros( (len(el), 3))

th_el = el + ['H']*len(t_sites)
th_coords = np.vstack((coords, t_sites))
vectors = np.vstack( ( zeroes, vectors ) )
write_xyz(file_out, th_el, th_coords, additional_data=vectors, mode='a' )

#4hollow

vectors = []
for site, fourplet in zip(f_sites, fourplets):
    vec = fourplet_normal(coords, fourplet)
    vectors.append(vec)

vectors = np.asarray(vectors)
zeroes  = np.zeros( (len(el), 3))

fh_el = el + ['H']*len(f_sites)
fh_coords = np.vstack((coords, f_sites))
vectors = np.vstack( ( zeroes, vectors ) )

write_xyz(file_out, fh_el, fh_coords, additional_data=vectors, mode='a' )
