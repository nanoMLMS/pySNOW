from snow.io.xyz import read_xyz
from snow.descriptors.layer_by_layer import *


path='/Users/sofiazinzani/Documents/Dottorato/Unimi/Codici/pySNOW/tutorial/tutorial_structures/Cu309_Ih.xyz'

cu_lattice=3.615 #AA
el, coords=read_xyz(path)
print(len(el))


lbl_info=cut_layers_from_frame(el, coords, cu_lattice, 'Cu', 'Cu')
print(lbl_info)
