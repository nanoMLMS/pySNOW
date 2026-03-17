from snow.io.xyz import read_xyz

from snow.descriptors.layer_by_layer import *


path='/Users/sofiazinzani/Documents/Dottorato/Unimi/Codici/pySNOW/tutorial/tutorial_structures/frame_200ns.xyz'

au_lattice=4.08 #AA
el, coords=read_xyz(path)
print(len(el))


lbl_info=cut_layers_from_frame(el, coords, au_lattice, 'Au', 'Pd', 'z')
print(lbl_info)




#########
from snow.io.xyz import write_xyz_movie
from snow.transform.rototranslation import align_to_axis, center_com

path='/Users/sofiazinzani/Documents/Dottorato/Unimi/Codici/pySNOW/tutorial/tutorial_structures/frame_200ns.xyz'

cu_lattice=3.615 #AA
el, coords=read_xyz(path)
print(len(el))

new_coords=align_to_axis(1, coords, (1,1,1))
write_xyz_movie(1, '/Users/sofiazinzani/Documents/Dottorato/Unimi/Codici/pySNOW/tutorial/tutorial_structures/frame_200ns_rotated.xyz', el, new_coords)


com=center_com(2, coords, el)
print(com)
