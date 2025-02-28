import numpy as np
import os
from snow.lodispp.utils import nearest_neighbours
from snow.descriptors.gcn import agcn_calculator
    

def LAE_xyz(index_frame: int, coords: np.ndarray, elements, cut_off):
    # Trova i vicini più prossimi per ogni atomo
    nearest_neigh = nearest_neighbours(index_frame, coords, cut_off)

    CN_list = []
    num_atom_same_species = []
    num_atom_other_species = []

    # Loop sugli atomi
    for j in range(len(elements)):
        CN = 0
        same = 0
        other = 0

        #print(elements[j])
        # Itera sui vicini di j
        for k in nearest_neigh[j]:
            CN += 1  # Incrementa il numero totale di vicini
            
            print(elements[j], elements[k])
            if elements[j] == elements[k]:
                same += 1  # Stesso elemento
            else:
                other += 1  # Elemento diverso
        
        print(same, other)
        CN_list.append(CN)
        num_atom_same_species.append(same)
        num_atom_other_species.append(other)

    return CN_list, num_atom_same_species, num_atom_other_species




def LAE_write_xyz_file(filename, elements, coords, lae_output, frame):
    if frame == 0 and os.path.exists(filename):
        os.remove(filename)

    if frame == 0:
        print("el, x, y, z, CN, num_atom_same_species, num_atom_other_species\n")

    with open(filename, "a") as f:
        num_atoms = len(elements)
        f.write(f"{num_atoms}\n\n")

        for i in range(num_atoms):
            x, y, z = coords[i]
            lae1, lae2, lae3= lae_output[0][i], lae_output[1][i], lae_output[2][i]
            f.write(f"{elements[i]} {x:.6f} {y:.6f} {z:.6f} {lae1} {lae2} {lae3}\n")


def LAE_Surf_write_xyz_file(index_frame: int, filename, elements, coords, lae_output, frame, cut_off):
    agcn_array = agcn_calculator(index_frame, coords, cut_off, gcn_max=12.0)

    if frame == 0 and os.path.exists(filename):
        os.remove(filename)

    if frame == 0:
        print("el, x, y, z, num_atom_same_species, num_atom_other_species, surface\n")

    with open(filename, "a") as f:
        num_atoms = len(elements)
        f.write(f"{num_atoms}\n\n")

        for i in range(num_atoms):
            x, y, z = coords[i]
            lae1, lae2 = lae_output[0][i], lae_output[1][i]

            # Identify surface atom
            surface = 1 if agcn_array[i] <= 8.5 else 0

            f.write(f"{elements[i]} {x:.6f} {y:.6f} {z:.6f} {lae1} {lae2} {surface}\n")

    
def LAE_count(index_frame: int, coords: np.ndarray, elements, cut_off, elem_1, elem_2, range_lower_bin, range_upper_bin, savepath):
    nearest_neigh = nearest_neighbours(index_frame, coords, cut_off)
    
    elements_nn = [[elements[el] for el in i] for i in nearest_neigh]
    
    num_atom_same_species = []
    num_atom_other_species = []
    num_NN = []
    
    for j, elem in enumerate(elements_nn):
        same = sum(1 for i in elem if i == elements[j])
        other = len(elem) - same
        num_atom_same_species.append(same)
        num_atom_other_species.append(other)
        num_NN.append(len(elem))
    
    results = {elem_1: [0] * len(range_lower_bin), elem_2: [0] * len(range_lower_bin)}
    
    for r in range(len(range_lower_bin)):
        for k, e in enumerate(elements):
            if range_lower_bin[r] <= num_atom_other_species[k] and num_atom_other_species[k]<= range_upper_bin[r]:
                results[e][r] += 1
    
    filename = f'{savepath}/LAE_count_results_{elem_1}.csv'
    with open(filename, 'a') as f:
    
            
        if index_frame==0:
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
            
        
        for elem in [elem_1, elem_2]:
            f.write(f'{elem},{index_frame}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
    
    
    filename = f'{savepath}/LAE_count_results_{elem_2}.csv'
    with open(filename, 'a') as f:
    
            
        if index_frame==0:
            f.write('Element, frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
            
        
        for elem in [elem_2]:
            f.write(f'{elem},{index_frame}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
            
            
    print(f'Results saved in {filename}')

        


def LAE_surf_count(index_frame: int, coords: np.ndarray, elements, cut_off, elem_1, elem_2, range_lower_bin, range_upper_bin, savepath):
    nearest_neigh = nearest_neighbours(index_frame, coords, cut_off)
    agcn_array = agcn_calculator(index_frame, coords, cut_off, gcn_max = 12.0)
    
    #print(agcn_array, len(agcn_array))
    elements_nn = [[elements[el] for el in i] for i in nearest_neigh]
    
    num_atom_same_species = []
    num_atom_other_species = []
    num_NN = []
    
    for j, elem in enumerate(elements_nn):
        same = sum(1 for i in elem if i == elements[j])
        other = len(elem) - same
        num_atom_same_species.append(same)
        num_atom_other_species.append(other)
        num_NN.append(len(elem))
    
    results = {elem_1: [0] * len(range_lower_bin), elem_2: [0] * len(range_lower_bin)}
    

    for r in range(len(range_lower_bin)):
        n_surf=0
        for k, e in enumerate(elements):
            if agcn_array[k] <= 8.5:  # Consider only surface atoms
                n_surf=n_surf+1
                
                #print('array: ', agcn_array[k] )
                if range_lower_bin[r] <= num_atom_other_species[k] <= range_upper_bin[r]:
                    results[e][r] += 1
                    
        #print('n_surf ', n_surf)
            
    filename = f'{savepath}/LAE_count_results_surface_{elem_1}.csv'
    with open(filename, 'a') as f:
        if index_frame==0:
                
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
        
        for elem in [elem_1]:
            f.write(f'{elem},{index_frame}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
    
    
    
    filename = f'{savepath}/LAE_count_results_surface_{elem_2}.csv'
    with open(filename, 'a') as f:
        if index_frame==0:
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
        
        for elem in [elem_2]:
            f.write(f'{elem},{index_frame+1}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
            
            
    print(f'Surface results saved in {filename}')



def LAE_inner_count(index_frame: int, coords: np.ndarray, elements, cut_off, elem_1, elem_2, range_lower_bin, range_upper_bin, savepath):
    nearest_neigh = nearest_neighbours(index_frame, coords, cut_off)
    agcn_array = agcn_calculator(index_frame, coords, cut_off, gcn_max = 12.0)
    
    #print(agcn_array, len(agcn_array))
    elements_nn = [[elements[el] for el in i] for i in nearest_neigh]
    
    num_atom_same_species = []
    num_atom_other_species = []
    num_NN = []
    
    for j, elem in enumerate(elements_nn):
        same = sum(1 for i in elem if i == elements[j])
        other = len(elem) - same
        num_atom_same_species.append(same)
        num_atom_other_species.append(other)
        num_NN.append(len(elem))
    
    results = {elem_1: [0] * len(range_lower_bin), elem_2: [0] * len(range_lower_bin)}
    

    for r in range(len(range_lower_bin)):
        n_inner=0
        for k, e in enumerate(elements):
            if agcn_array[k] > 8.5:  # Consider only surface atoms
                n_inner=n_inner+1
                
                #print('array: ', agcn_array[k] )
                if range_lower_bin[r] <= num_atom_other_species[k] <= range_upper_bin[r]:
                    results[e][r] += 1
                    
        #print('n_inner ', n_inner)
            
    filename = f'{savepath}/LAE_count_results_inner_{elem_1}.csv'
    with open(filename, 'a') as f:
        if index_frame==0:
                
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
        
        for elem in [elem_1]:
            f.write(f'{elem},{index_frame}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
    
    
    
    filename = f'{savepath}/LAE_count_results_inner_{elem_2}.csv'
    with open(filename, 'a') as f:
        if index_frame==0:
            f.write('Element,frame')
            for r in range(len(range_lower_bin)):
                f.write(f',[{range_lower_bin[r]},{range_upper_bin[r]}]')
            f.write('\n')
        
        for elem in [elem_2]:
            f.write(f'{elem},{index_frame+1}')
            for count in results[elem]:
                f.write(f',{count}')
            f.write('\n')
            
            
    #print(f'Inner results saved in {filename}')
