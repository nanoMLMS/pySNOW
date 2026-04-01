import numpy as np
import os
from snow.descriptors.utils import nearest_neighbours
from snow.descriptors.coordination import agcn_calculator
    

def LAE(coords: np.ndarray, elements, cut_off):
    # Trova i vicini più prossimi per ogni atomo
    """
    Calculates the local atomic environment (LAE) of an atom in a structure.
    
    Parameters
    ----------
        Placeholder for the index of the frame in a trajectory type structure, derived from molecular dynamics simulations.
    coords : np.ndarray
        Coordinates of the atoms in the structure.
    elements : list
        List of atomic symbols for each atom in the structure.
    cut_off : float
        Cut-off distance for the nearest neighbors, in Angstroms.
    
    Returns
    -------
    CN_list : list
        List of the number of nearest neighbors for each atom.
    num_atom_same_species : list
        List of the number of atoms of the same species for each atom.
    num_atom_other_species : list
        List of the number of atoms of different species for each atom.
    """
    nearest_neigh = nearest_neighbours( coords, cut_off)

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


