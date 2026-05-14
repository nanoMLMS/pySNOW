import numpy as np
from snow.descriptors.utils import nearest_neighbours    

def LAE(elements: list[str], coords: np.ndarray, cut_off):
    """
    Calculates the local atomic environment (LAE) of an atom in a structure.

    Here, the lae is characterized as the number of homo-bonds (with atoms of the same species)
    and the number of hetero bonds (with atoms of other chemical species). The function
    computes neighbours, checks the bonds types, and returns a count of homo and hetero bonds.
    
    Parameters
    ----------
    elements : list[str]
        list of chemical symbols of atoms in your system
    coords : np.ndarray
        Coordinates of the atoms in the structure.
    cut_off : float
        Cut-off distance for the nearest neighbors, in Angstroms.
    
    Returns
    -------
    CN_list : np.ndarray
        List of the number of nearest neighbors (coordination number) for each atom.
    num_atom_same_species : np.ndarray
        List of the number of atoms of the same species (homo bonds) for each atom.
    num_atom_other_species : np.ndarray
        List of the number of atoms of different species (hetero bonds) for each atom.
    """

    nearest_neigh = nearest_neighbours(coords, cut_off)

    CN_list = []
    num_atom_same_species = []
    num_atom_other_species = []

    # Loop over atoms
    for j in range(len(elements)):
        CN = 0
        same = 0
        other = 0

        # Iterate on neighbours of j
        for k in nearest_neigh[j]:
            CN += 1  # increase neighbours counter
            
            #print(elements[j], elements[k])
            #check the bond type
            if elements[j] == elements[k]:
                same += 1  # homo
            else:
                other += 1 # hetero
        
        #print(same, other)
        CN_list.append(CN)
        num_atom_same_species.append(same)
        num_atom_other_species.append(other)

    return np.asarray(CN_list), np.asarray(num_atom_same_species), np.asarray(num_atom_other_species)


