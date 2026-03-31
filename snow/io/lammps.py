from typing import Tuple
import numpy as np
import os
import inspect


def read_lammps_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read structure from a LAMMPS data file at a certain path

    Parameters
    ----------
    file_path : str
        Path to the lammps data file

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Elements and coordinates array of the system
    """
    coordinates = []
    elements = []
    atoms_section = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect "Atoms" section
            if line.startswith("Atoms"):
                atoms_section = True
                continue
            
            # Process lines within the Atoms section
            if atoms_section:
                if not line or line.startswith("#"):
                    continue  # Skip empty or comment lines
                
                # Split line and check the format
                parts = line.split()
                if len(parts) < 5:
                    continue  # Skip invalid lines

                try:
                    atom_type = int(parts[1])
                    x, y, z = map(float, parts[2:5])
                except ValueError:
                    print(f"Skipping invalid line: {line}")
                    continue
                
                coordinates.append([x, y, z])
                elements.append(atom_type)
            
            # End of Atoms section
            if atoms_section and not line:
                break

    # Convert to numpy array for easier manipulation
    coordinates = np.array(coordinates)
    
    return elements, coordinates

        
def read_order_lammps_dump(filename, id_index: int = 0, type_index: int = 1, coords_indexes: list = [2,3,4], scaled_coords=True):
    """
    Extract a movie ( Tuple[np.ndarray, np.ndarray] ) from a lammps dump file. Atoms are not written in a consistent \n
    order in dump files, so you generally need to reorder them. 
    You can choose the columns to get the information about id, type, and coords from the lammps dump file. Default is 
    to 'atomic' style, which has the shape 'id type xs ys zs'.

    Parameters
    ----------
    filename : str
        filename for the lammps-dump file to extract atoms from.
    id_index: int
        index of the column that contains ids of your atoms in the lammps dump -  default to 0
    type_index: int
        index of the column that contains the type of atoms in the dump (in lammps these are mapped to numbers)
    coords_indexes: list of ints
        list of indexes of the columns that contain the positions of your atoms - default to [1,2,3]
    scaled_coords: bool
        bool to check if coordinates are scaled (written in terms of the box sizes length). Default to True, which is 
        lammps' default. Probably this can be dealt with automatically by checking if all positions are between 0 and 1,
        but not super general and robust

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        species ids and positions from the lammps dump with consistent ordering of atoms. Here, pos[i] is a Nx3 array with positions of the i-th frame and so on.
    """

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error opening file: {e}")
        return []
    
    reading_atoms = False
    movie         = []
    species       = []

    for i, line in enumerate(lines):

        if line.startswith("ITEM: BOX BOUNDS"):
        #read box size - atoms positions are written in units of box size

            parts_1 = lines[i+1].split()
            parts_2 = lines[i+2].split()
            parts_3 = lines[i+3].split()

            xbox = float(parts_1[1]) - float(parts_1[0])
            ybox = float(parts_2[1]) - float(parts_2[0])
            zbox = float(parts_3[1]) - float(parts_3[0])

            continue

        elif line.startswith("ITEM: ATOMS"):
        #atoms coordinates start in the next line

            reading_atoms = True
            curr_ids = []
            curr_species = []
            curr_frame = []

            continue

        elif line.startswith("ITEM:") and reading_atoms: 
        #this frame is over
        
            reading_atoms = False
            #reorder and save atoms
            curr_frame = np.array(curr_frame)
            curr_species = np.array(curr_species)
            order = np.argsort(curr_ids)
            curr_frame = curr_frame[order]
            curr_species = curr_species[order]

            movie.append(curr_frame)
            species.append(curr_species)

            continue

        elif reading_atoms:
        #read atomic coordinates

            parts = line.split()
            try:
                curr_ids.append(int(parts[id_index]) - 1) #lammps has 1-based ids
                if scaled_coords:
                    curr_frame.append([float(parts[coords_indexes[0]])*xbox, float(parts[coords_indexes[1]])*ybox, float(parts[coords_indexes[2]])*zbox])
                else:
                    curr_frame.append([float(parts[coords_indexes[0]]), float(parts[coords_indexes[1]]), float(parts[coords_indexes[2]])])
                curr_species.append(int(parts[type_index]))
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Malformed atom line at line {i}: {line.strip()}"
                ) from e
    
    #save last frame
    if reading_atoms:
        curr_frame = np.array(curr_frame)
        curr_species = np.array(curr_species)

        order = np.argsort(curr_ids)
        curr_frame = curr_frame[order]
        curr_species = curr_species[order]

        movie.append(curr_frame)
        species.append(curr_species)
    

    return species, movie    
