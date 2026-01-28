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

        
def read_order_lammps_dump(filename, style='atomic'):
    """
    Extract a movie ( Tuple[np.ndarray, np.ndarray] ) from a lammps dump file. Atoms are not written in a consistent \n
    order in dump files, so you generally need to reorder them. 
    WARNING: this only works with standard atomic style dumps, where (at least the first) columns in the dump file \n
    are: id type xs ys zs

    Parameters
    ----------
    filename : str
        filename for the lammps-dump file to extract atoms from.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        species and movie (positions) from the lammps dump with consistent ordering of atoms.
    """

    if style != "atomic":
        raise NotImplementedError("Only dump style 'atomic' is currently supported")

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
                curr_ids.append(int(parts[0]) - 1) #lammps has 1-based ids
                curr_frame.append([float(parts[2])*xbox, float(parts[3])*ybox, float(parts[4])*zbox])
                curr_species.append(int(parts[1]))
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