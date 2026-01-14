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



def get_ids_lammpsdump(file):
    """
    Get atomic ids from a lammps dumpfile. This is useful to track down individual atoms in the trajectories,
    as the order in which these are written by lammps is not constant throughout a simulation. 

    Parameters
    ----------
    file : str
        (name of the) text file that contains the trajectory to be opened.

    Returns
    -------
    Tuple[np.ndarray]
        atomic ids for the provided trajectory 
    """

    try:
        with open(file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error opening file: {e}")
        return []
    
    reading_atoms=False
    ids = []

    for line in lines:
        if line.startswith("ITEM: ATOMS"):
            reading_atoms = True
            tmp_id_list = []
        elif line.startswith("ITEM:") and reading_atoms: #this frame is over
            reading_atoms = False
            ids.append( np.array(tmp_id_list) )
        elif reading_atoms:   #read data
            parts = line.strip().split()
            if parts:
                try:
                    tmp_id_list.append(int(parts[0]))
                except ValueError:
                    print(f"Could not convert the value to int: {parts[0]}")
    
    ids.append( np.array(tmp_id_list) )#append last list
    return ids