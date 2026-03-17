from typing import Tuple
import numpy as np
import os
import inspect

def read_rgl(filepot: str):
    with open(filepot, 'r') as file:
        lines = file.readlines()
    
    # Preprocess lines to replace 'd' with 'e' for compatibility with Python floats
    lines = [line.replace('d', 'e').replace('D', 'E') for line in lines]
    
    #get the two elements from the file
    element_a, element_b = lines[2].split()
    
    #potential parameters
    p_line = lines[5].split()
    p_a = float(p_line[0])
    p_b = float(p_line[1])
    p_ab = float(p_line[2])
    
    q_line = lines[6].split()
    q_a = float(q_line[0])
    q_b = float(q_line[1])
    q_ab = float(q_line[2])
    
    a_line = lines[7].split()
    a_a = float(a_line[0])
    a_b = float(a_line[1])
    a_ab = float(a_line[2])
        
    qsi_line = lines[8].split()
    qsi_a = float(qsi_line[0])
    qsi_b = float(qsi_line[1])
    qsi_ab = float(qsi_line[2])
    
    #metal properties
    coh_energies = lines[11].split()[0:2]
    coh_energy_a = float(coh_energies[0])
    coh_energy_b = float(coh_energies[1])
    
    
    atom_radiuses = lines[12].split()[0:2]
    atom_radius_a = float(atom_radiuses[0])
    atom_radius_b = float(atom_radiuses[1])
    
    masses = lines[13].split()[0:2]
    mass_a = float(masses[0])
    mass_b = float(masses[1])
    
    #cut off 
    cutoffs = lines[16].split()[0:2]
    cut_start = float(cutoffs[0])
    cut_end = float(cutoffs[1])   
        
    #bulk nearest enighbour distances
    dist = np.array([atom_radius_a * np.sqrt(8.0), 
                      atom_radius_b * np.sqrt(8.0) ,
                      (atom_radius_a + atom_radius_b) * np.sqrt(2.0)])
    #organizing params in array for iteration 
    a = np.array([a_a, a_b, a_ab])
    p = np.array([p_a, p_b, p_ab])
    q = np.array([q_a, q_b, q_ab])
    qsi = np.array([qsi_a, qsi_b, qsi_ab])
    
    x3 = np.zeros(3)
    x4 = np.zeros(3)
    x5 = np.zeros(3)
    
    a3 = np.zeros(3)
    a4 = np.zeros(3)
    a5 = np.zeros(3)
    for i in range(3):
        d_ik_0 = dist[i]
        
        ar = -a[i] * np.exp(-p[i]* (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start ) ** 3
        br = -(p[i] / d_ik_0) * a[i] * np.exp(-p[i] * (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start) ** 2
        cr = -((p[i] / d_ik_0) ** 2) * a[i] * np.exp(-p[i] * (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start)

        ab = -qsi[i] * np.exp( -q[i] * (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start) ** 3
        bb= -(q[i] / d_ik_0) * qsi[i] * np.exp( -q[i] * (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start) ** 2
        cb= -((q[i] / d_ik_0) ** 2) * qsi[i] * np.exp( -q[i]*(cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start)
        
        x5[i] = (12.0 * ab - 6.0 * bb + cb) / (2.0 * (cut_end - cut_start) ** 2) 
        x4[i] = (15.0 * ab - 7.0 * bb + cb) / (cut_end - cut_start)
        x3[i] = (20.0 * ab - 8.0 * bb + cb) / 2.0
        
        a5[i] = (12.0 *ar - 6.0 * br + cr) / (2.0 * (cut_end - cut_start) ** 2)
        a4[i] = (15.0 *ar - 7.0 * br + cr) / (cut_end - cut_start)
        a3[i] = (20.0 *ar - 8.0 * br + cr) / 2.0
        
    return {
        "el_a": element_a,
        "el_b": element_b,
        
        "p": p,
        
        "q": q, 
        
        "a": a,
        
        "qsi": qsi,
        
        "coh_a": coh_energy_a,
        "coh_b": coh_energy_b,
        
        "r_a": atom_radius_a,
        "r_b": atom_radius_b,
        
        "m_a": mass_a,
        "m_b": mass_b,
        
        "cut_start": cut_start,
        "cut_end": cut_end,
        
        "dist": dist,
        
        "x3": x3,
        "x4": x4,
        "x5": x5,
        
        "a3": a3,
        "a4": a4,
        "a5": a5,
        
        
    }
       
def read_xyz(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads the elements and coordinates of atoms from an xyz file at a given location

    Parameters
    ----------
    file_path : str
        Path to the xyz file with the structure

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Elements and coordinates array of the system

    Raises
    ------
    FileNotFoundError
        No file was found at the given location
    ValueError
        Some error while reading the file
    Exception
        Unexpected errors along the way
    """
    try:
        # Get the path of the calling script
        caller_frame = inspect.stack()[1]  # Get the caller's frame
        caller_script = caller_frame.filename  # Get the caller's script path
        
        # Get the directory where the calling script is located
        script_dir = os.path.dirname(os.path.realpath(caller_script))
        
        # Construct the full path to the file
        filepath = os.path.join(script_dir, file_path)

        # Open the file
        with open(filepath, "r") as xyz_file:
            # Number of atoms
            n_atoms = int(xyz_file.readline().strip())

            # Skip the comment line
            _ = xyz_file.readline().strip()

            # Initialize containers
            elements = []
            coordinates = np.zeros((n_atoms, 3))

            # Read the data
            for i in range(n_atoms):
                line = xyz_file.readline().split()
                elements.append(line[0])  # Append element symbol
                coordinates[i, :] = list(map(float, line[1:4]))  # Convert coordinates to float

        return elements, coordinates

    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except ValueError as e:
        raise ValueError(f"Error reading '{file_path}': {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")





def read_xyz_movie(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Obtains the coordinates and elements for each frame of an xyz trajectory (for now it only supports trajectories
    where the number of atoms and chemical composition is fixed through the whole trajectory).

    Note that it only creates a singe array for the elements rather than a per-frame array.

    Parameters
    ----------
    file_path : str
        Path to the xyz file with the structure


    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Elements array and a (n_frames x 3 x n_atoms) array for the coordinates
    """
    
    
    with open(file_path, "r") as xyz_file:
        # Read the total number of atoms from the first line
        n_atoms = int(xyz_file.readline().strip())
    
    # Calculate the number of frames in the file
    num_lines = sum(1 for _ in open(file_path))
    n_frames = num_lines // (n_atoms + 2)

    # Initialize arrays
    coords = np.zeros((n_frames, n_atoms, 3))
    elements = []

    # Parse the file to extract data
    with open(file_path, "r") as xyz_file:
        for frame in range(n_frames):
            _ = xyz_file.readline().strip()  # Skip atom count line
            _ = xyz_file.readline().strip()  # Skip comment line
            
            for atom in range(n_atoms):
                line = xyz_file.readline().split()
                if frame == 0:  # Store elements only once
                    elements.append(line[0])
                coords[frame, atom, :] = list(map(float, line[1:4]))

    return elements, coords
        
            
        



def read_lammps(file_path):

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

def write_xyz(filename, elements, coords, additional_data=None):
    """
    Writes atomic data to an XYZ file in OVITO-compatible format.

    Parameters
    ----------
    filename: str
        Name of the output .xyz file.
    elements: ndarray
        List of atomic symbols (e.g., ['Au', 'Au', ...]).
    coords: ndarray)
        Nx3 array of atomic coordinates.
    additional_data: list or np.ndarray, optional
        Additional per-atom data, such as coordination numbers.

    Returns:
        An xyz file containing the elements and coordinates of each atom and any additional per atom data (e.g. coordination number, agcn, strain...) 
    """
    n_atoms = len(elements)
    
    # Check if additional_data is provided and has the correct shape
    if additional_data is not None:
        additional_data = np.array(additional_data)
        if additional_data.shape[0] != n_atoms:
            raise ValueError(f"The number of rows in additional_data ({additional_data.shape[0]}) must match the number of atoms ({n_atoms}).")
    
    with open(filename, 'w') as xyz_file:
        # Write header
        xyz_file.write(f"{n_atoms}\n")
        xyz_file.write("Generated XYZ file with optional properties\n")
        
        # Write atom data
        for i in range(n_atoms):
            atom_line = f"{elements[i]} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}"
            if additional_data is not None:
                
                # Add the additional per-atom data
                atom_line += ' ' + ' '.join([f"{additional_data[i, j]:.6f}" for j in range(additional_data.shape[1])])
            xyz_file.write(atom_line + "\n")


def write_movie_xyz(frame, filename, elements, coords, additional_data=None):
    """
    Writes atomic data to an XYZ file in OVITO-compatible format.

    Parameters
    ----------
    frame: int
        Frame number.
    filename: str
        Name of the output .xyz file.
    elements: ndarray
        List of atomic symbols (e.g., ['Au', 'Au', ...]).
    coords: ndarray)
        Nx3 array of atomic coordinates.
    additional_data: list or np.ndarray, optional
        Additional per-atom data, such as coordination numbers.

    Returns:
        An xyz file containing the elements and coordinates of each atom and any additional per atom data (e.g. coordination number, agcn, strain...)
    """

    if frame==0 and os.path.exists(filename):
            os.remove(filename)

    n_atoms = len(coords)

    # Check if additional_data is provided and has the correct shape
    if additional_data is not None:
        additional_data = np.array(additional_data)
        if additional_data.shape[0] != n_atoms:
            raise ValueError(
                f"The number of rows in additional_data ({additional_data.shape[0]}) must match the number of atoms ({n_atoms}).")

    with open(filename, 'a') as xyz_file:
        # Write header
        xyz_file.write(f"{n_atoms}\n\n")
        #xyz_file.write(f"\n{frame}\n")
        #xyz_file.write("Generated XYZ file with optional properties\n")

        # Write atom data
        for i in range(n_atoms):
            atom_line = f"{elements[i]} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}"
            if additional_data is not None:
                # Add the additional per-atom data
                atom_line += ' ' + ' '.join([f"{additional_data[i, j]:.6f}" for j in range(additional_data.shape[1])])
            xyz_file.write(atom_line + "\n")
