
from typing import Tuple
import numpy as np
import os
import inspect


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
        
    num_lines = sum(1 for _ in open(file_path))
    n_frames = num_lines // (n_atoms + 2)

    coords = np.zeros((n_frames, n_atoms, 3))
    elements = []

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


def write_xyz_movie(frame, filename, elements, coords, additional_data=None):
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


def write_phantom_xyz(filename, coords, additional_data=None):
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
    n_atoms = len(coords)
    elements=['X'] * n_atoms
    
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





def write_xyz_movie_with_str(frame, filename, elements, coords, additional_data=None):
    """
    Writes atomic data to an XYZ file in OVITO-compatible format.

    Parameters
    ----------
    frame: int
        Frame number.
    filename: str
        Name of the output .xyz file.
    elements: list or ndarray
        List of atomic symbols (e.g., ['Au', 'Au', ...]).
    coords: ndarray
        Nx3 array of atomic coordinates.
    additional_data: list or np.ndarray, optional
        Additional per-atom data, such as coordination numbers, site types, etc.
        Can contain both numeric and string data.

    Returns
    -------
    None
    """

    if frame == 0 and os.path.exists(filename):
        os.remove(filename)

    n_atoms = len(coords)

    # Convert additional_data to array of objects if provided
    if additional_data is not None:
        additional_data = np.array(additional_data, dtype=object)
        if additional_data.shape[0] != n_atoms:
            raise ValueError(
                f"The number of rows in additional_data ({additional_data.shape[0]}) "
                f"must match the number of atoms ({n_atoms})."
            )

    with open(filename, 'a') as xyz_file:
        # Write header
        xyz_file.write(f"{n_atoms}\n")
        xyz_file.write(f"Frame {frame}\n")

        # Write atom data
        for i in range(n_atoms):
            atom_line = f"{elements[i]} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}"
            if additional_data is not None:
                for j in range(additional_data.shape[1]):
                    val = additional_data[i, j]
                    # Numeric values formatted with 6 decimals, strings as-is
                    if isinstance(val, (int, float, np.number)):
                        atom_line += f" {val:.6f}"
                    else:
                        atom_line += f" {val}"
            xyz_file.write(atom_line + "\n")
