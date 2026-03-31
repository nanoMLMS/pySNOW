from typing import Tuple
import numpy as np
import os
import inspect

def read_xyz_movie(file_path: str, extra_cols_indexes: list = None) -> Tuple[list, np.ndarray]:
    """
    Obtains the coordinates and elements for each frame of an xyz trajectory.

    Parameters
    ----------
    file_path : str
        Path to the xyz file with the structure
    extra_frames_indexes : str
        index for the extra columns of per-atom data to be extracted from the .xyz file. Consider that the first three 'indexes'
        are element and three cartesian coordinates and are returned by deafult from the function.
        Example: if your .xyz file has per-atom information like " El pos1 pos2 pos3 force1 force2 force3 charge ",
        you can get the extra columns force1, force2, charge by passing extra_cols_indexes=[4, 5, 7].
        For now only float values parsing is supported.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        list of lists of chemical symbols and a list of (n_atoms, 3) arrays for the coordinates
    """
    
    el_list = []
    coords_list = []

    if extra_cols_indexes is not None:
        n_extra_cols = len(extra_cols_indexes)
        extra_cols_list = []

    with open(file_path, "r") as file:

        conf_line_iter = 0

        for line in file:
            
            conf_line_iter +=1

            #beginning frame
            if conf_line_iter == 1:
                n_atoms = int(line.strip())

                elements   = []
                coords     = np.zeros((n_atoms, 3))
                if extra_cols_indexes is not None:
                    extra_cols = np.zeros((n_atoms, n_extra_cols))

            elif conf_line_iter == 2:
                #skip the comment/general information line
                pass
                #continue 

            elif conf_line_iter >2 and conf_line_iter <= n_atoms+2: 
                #read elements and positions
                parts = line.strip().split()

                if len(parts)<4:
                    print(f'warning this line should have at least 4 values in it but has {len(parts)}')
                
                elements.append(parts[0])

                atom_index = int(conf_line_iter-3)

                coords[ atom_index, 0] = float(parts[1])
                coords[ atom_index, 1] = float(parts[2])
                coords[ atom_index, 2] = float(parts[3])

                if extra_cols_indexes is not None:
                    for i, index in enumerate(extra_cols_indexes):
                        extra_cols[atom_index, i] = float(parts[index])

                #frame is over
                if conf_line_iter == n_atoms+2:
                    el_list.append(elements)
                    coords_list.append(coords)
                    conf_line_iter = 0
                    if extra_cols_indexes is not None:
                        extra_cols_list.append(extra_cols)

    if extra_cols_indexes is not None:
        return el_list, coords_list, extra_cols_list

    return el_list, coords_list

def read_xyz(file, extra_cols_indexes=None):
    """
    wrapper of read_xyz_movie to read single-frame movies - mostyl for compatibility
    """

    if extra_cols_indexes is not None:
        el, coords, extra_cols = read_xyz_movie(file, extra_cols_indexes)
        return el[0], coords[0], extra_cols[0]
    
    else:
        el, coords = read_xyz_movie(file)
        return el[0], coords[0]


def write_xyz(filename, elements, coords, additional_data=None, box=None, mode='w'):
    """
    Writes atomic data to an XYZ file in OVITO-compatible format. Currently only accepting numbers
    as additional data.

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
    box: np.ndarray
        a box to be written to file
    mode: str
        mode for writing ('a'->append,  'w'->(over)write)

    Returns:
        None
    """

    n_atoms = len(elements)

    #some controls to cast data in the right shape (convert to shape==(n_atoms, 1) if possible)
    if type(additional_data) == np.ndarray and additional_data.shape == (n_atoms, ):
        additional_data = additional_data[:,None]
    elif type(additional_data) == list:
        additional_data = np.array(additional_data)[:,None]
    elif type(additional_data) == np.ndarray or additional_data is None:
        pass
    else:
        raise ValueError('Please provide additional data as either a list or a np.ndarray')
    
    # Check if additional_data is provided and has the correct shape
    if additional_data is not None:
        additional_data = np.array(additional_data)
        if additional_data.shape[0] != n_atoms:
            raise ValueError(f"The number of rows in additional_data ({additional_data.shape[0]}) must match the number of atoms ({n_atoms}).")
    
    with open(filename, mode) as xyz_file:
        # Write header
        xyz_file.write(f"{n_atoms}\n")

        #write general info line
        if box is not None:
            xyz_file.write('Lattice="')
            #suppose box is shape=(3,3)
            if box.shape == (3,3):
                for i in range(3):
                    for j in range(3):
                        xyz_file.write(f'{box[i,j]} ')
            elif box.shape == (3,1):
                xyz_file.write(f'{box[0,0]} 0.0 0.0 ')
                xyz_file.write(f'0.0 {box[1,0]} 0.0 ')
                xyz_file.write(f'0.0 0.0 {box[2,0]}')
            elif box.shape == (3,2):
                xyz_file.write(f'{box[0,0]} {box[0,1]} 0.0 ')
                xyz_file.write(f'{box[1,0]} {box[1,1]} 0.0 ')
                xyz_file.write(f'{box[2,0]} {box[2,1]} 0.0')  
            else:
                raise Exception('only implemented style for boxes are np.ndarrays with shape (3,3) or (3,2) or (3,1).')       
            xyz_file.write('" - ')
        xyz_file.write("Generated XYZ file with optional properties\n")
        
        # Write atom data
        for i in range(n_atoms):
            atom_line = f"{elements[i]} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}"
            if additional_data is not None:
                
                # Add the additional per-atom data
                atom_line += ' ' + ' '.join([f"{additional_data[i, j]:.6f}" for j in range(additional_data.shape[1])])
            xyz_file.write(atom_line + "\n")


def write_xyz_movie(filename, elements_list, coords_list, additional_data_list=None, box_list=None):
    """
    Writes an xyz movie by reiterating the usage of write_xyz function.
    
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
    box: np.ndarray
        a box to be written to file

    Returns:
        None
    """

    if additional_data_list is None:
        additional_data_list = [None] * len(elements_list)
    if box_list is None:
        box_list = [None] * len(elements_list)

    for iframe, (els, coords, add_data, box) in enumerate( zip(elements_list, coords_list, additional_data_list, box_list) ):
        if iframe == 0:
            write_xyz(filename, els, coords, add_data, box=box, mode='w')
        else:
            write_xyz(filename, els, coords, add_data, box=box, mode='a')


def write_xyz_movie_old(frame, filename, elements, coords, additional_data=None):
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
