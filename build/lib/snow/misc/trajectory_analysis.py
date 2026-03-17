#use ase+pysnow to compute the gyration radius along a trajectory
#or write a function that might be put in snow?
#NB gyr rad can be also calculated as the sum of the gyration tensor eigenvalues
#remember to add the weights (masses) in the gyration tensor calculation - and maybe to default to [1]*len(atoms)

import numpy as np

def compute_MSD(movie, ids=None, masks=None, step_by_step=False):
    """
    Computes the mean squared displacement along a trajectory. The MSD is computed by tracking individual
    atoms in their movement, so you might need to provide a list of ids to follow individual atoms along the \n
    list of coordinates (trajectory). A mask can be used to select specific (groups of) atoms to contribute \n
    to the calculation. The calculation can be done relative to the initial positions or to the preceding \n
    timestep by appropriately setting the step_by_step flag.

    Parameters
    ----------
    movie : Tuple[np.ndarray] - or list?
        (list of) coordinates of sequential frames of the system
    ids : Tuple[np.ndarray] - or list?
        (list of) IDs of atoms in the frames - 0-based! Be careful, LAMMPS gives this as 1-based. \n
        default to range(Natoms) for each frame
    masks : Tuple[np.ndarray] - or list?
        (list of) array of bools for whether to count for an atom or not in the calculation of the MSD. \n
        This should follow the order / convention of the positions in movie to refer to individual atoms, and will be 
        reordered according to the ID mapping by the function. Default to all True
    step_by_step : bool
        If True, the MSD at each timestep is computed relative to the timestep before. If False, it is 
        computed relative to the initial position. Default to False

    Returns
    -------
    ndarray
        mean squared displacement along the trajectory
    
    """

    if masks is None: #set default mask
        masks = [[True]*len(movie[0])]*len(movie)
    
    if ids is None: #set default ids
        ids = [list( range( len(movie[0]) ) )]*len(movie)
    
    for i in range(len(ids)):
        if 0 not in ids[i]:
            ids[i] = [id_ - 1 for id_ in ids[i]]

    ref_pos = movie[0]
    ref_pos = ref_pos[ids[0]] #reorder w.r.t ids
    MSD = np.zeros(len(movie))

    for i, pos in enumerate(movie):

        if i==0:
            continue

        if(step_by_step):
            ref_pos = movie[i-1]
            ref_pos = ref_pos[ids[i-1]]
        
        #compute MSD
        pos    = pos[ids[i]]
        diff2  = np.sum( (pos - ref_pos)**2, axis=1)
        mask   = np.array(masks[i])
        ordered_mask = mask[ ids[i] ]
        MSD[i] = diff2[ ordered_mask ].mean() #the mask is reordered according to the ids before being applied

    return MSD

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