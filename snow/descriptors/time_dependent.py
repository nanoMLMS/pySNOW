import numpy as np

def compute_MSD(movie, masks=None, step_by_step=False):
    """
    Computes the mean squared displacement along a trajectory. A mask can be used to select specific (groups of) \n
    atoms to contribute to the calculation. The calculation can be done relative to the initial positions or to \n
    the preceding timestep by appropriately setting the step_by_step flag.

    Parameters
    ----------
    movie : Tuple[np.ndarray] - or list?
        (list of) coordinates of sequential frames of the system
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

    ref_pos = movie[0]
    MSD = np.zeros(len(movie))

    for i, pos in enumerate(movie):

        if i==0:
            continue

        if(step_by_step):
            ref_pos = movie[i-1]
        
        #compute MSD
        diff2  = np.sum( (pos - ref_pos)**2, axis=1)
        mask   = np.array(masks[i])
        MSD[i] = diff2[ mask ].mean()

    return MSD