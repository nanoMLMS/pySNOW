import numpy as np

def compute_MSD(movie, masks=None, step_by_step=False):
    """
    Computes the mean squared displacement (MSD) along a trajectory. 
    
    A mask can be used to select specific (groups of) atoms to contribute to the MSD.
    The calculation can be done relative to the initial positions or to
    the preceding timestep by appropriately setting the step_by_step flag.

    Parameters
    ----------
    movie : list[np.ndarray]
        list of coordinates of sequential frames of the system. 
        Each list item should be a (n,3) np.ndarray coordinates array
    masks : list[list[bool]], optional
        list of lists of bools for whether to count for an atom or not in the calculation of the MSD.
        Default to all True.
    step_by_step : bool, default False
        If True, the MSD at each timestep is computed relative to the timestep before. If False, it is 
        computed relative to the initial position. Default to False

    Returns
    -------
    MSD : np.ndarray
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