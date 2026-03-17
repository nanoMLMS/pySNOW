from snow.lodispp.utils import pair_list, coordination_number
import numpy as np

def progress_bar(current, total, length=50):
    """
    Prints a nice progess bar to make look like it is doing something
    """
    percent = current / total
    filled_length = int(length * percent)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r[{bar}] {percent * 100:.2f}%', end='')
    return


def agcn_calculator(index_frame, coords, cut_off, dbulk : list[float], thr_cn: int, gcn_max = 12.0, strained: bool = False):
    """
    """
    neigh_list, coord_numbers = coordination_number(index_frame, coords, cut_off, neigh_list=True)
    n_atoms = len(coord_numbers)
    agcn = np.zeros(n_atoms)
    sites=[]

    for i, atom_neighbors in enumerate(neigh_list):
        if coord_numbers[i] >= thr_cn:
            continue
        sites.append(coords[i])
        if strained:
            sgcn=0
            for nb in neigh_list[i]:
                for nnb in neigh_list[nb]:
                    d_nb_nnb= np.linalg.norm(coords[nb] - coords[nnb])
                    sgcn += dbulk/d_nb_nnb
            self_sgcn=0
            for nb in neigh_list[i]:
                break
                d_nb_nnb= np.linalg.norm(coords[nb] - coords[i])
                self_sgcn += dbulk/d_nb_nnb
            agcn[i]=((sgcn-self_sgcn)/gcn_max)
        else:
            agcn_i = sum(coord_numbers[neigh] for neigh in atom_neighbors)# - coord_numbers[i]
            agcn[i]=(agcn_i / gcn_max)
    return sites,agcn



def bridge_gcn(index_frame: int, coords: np.ndarray, cut_off: float, dbulk : list[float], thr_cn: int, gcn_max=18.0, phantom=False, strained: bool = False):
    """
    Identifies bridge absorption sites and computes the Generalized Coordination Number (GCN)
    for a site. The GCN is defined as the sum of the coordination numbers of the neighbors
    of the two atoms forming the site, counted only once.

    Parameters
    ----------
    - index_frame : int
        Index of the frame relative to the snapshot, primarily for reference.
    - coords : ndarray
        Array of the coordinates of the atoms forming the system.
    - cut_off : float
        The cutoff distance for determining nearest neighbors.
    - gcn_max : float, optional
        Maximum typical coordination number in the specific system (default is 18.0).
    - phantom : bool, optional
        If True, also returns the coordinates of the midpoints between pairs for
        representation and testing (default is False).

    Returns
    -------
    If phantom is True:
def three_hollow_gcn(index_frame: int, coords: np.ndarray, cut_off: float, thr_cn : int , gcn_max : float =18.0, phantom : bool =False) -> list:
        tuple
            - ndarray: Coordinates of the midpoints.
            - list: List of pairs.
            - ndarray: Values of the bridge GCN ordered as the pairs.
    Otherwise:
        tuple
            - list: List of pairs.
            - ndarray: Values of the bridge GCN.
    """
    pairs = pair_list(index_frame=index_frame, coords=coords, cut_off=cut_off)
    neigh_list, coord_numb = coordination_number(index_frame=index_frame, coords=coords, cut_off=cut_off, neigh_list=True)
    #b_gcn = np.zeros(len(pairs))
    b_gcn=[]
    sites=[]
    for i, p in enumerate(pairs):
        if not (coord_numb[p[0]] < thr_cn and coord_numb[p[1]] < thr_cn):
            continue
        neigh_1 = neigh_list[p[0]]
        neigh_2 = neigh_list[p[1]]
        neigh_unique_12 = np.unique(np.concatenate((neigh_1, neigh_2)))
        if strained:
            sgcn = 0
            for nb in neigh_unique_12:
                for nnb in neigh_list[nb]:
                    d_nb_nnb= np.linalg.norm(coords[nb] - coords[nnb])
                    sgcn += dbulk/d_nb_nnb
            self_sgcn=0
            for nb in p :
                for nnb in neigh_list[nb]:
                    d_nb_nnb= np.linalg.norm(coords[nb] - coords[nnb])
                    self_sgcn += dbulk/d_nb_nnb
            b_gcn.append((sgcn-self_sgcn)/gcn_max)
        else:
            b_gcn_i = sum(coord_numb[neigh] for neigh in neigh_unique_12) - (coord_numb[p[0]] + coord_numb[p[1]])
            b_gcn.append(b_gcn_i / gcn_max)
        if phantom:
            pos_1 = coords[p[0]]
            pos_2 = coords[p[1]]
            sites.append((pos_1 + pos_2) / 2)
    if phantom:
        return sites, b_gcn
    else:
        return b_gcn


    

def three_hollow_gcn(index_frame: int, coords: np.ndarray, cut_off: float, thr_cn: int, dbulk: list[float],
                     gcn_max: float = 22.0, strained: bool = False) -> list:
    """
    Finds the location of three-hollow sites and returns their location and GCN
    Parameters
    ----------
        index_frame : int
            Number of the frame if a movie.
        coords: ndarray
            Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
        cut_off : float
            Cutoff distance for finding neighbors in angstrom.
        thr_cn : int
            An atom is considered in the surface if its CN < thr_cn
        gcn_max:
            GCN max in the formula for computing the GCN
        strained:
        dbulk

    Returns
    -------
        sites : list
            Midpoint of triplets that form a three hollow site
        th_gcn: list
            GCN of the three hollow sites

    """
    triplets = []
    sites = []
    th_gcn = []
    pairs = pair_list(index_frame=index_frame, coords=coords, cut_off=cut_off)
    # neighbor list and coordination number not compatible!
    neigh_list, coord_numb = coordination_number(index_frame=index_frame, coords=coords, cut_off=cut_off,
                                                 neigh_list=True)
    for i, p in enumerate(pairs):
        progress_bar(i, len(pairs) - 1, 50)
        # we check that both are at the surface
        # thr_cn is a threshold on cn to check if we are at surface
        if not (coord_numb[p[0]] < thr_cn and coord_numb[p[1]] < thr_cn):
            continue
        neigh_1 = neigh_list[p[0]]
        neigh_2 = neigh_list[p[1]]
        neigh_unique_12 = np.intersect1d(neigh_1, neigh_2, assume_unique=True)
        for cn in neigh_unique_12:
            if cn != p[0] and cn != p[1] and coord_numb[cn] < thr_cn:
                new_triplet = sorted([p[0], p[1], cn])
                if not new_triplet in triplets:  # this is a new surface triplet
                    triplets.append(new_triplet)
                    sites.append((coords[p[0]] + coords[p[1]] + coords[cn]) / 3)
                    neigh_unique_triplet = []
                    for idx in new_triplet:
                        neigh_unique_triplet += neigh_list[idx]
                    neigh_unique_triplet = np.unique(neigh_unique_triplet)
                    if strained:
                        sgcn = 0
                        for nb in neigh_unique_triplet:
                            for nnb in neigh_list[nb]:
                                d_nb_nnb = np.linalg.norm(coords[nb] - coords[nnb])
                                sgcn += dbulk / d_nb_nnb
                        self_sgcn = 0
                        for nb in new_triplet:
                            for nnb in neigh_list[nb]:
                                d_nb_nnb = np.linalg.norm(coords[nb] - coords[nnb])
                                self_sgcn += dbulk / d_nb_nnb
                        th_gcn.append((sgcn - self_sgcn) / gcn_max)
                    else:
                        th_gcn_i = sum(coord_numb[neigh] for neigh in neigh_unique_triplet) - sum(
                            coord_numb[neigh] for neigh in new_triplet)
                        th_gcn.append(th_gcn_i / gcn_max)

    print("\nDone three hollow")
    return sites, th_gcn


def four_hollow_gcn(index_frame: int, coords: np.ndarray, cut_off: float, thr_cn: int, dbulk: list[float],
                    gcn_max: float = 26.0, strained: bool = False) -> list:
    """
    Finds the location of four-hollow sites and returns their location and GCN
    Parameters
    ----------
        index_frame : int
            Number of the frame if a movie.
        coords: ndarray
            Array with the XYZ coordinates of the atoms, shape (n_atoms, 3).
        cut_off : float
            Cutoff distance for finding neighbors in angstrom.
        thr_cn : int
            An atom is considered in the surface if its CN < thr_cn
        gcn_max:
            GCN max in the formula for computing the GCN

    Returns
    -------
        sites : list
            Midpoint of triplets that form a four hollow site
        th_gcn: list
            GCN of the four hollow sites

    """
    fours = []
    sites = []
    fh_gcn = []
    pairs = pair_list(index_frame=index_frame, coords=coords, cut_off=cut_off)
    # neighbor list and coordination number not compatible!
    neigh_list, coord_numb = coordination_number(index_frame=index_frame, coords=coords, cut_off=cut_off,
                                                 neigh_list=True)
    snb, _ = coordination_number(index_frame=index_frame, coords=coords, cut_off=cut_off * 1.3, neigh_list=True)
    current = 0
    for j in range(len(pairs)):
        for k in range(j):
            progress_bar(current, len(pairs) * (len(pairs) - 1) / 2)
            current += 1
            indices = [pairs[j][0], pairs[j][1], pairs[k][0], pairs[k][1]]
            check = False
            for idx in indices:  # check if all atoms are in surface
                if coord_numb[idx] >= thr_cn:
                    check = True
                # print(coord_numb[idx],thr_cn,check)
            if check:
                continue
            check = len(set(indices))
            if check != 4:
                continue  # got a shared atom
            # check if pairs are common atmostsecond neighbors
            common_nb_j = np.intersect1d(snb[indices[0]], snb[indices[1]], assume_unique=True)
            if (indices[2] in common_nb_j) and (indices[3] in common_nb_j):
                new_fours = sorted(indices)
                if not new_fours in fours:
                    fours.append(new_fours)
                    newsite = np.array([0., 0., 0.])
                    for at in new_fours:
                        newsite += coords[at]
                    sites.append(newsite / 4.0)
                    neigh_unique_four = []
                    for idx in new_fours:
                        neigh_unique_four += neigh_list[idx]
                    neigh_unique_four = np.unique(neigh_unique_four)
                    if strained:
                        sgcn = 0
                        for nb in neigh_unique_four:
                            for nnb in neigh_list[nb]:
                                d_nb_nnb = np.linalg.norm(coords[nb] - coords[nnb])
                                sgcn += dbulk / d_nb_nnb
                        self_sgcn = 0
                        for nb in new_fours:
                            for nnb in neigh_list[nb]:
                                d_nb_nnb = np.linalg.norm(coords[nb] - coords[nnb])
                                self_sgcn += dbulk / d_nb_nnb
                        fh_gcn.append((sgcn - self_sgcn) / gcn_max)
                    else:
                        fh_gcn_i = sum(coord_numb[neigh] for neigh in neigh_unique_four) - sum(
                            coord_numb[neigh] for neigh in new_fours)
                        fh_gcn.append(fh_gcn_i / gcn_max)
    # for f in fours[:5]:
    # print(coord_numb[f[0]],coord_numb[f[1]],coord_numb[f[2]],coord_numb[f[3]])
    print("\nDone four hollow")
    return sites, fh_gcn