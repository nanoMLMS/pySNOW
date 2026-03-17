import os
from os import write
from scipy.spatial import KDTree
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not installed, define a dummy tqdm that does nothing.
    def tqdm(iterable, **kwargs):
        return iterable
from snow.lodispp.utils import (
    adjacency_matrix,
    coordination_number,
    nearest_neighbours,
    pair_list,
)



def longest_path_or_cycle(neigh_common, neigh_list):

    graph = {node: set() for node in neigh_common}
    for node in neigh_common:
        for neighbor in neigh_list[node]:
            if neighbor in neigh_common:
                graph[node].add(neighbor)

    def dfs(node, start, visited, path_length):
        visited.add(node)
        max_length = path_length
        is_cycle = False

        for neighbor in graph[node]:
            if neighbor == start and path_length > 1:
                max_length = max(max_length, path_length + 1)
                is_cycle = True
            elif neighbor not in visited:
                current_length, current_is_cycle = dfs(
                    neighbor, start, visited, path_length + 1
                )
                if current_is_cycle:
                    is_cycle = True
                max_length = max(max_length, current_length)

        visited.remove(node)
        return max_length, is_cycle

    longest_length = 0
    for node in neigh_common:
        visited = set()
        length, is_cycle = dfs(node, node, visited, 0)
        longest_length = max(longest_length, length)

    return longest_length


def calculate_cna(
    index_frame, coords, cut_off, return_pair=False
) -> tuple[int, np.ndarray]:
    """_summary_

    Parameters
    ----------
    index_frame : int
        _description_
    coords : ndarray
        3xNatoms array containing the coordainates of each atom
    cut_off : float
        cutoff radius for the determination of nearest neaighbours
    return_pair : bool, optional
        Wether to return an ordered list of the inideces of the atoms forming a given pair, by default False

    Returns
    -------
    tuple[int, np.ndarray, list]
        The number of pairs, the cna signatures [r, s, t] for each pair and the ordered list of pairs (if return_pair == True)
    tuple[int, np.ndarray]
        The number of pairs, the cna signatures [r, s, t] for each pair
    """

    neigh_list = nearest_neighbours(index_frame, coords, cut_off)

    pairs = pair_list(index_frame=index_frame, coords=coords, cut_off=cut_off)
    
    
    r = np.zeros(len(pairs))
    s = np.zeros(len(pairs))
    t = np.zeros(len(pairs))

    if return_pair:
        ret_pair = []

    for i, p in enumerate(pairs):
        neigh_1 = neigh_list[p[0]]
        neigh_2 = neigh_list[p[1]]
        neigh_common = np.intersect1d(neigh_1, neigh_2)
        if return_pair:
            ret_pair.append(p)

        # Calculate r and s
        r[i] = len(neigh_common)
        s_i = 0
        for j in neigh_common:
            for n in neigh_list[j]:
                if n in neigh_common:
                    s_i += 1
        s[i] = s_i / 2

        # Calculate the longest chain length
        t[i] = longest_path_or_cycle(neigh_common, neigh_list)

    cna = np.column_stack((r, s, t))

    if return_pair:
        return len(pairs), cna, ret_pair

    return len(pairs), cna

def calculate_cna_fast(index_frame, coords, cut_off = None, return_pair=False, pbc = False):
    """
    Faster version of calculate_cna that precomputes neighbor sets.
    
    Parameters
    ----------
    index_frame : int
        _description_
    coords : ndarray
        3xNatoms array containing the coordinates of each atom
    cut_off : float
        cutoff radius for the determination of nearest neighbors
    return_pair : bool, optional
        Whether to return an ordered list of the indices of the atoms forming a given pair, by default False

    Returns
    -------
    tuple[int, np.ndarray, list]
        The number of pairs, the cna signatures [r, s, t] for each pair and the ordered list of pairs (if return_pair == True)
    tuple[int, np.ndarray]
        The number of pairs, the cna signatures [r, s, t] for each pair
    """
    # Get neighbor list and pair list (assumed to be implemented efficiently)

    if (cut_off == None):
        r_i = np.zeros(len(coords))


    neigh_list = nearest_neighbours(index_frame = index_frame, coords = coords, cut_off = cut_off, pbc=pbc)
    pairs = pair_list(index_frame=index_frame, coords=coords, cut_off=cut_off, pbc = pbc)
    
    # Precompute neighbor sets for fast membership tests
    neigh_sets = [set(neigh) for neigh in neigh_list]

    # Initialize result arrays
    r = np.empty(len(pairs), dtype=int)
    s = np.empty(len(pairs), dtype=float)
    t = np.empty(len(pairs), dtype=float)
    ret_pair = [] if return_pair else None



    for i, p in enumerate(tqdm(pairs, desc="Processing pairs")):
        # Get neighbor sets for the two atoms in the pair
        set1 = neigh_sets[p[0]]
        set2 = neigh_sets[p[1]]
        # Compute common neighbors using set intersection
        common = set1 & set2
        r[i] = len(common)
        
        # For s, sum the number of common neighbors between each neighbor in 'common'
        # and the set 'common' itself, then divide by 2.
        s_val = sum(len(neigh_sets[j] & common) for j in common) / 2
        s[i] = s_val

        # Compute t using the existing function.
        # If longest_path_or_cycle expects a numpy array, we convert common accordingly.
        t[i] = longest_path_or_cycle(np.array(list(common)), neigh_list)

        if return_pair:
            ret_pair.append(p)

    cna = np.column_stack((r, s, t))
    if return_pair:
        return len(pairs), cna, ret_pair
    return len(pairs), cna


def write_cna(
    frame,
    len_pair,
    cna,
    pair_list,
    file_path=None,
    signature=True,
    cna_unique=True
):

    if frame == 0 and os.path.exists(file_path + "signatures.csv"):
        os.remove(file_path + "signatures.csv")

    if frame == 0 and os.path.exists(file_path + "cna_unique.csv"):
        os.remove(file_path + "cna_unique.csv")

    perc = 100 * np.unique(cna, axis=0, return_counts=True)[1] / len_pair

    if signature == True:

        with open(file_path + "signatures.csv", "a") as f:
            f.write(f"\n{frame}\n")

            for i, p in enumerate(pair_list):
                f.write(f"{p[0]}, {p[1]}, {cna[i]}\n")

    if pattern == True:
        with open(file_path + "pattern.csv", "a") as f:
            f.write(f"\n{frame}\n")

            for i, p in enumerate(perc):
                f.write(
                    f"{np.unique(cna, axis=0, return_counts=True)[0][i]}, {np.unique(cna, axis=0, return_counts=True)[1][i]},{p}\n"
                )



def cna_peratom(index_frame: int, coords: np.ndarray, cut_off: float = None, pbc: bool = False):
    """
    Optimized per-atom CNA calculation by precomputing a mapping from atom indices
    to pair indices. This avoids scanning the entire pair list for every atom.
    
    Parameters
    ----------
    index_frame : int
        _description_
    coords : np.ndarray
        Array containing the coordinates of each atom.
    cut_off : float
        Cutoff radius for nearest-neighbor determination.
        
    Returns
    -------
    list of tuple[np.ndarray, np.ndarray]
        For each atom, a tuple (unique_signatures, counts) representing the unique
        CNA signatures from all pairs involving that atom and their respective counts.
    """
    # Compute CNA signatures and the corresponding pair list

    
    _, cna, pair_list = calculate_cna_fast(
        index_frame=index_frame, coords=coords, cut_off=cut_off, return_pair=True, pbc=pbc
    )
    num_atoms = len(coords)
    
    # Precompute a mapping from each atom to the indices of pairs that involve it.
    atom_to_pair = [[] for _ in range(num_atoms)]
    for idx, (i, j) in enumerate(pair_list):
        atom_to_pair[i].append(idx)
        atom_to_pair[j].append(idx)
    
    # Build the per-atom CNA information using the precomputed mapping.
    cna_atom = []
    for i in range(num_atoms):
        pair_indices = atom_to_pair[i]
        if pair_indices:
            # Use NumPy advanced indexing to quickly select the relevant CNA rows.
            cna_i = cna[pair_indices]
            unique_signatures, counts = np.unique(cna_i, axis=0, return_counts=True)
            cna_atom.append((unique_signatures, counts))
        else:
            # If no pairs include atom i, return empty arrays.
            cna_atom.append((np.array([]), np.array([])))
    return cna_atom

def cnap_peratom(index_frame: int, coords: np.ndarray, cut_off: float = None, pbc: bool = False) -> np.ndarray:
    """Computes the CNA Patterns for each atom in the system, assigning to each an integer which can be used to 
    identify the local structure, mapping integer-structure can be found in the README.


    Parameters
    ----------
    index_frame : int
        _description_
    coords : np.ndarray
        3xNatoms array containing the coordainates of the atoms in the system
    cut_off : float
        Cutoff radius for the determination of neighbours

    Returns
    -------
    np.ndarray
        1D array containing for each atom an integer identifing the structure (check the documentation)
        
    """
    cna = cna_peratom(1, coords, cut_off, pbc=pbc)
    cna_atom = np.zeros(len(coords))
    count = 0
    for i in tqdm(range(len(coords)), desc="Processing patterns"):
        

        n_sigs = len(
            cna[i][1]
        )  # number of unique signatures to which atom i partecipatr
        sigs = cna[i][0]  # the unique signatures themselves
        count = cna[i][1]  # the count of each signature
        if n_sigs == 1 and count[0] == 12:
            if (sigs[0] == [5, 5, 5]).all():
                cna_atom[i] = 5
            elif (sigs[0] == [4, 2, 1]).all():
                cna_atom[i] = 4
        elif n_sigs == 2:
            if [4, 2, 2] in sigs and [5, 5, 5] in sigs:

                idx_422 = np.where(np.all(sigs == [4, 2, 2], axis=1))[0]
                idx_555 = np.where(np.all(sigs == [5, 5, 5], axis=1))[0]
                n_422 = count[idx_422][0] if count[idx_422].size > 0 else 0
                n_555 = count[idx_555][0] if count[idx_555].size > 0 else 0

                if n_422 == 10 and n_555 == 2:
                    cna_atom[i] = 3
            if [4, 2, 1] in sigs and [3, 1, 1] in sigs:
                idx_311 = np.where(np.all(sigs == [3, 1, 1], axis=1))[0]
                idx_421 = np.where(np.all(sigs == [4, 2, 1], axis=1))[0]
                n_311 = count[idx_311][0] if count[idx_311].size > 0 else 0
                n_421 = count[idx_421][0] if count[idx_421].size > 0 else 0

                if n_311 == 6 and n_421 == 3:
                    cna_atom[i] = 15
            if [2, 1, 1] in sigs and [4, 2, 1] in sigs:
                idx_421 = np.where(np.all(sigs == [4, 2, 1], axis=1))[0]
                idx_211 = np.where(np.all(sigs == [2, 1, 1], axis=1))[0]

                if (
                    idx_421.size > 0 and idx_211.size > 0
                ):  # Ensure non-empty indices
                    n_421 = count[idx_421][0] if count[idx_421].size > 0 else 0
                    n_211 = count[idx_211][0] if count[idx_211].size > 0 else 0

                    if n_211 == 4 and n_421 == 1:
                        cna_atom[i] = 11
                    elif n_211 == 4 and n_421 == 4:
                        cna_atom[i] = 12
            if [3, 2, 2] in sigs and [5, 5, 5] in sigs:
                idx_322 = np.where(np.all(sigs == [3, 2, 2], axis=1))[0]
                idx_555 = np.where(np.all(sigs == [5, 5, 5], axis=1))[0]
                n_322 = count[idx_322][0] if count[idx_322].size > 0 else 0
                n_555 = count[idx_555][0] if count[idx_555].size > 0 else 0

                if n_322 == 5 and n_555 == 1:
                    cna_atom[i] = 14
            if [4, 2, 1] in sigs and [4, 2, 2] in sigs:
                idx_421 = np.where(np.all(sigs == [4, 2, 1], axis=1))[0]
                idx_422 = np.where(np.all(sigs == [4, 2, 2], axis=1))[0]
                n_422 = count[idx_422][0] if count[idx_422].size > 0 else 0
                n_421 = count[idx_421][0] if count[idx_421].size > 0 else 0

                if n_421 == 6 and n_422 == 6:
                    cna_atom[i] = 16
        elif n_sigs == 3:
            if [1, 0, 0] in sigs and [2, 1, 1] in sigs and [4, 2, 2] in sigs:
                idx_100 = np.where(np.all(sigs == [1, 0, 0], axis=1))[0]
                idx_211 = np.where(np.all(sigs == [2, 1, 1], axis=1))[0]
                idx_422 = np.where(np.all(sigs == [4, 2, 2], axis=1))[0]
                n_100 = count[idx_100][0] if count[idx_100].size > 0 else 0
                n_211 = count[idx_211][0] if count[idx_211].size > 0 else 0
                n_422 = count[idx_422][0] if count[idx_422].size > 0 else 0

                if n_100 == 2 and n_211 == 2 and n_422 == 2:
                    cna_atom[i] = 6
            if [2, 0, 0] in sigs and [3, 1, 1] in sigs and [4, 2, 1] in sigs:
                idx_200 = np.where(np.all(sigs == [2, 0, 0], axis=1))[0]
                idx_311 = np.where(np.all(sigs == [3, 1, 1], axis=1))[0]
                idx_421 = np.where(np.all(sigs == [4, 2, 1], axis=1))[0]
                n_200 = count[idx_200][0] if count[idx_200].size > 0 else 0
                n_311 = count[idx_311][0] if count[idx_311].size > 0 else 0
                n_421 = count[idx_421][0] if count[idx_421].size > 0 else 0

                if n_200 == 2 and n_311 == 4 and n_421 == 1:
                    cna_atom[i] = 8
            if [2, 1, 1] in sigs and [3, 1, 1] in sigs and [4, 2, 1] in sigs:
                idx_211 = np.where(np.all(sigs == [2, 1, 1], axis=1))[0]
                idx_311 = np.where(np.all(sigs == [3, 1, 1], axis=1))[0]
                idx_421 = np.where(np.all(sigs == [4, 2, 1], axis=1))[0]
                n_211 = count[idx_211][0] if count[idx_211].size > 0 else 0
                n_311 = count[idx_311][0] if count[idx_311].size > 0 else 0
                n_421 = count[idx_421][0] if count[idx_421].size > 0 else 0

                if n_211 == 3 and n_311 == 2 and n_421 == 2:
                    cna_atom[i] = 10
            if [3, 1, 1] in sigs and [3, 2, 2] in sigs and [4, 2, 2] in sigs:
                idx_311 = np.where(np.all(sigs == [3, 1, 1], axis=1))[0]
                idx_322 = np.where(np.all(sigs == [3, 2, 2], axis=1))[0]
                idx_422 = np.where(np.all(sigs == [4, 2, 2], axis=1))[0]
                n_311 = count[idx_311][0] if count[idx_311].size > 0 else 0
                n_322 = count[idx_322][0] if count[idx_322].size > 0 else 0
                n_422 = count[idx_422][0] if count[idx_422].size > 0 else 0

                if n_311 == 4 and n_322 == 2 and n_422 == 2:
                    cna_atom[i] = 13
        elif n_sigs == 4:

            if (
                [1, 0, 0] in sigs
                and [2, 1, 1] in sigs
                and [3, 2, 2] in sigs
                and [4, 2, 2] in sigs
            ):
                idx_100 = np.where(np.all(sigs == [1, 0, 0], axis=1))[0]
                idx_211 = np.where(np.all(sigs == [2, 1, 1], axis=1))[0]
                idx_322 = np.where(np.all(sigs == [3, 2, 2], axis=1))[0]
                idx_422 = np.where(np.all(sigs == [4, 2, 2], axis=1))[0]
                n_100 = count[idx_100][0] if count[idx_100].size > 0 else 0
                n_211 = count[idx_211][0] if count[idx_211].size > 0 else 0
                n_322 = count[idx_322][0] if count[idx_322].size > 0 else 0
                n_422 = count[idx_422][0] if count[idx_422].size > 0 else 0

                if n_100 == 1 and n_211 == 2 and n_322 == 1 and n_422 == 1:
                    cna_atom[i] = 1

            if (
                [2, 0, 0] in sigs
                and [2, 1, 1] in sigs
                and [3, 1, 1] in sigs
                and [4, 2, 1] in sigs
            ):
                idx_200 = np.where(np.all(sigs == [2, 0, 0], axis=1))[0]
                idx_211 = np.where(np.all(sigs == [2, 1, 1], axis=1))[0]
                idx_311 = np.where(np.all(sigs == [3, 1, 1], axis=1))[0]
                idx_421 = np.where(np.all(sigs == [4, 2, 1], axis=1))[0]
                n_200 = count[idx_200][0] if count[idx_200].size > 0 else 0
                n_211 = count[idx_211][0] if count[idx_211].size > 0 else 0
                n_311 = count[idx_311][0] if count[idx_311].size > 0 else 0
                n_421 = count[idx_421][0] if count[idx_421].size > 0 else 0

                if n_200 == 1 and n_211 == 2 and n_311 == 2 and n_421 == 1:
                    cna_atom[i] = 2

            if (
                [3, 0, 0] in sigs
                and [3, 1, 1] in sigs
                and [4, 2, 1] in sigs
                and [4, 2, 2] in sigs
            ):
                idx_300 = np.where(np.all(sigs == [3, 0, 0], axis=1))[0]
                idx_311 = np.where(np.all(sigs == [3, 1, 1], axis=1))[0]
                idx_421 = np.where(np.all(sigs == [4, 2, 1], axis=1))[0]
                idx_422 = np.where(np.all(sigs == [4, 2, 2], axis=1))[0]
                n_300 = count[idx_300][0] if count[idx_300].size > 0 else 0
                n_311 = count[idx_311][0] if count[idx_311].size > 0 else 0
                n_421 = count[idx_421][0] if count[idx_421].size > 0 else 0
                n_422 = count[idx_422][0] if count[idx_422].size > 0 else 0

                if n_300 == 2 and n_311 == 4 and n_421 == 2 and n_422 == 2:
                    cna_atom[i] = 9
        elif n_sigs == 5:
            if (
                [2, 0, 0] in sigs
                and [3, 0, 0] in sigs
                and [3, 1, 1] in sigs
                and [3, 2, 2] in sigs
                and [4, 2, 2] in sigs
            ):
                idx_200 = np.where(np.all(sigs == [2, 0, 0], axis=1))[0]
                idx_300 = np.where(np.all(sigs == [3, 0, 0], axis=1))[0]
                idx_311 = np.where(np.all(sigs == [3, 1, 1], axis=1))[0]
                idx_322 = np.where(np.all(sigs == [3, 2, 2], axis=1))[0]
                idx_422 = np.where(np.all(sigs == [4, 2, 2], axis=1))[0]
                n_200 = count[idx_200][0] if count[idx_200].size > 0 else 0
                n_300 = count[idx_300][0] if count[idx_300].size > 0 else 0
                n_311 = count[idx_311][0] if count[idx_311].size > 0 else 0
                n_322 = count[idx_322][0] if count[idx_322].size > 0 else 0
                n_422 = count[idx_422][0] if count[idx_422].size > 0 else 0

                if (
                    n_200 == 2
                    and n_300 == 1
                    and n_311 == 2
                    and n_322 == 1
                    and n_422 == 1
                ):
                    cna_atom[i] = 7
    return cna_atom
