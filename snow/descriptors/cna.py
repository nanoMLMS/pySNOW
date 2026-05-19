import os
from os import write

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not installed, define a dummy tqdm that does nothing.
    def tqdm(iterable, **kwargs):
        return iterable


from snow.descriptors.utils import (
    nearest_neighbours,
    pair_list,
)

def longest_path_or_cycle(neigh_common, neigh_list):
    """
    Find the longest path or cycle in the subgraph induced by a subset of nodes.

    Builds a subgraph from neigh_common using adjacency information from neigh_list,
    then runs a depth-first search (DFS) from each node to find the longest path
    or cycle in the subgraph.

    Parameters
    ----------
    neigh_common : iterable
        Subset of nodes to consider (e.g. common neighbors of two atoms).
    neigh_list : dict
        Full adjacency list of the graph, mapping each node to its neighbors.
        Only edges between nodes in neigh_common are considered.

    Returns
    -------
    int
        Length of the longest path or cycle found in the subgraph.
    """

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
    coords, cut_off, return_pairs=False, pbc=False, box=None
) -> tuple[int, np.ndarray]:
    """perform the common neighbour analysis for the provided coords.

    Automatically finds pairs and assigns a cna signature to each pair. The pairs can
    be returned as tuples of indexes of the coordinates if `return_pair`=True.

    Parameters
    ----------
    coords : np.ndarray
        array containing the coordainates of each atom
    cut_off : float
        cutoff radius for the determination of nearest neighbours. If None, an adaptive cutoff is computed
    return_pairs : bool, default False
        Wether to return an ordered list of the indices of the atoms forming each pair, by default False
    pbc : bool, default False
        whether to use periodic boundary conditions or not.
    box : ndarray, default None
        if pbc are enabled, the simulation box is needed to compute periodic neighbours.

    Returns
    -------
    tuple
        - int : number of pairs found
        - np.ndarray : cna signatures (r,s,t) for each found pair in a (n_pairs, 3) array
        - list : the indexes of atoms in pairs corresponding to the computed cna signatures. Only returned if `return_pairs`=True
    """

    neigh_list = nearest_neighbours(
        coords=coords, cut_off=cut_off, pbc=pbc, box=box
    )
    pairs = pair_list(
        coords=coords, cut_off=cut_off, pbc=pbc, box=box
    )

    r = np.zeros(len(pairs))
    s = np.zeros(len(pairs))
    t = np.zeros(len(pairs))

    if return_pairs:
        ret_pair = []

    for i, p in enumerate(pairs):
        neigh_1 = neigh_list[p[0]]
        neigh_2 = neigh_list[p[1]]
        neigh_common = np.intersect1d(neigh_1, neigh_2)
        if return_pairs:
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

    if return_pairs:
        return len(pairs), cna, ret_pair
    else:
        return len(pairs), cna


def calculate_cna_fast(
    coords, cut_off, return_pairs=False, pbc=False, box=None, display_progress=False
):
    """
    Faster version of calculate_cna that precomputes neighbor sets.

    performs common neighbour analysis with a faster algorithm, useful for N>1e5 atoms.
    This method avoids scanning the entire pair list for every atom. 

    Parameters
    ----------
    coords : ndarray
        array containing the coordinates of atoms in the system
    cut_off : float
        cutoff radius for the determination of nearest neighbors. If None, an adaptive cutoff is computed
    return_pair : bool, default False
        Whether to return an ordered list of the indices of the atoms forming a given pair, by default False
    pbc : bool, default False
        whether to use periodic boundary conditions or not.
    box : ndarray, default None
        if pbc are enabled, the simulation box is needed to compute periodic neighbours.
    display_progress: bool, default False
        Wheter to display a progress bar - needs the tqdm library

    Returns
    -------
    tuple
        - int : number of pairs found
        - np.ndarray : cna signatures (r,s,t) for each found pair in a (n_pairs, 3) array
        - list : the indexes of atoms in pairs corresponding to the computed cna signatures. Only returned if `return_pairs`=True
    """

    # Get neighbor list and pair list
    neigh_list = nearest_neighbours(
        coords=coords, cut_off=cut_off, pbc=pbc, box=box
    )
    pairs = pair_list(
        coords=coords, cut_off=cut_off, pbc=pbc, box=box
    )

    # Precompute neighbor sets for fast membership tests
    neigh_sets = [set(neigh) for neigh in neigh_list]

    # Initialize result arrays
    r = np.empty(len(pairs), dtype=int)
    s = np.empty(len(pairs), dtype=float)
    t = np.empty(len(pairs), dtype=float)
    ret_pair = [] if return_pairs else None

    iterator = enumerate(tqdm(pairs, desc="Processing pairs")) if display_progress \
            else enumerate(pairs)
    for i, p in iterator:
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

        if return_pairs:
            ret_pair.append(p)

    cna = np.column_stack((r, s, t))
    if return_pairs:
        return len(pairs), cna, ret_pair
    return len(pairs), cna


def cna_peratom(
    coords: np.ndarray,
    cut_off: float,
    pbc: bool = False,
    box: np.ndarray = None
):
    """
    Optimized per-atom CNA calculation. 
    
    Computes the cna signatures for all pairs in the system, and assign to each atom the list of 
    signatures it participates to.
    

    Parameters
    ----------
    coords : np.ndarray
        Array containing the coordinates of the atoms in your system.
    cut_off : float
        Cutoff radius for nearest-neighbor determination. If None, an adaptive cutoff is computed
    pbc : bool, default False
        Whether to use or not periodic boundary conditions
    box : np.ndarray, default None
        Simulation box. Only needed if you enable PBC

    Returns
    -------
    list of tuple[np.ndarray, np.ndarray]
        For each atom, a tuple (unique_signatures, counts) representing the unique
        CNA signatures from all pairs involving that atom and their respective counts.
    """
    # Compute CNA signatures and the corresponding pair list

    _, cna, pair_list = calculate_cna_fast(
        coords=coords,
        cut_off=cut_off,
        return_pairs=True,
        pbc=pbc,
        box = box
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
            unique_signatures, counts = np.unique(
                cna_i, axis=0, return_counts=True
            )
            cna_atom.append((unique_signatures, counts))
        else:
            # If no pairs include atom i, return empty arrays.
            cna_atom.append((np.array([]), np.array([])))
    return cna_atom


def cnap_peratom(
    coords: np.ndarray,
    cut_off: float,
    pbc: bool = False,
    box: np.ndarray = None,
    display_progress: bool = False) -> np.ndarray:
    """
    Computes the per-atom CNA patterns and assigns an integer structure ID.

    Tries to match the cna per atom patterns to known patterns in a database for atomic 
    environment characterization (see README.md for ID-structure mapping).

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array with atomic coordinates
    cut_off : float
        Cutoff radius for neighbor determination. If None, an adaptive cutoff is used
    pbc : bool, default False
        Whether to use or not periodic boundary conditions
    box : np.ndarray, default None
        Simulation box. Only needed if you enable PBC
    display_progress: bool, default False
        Wheter to display a progress bar - needs the tqdm optional dependency library.

    Returns
    -------
    np.ndarray
        Array of (integers) structure IDs per atom
    """

    # Compute CNA info
    cna = cna_peratom(coords, cut_off, pbc=pbc, box=box)
    n_atoms = len(coords)
    cna_atom = np.zeros(n_atoms, dtype=int)

    # --- Define pattern rules as a lookup table ---
    # Each rule is a tuple (required signatures, required counts) -> assigned ID
    PATTERNS = [
        # n_sigs == 1
        ((([5, 5, 5],), (12,)), 5),
        (([[4, 2, 1]], (12,)), 4),
        # n_sigs == 2
        (([[4, 2, 2], [5, 5, 5]], (10, 2)), 3),
        (([[4, 2, 1], [3, 1, 1]], (3, 6)), 15),
        (([[2, 1, 1], [4, 2, 1]], (4, 1)), 11),
        (([[2, 1, 1], [4, 2, 1]], (4, 4)), 12),
        (([[3, 2, 2], [5, 5, 5]], (5, 1)), 14),
        (([[4, 2, 1], [4, 2, 2]], (6, 6)), 16),
        # n_sigs == 3
        (([[1, 0, 0], [2, 1, 1], [4, 2, 2]], (2, 2, 2)), 6),
        (([[2, 0, 0], [3, 1, 1], [4, 2, 1]], (2, 4, 1)), 8),
        (([[2, 1, 1], [3, 1, 1], [4, 2, 1]], (3, 2, 2)), 10),
        (([[3, 1, 1], [3, 2, 2], [4, 2, 2]], (4, 2, 2)), 13),
        (([[2, 1, 1], [3, 1, 1], [4, 2, 1]], (1, 4, 5)), 17),
        # n_sigs == 4
        (([[1, 0, 0], [2, 1, 1], [3, 2, 2], [4, 2, 2]], (1, 2, 1, 1)), 1),
        (([[2, 0, 0], [2, 1, 1], [3, 1, 1], [4, 2, 1]], (1, 2, 2, 1)), 2),
        (([[3, 0, 0], [3, 1, 1], [4, 2, 1], [4, 2, 2]], (2, 4, 2, 2)), 9),
        (([[4, 2, 1], [4, 2, 2], [4, 3, 3], [5, 4, 4]], (4, 4, 2, 2)), 18),
        # n_sigs == 5
        (
            (
                [[2, 0, 0], [3, 0, 0], [3, 1, 1], [3, 2, 2], [4, 2, 2]],
                (2, 1, 2, 1, 1),
            ),
            7,
        ),
    ]

    def match_pattern(sigs, counts):
        """Try to match CNA signatures/counts to a known structure pattern."""
        for (req_sigs, req_counts), struct_id in PATTERNS:
            if len(req_sigs) != len(sigs):
                continue
            # Convert both to sets of tuples for order-insensitive comparison
            sig_dict = {tuple(sig): cnt for sig, cnt in zip(sigs, counts)}
            if all(
                tuple(rs) in sig_dict and sig_dict[tuple(rs)] == rc
                for rs, rc in zip(req_sigs, req_counts)
            ):
                return struct_id
        return 0  # default (unidentified)

    # --- Process atoms ---
    iterator = tqdm(range(n_atoms), desc="Processing CNA patterns") if display_progress \
            else range(n_atoms)
    for i in iterator:
        sigs = np.array(cna[i][0])
        counts = np.array(cna[i][1]).flatten()

        if len(sigs) == 0:
            continue

        cna_atom[i] = match_pattern(sigs, counts)

    return cna_atom

def write_cna(
    frame,
    len_pair,
    cna,
    pair_list,
    file_path='./',
    signature=True,
    cna_unique=True
):
    """
    export cna analysis to files in .csv format

    save to file the indexes of atoms making up a pair and their signature (if `signature`=True).
    save to file the unique cna signatures, and their occurrence wrt to the total number as a percentage
    Note that if `frame`==0, previous files named 'signatures.csv' and 'cna_unique.csv' will be flushed.

    Parameters
    ----------
    frame : int
        config frame id, mostly for reference (and written to file)
    len_pair : int
        number of pairs in the system
    cna : np.ndarray
        array of the (previously computed) cna signatures
    pair_list: list
        list of lists of atomic indexes, to label atoms making up pairs
    file_path : str, default './'
        folder to write output files in
    signature: bool, default True
        write cna signatures together with corresponding pair indexes in a signatures.csv file
    cna_unique : bool, default True
        write unique cna signatures and their occurrence (as a percentage) in a cna_unique.csv file
    """
    #remove old files
    if frame == 0 and os.path.exists(file_path + "signatures.csv") and signature:
        os.remove(file_path + "signatures.csv")

    if frame == 0 and os.path.exists(file_path + "cna_unique.csv") and cna_unique:
        os.remove(file_path + "cna_unique.csv")

    #len_pair = len(cna)
    perc = 100 * np.unique(cna, axis=0, return_counts=True)[1] / len_pair

    if signature == True:

        with open(file_path + "signatures.csv", "a") as f:
            f.write(f"\n{frame}\n")

            for i, p in enumerate(pair_list):
                f.write(f"{p[0]}, {p[1]}, {cna[i]}\n")

    if cna_unique == True:
        with open(file_path + "cna_unique.csv", "a") as f:
            f.write(f"\n{frame}\n")

            for i, p in enumerate(perc):
                f.write(
                    #f"{np.unique(cna, axis=0, return_counts=True)[0][i]}, {np.unique(cna, axis=0, return_counts=True)[1][i]},{p}\n"
                    f"{p}, {np.unique(cna, axis=0, return_counts=True)[0][i]},\n" #percentage and signature
                )