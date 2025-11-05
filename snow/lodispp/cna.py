import os
from os import write

import numpy as np
from scipy.spatial import KDTree

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


def calculate_cna_fast(
    index_frame, coords, cut_off=None, return_pair=False, pbc=False
):
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

    if cut_off == None:
        r_i = np.zeros(len(coords))

    neigh_list = nearest_neighbours(
        index_frame=index_frame, coords=coords, cut_off=cut_off, pbc=pbc
    )
    pairs = pair_list(
        index_frame=index_frame, coords=coords, cut_off=cut_off, pbc=pbc
    )

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
    cna_unique=True,
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


def cna_peratom(
    index_frame: int,
    coords: np.ndarray,
    cut_off: float = None,
    pbc: bool = False,
):
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
        index_frame=index_frame,
        coords=coords,
        cut_off=cut_off,
        return_pair=True,
        pbc=pbc,
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


import numpy as np
from tqdm import tqdm


def cnap_peratom(
    index_frame: int,
    coords: np.ndarray,
    cut_off: float = None,
    pbc: bool = False,
) -> np.ndarray:
    """
    Computes the CNA patterns per atom and assigns an integer structure ID
    (see README for mapping).

    Parameters
    ----------
    index_frame : int
        Frame index (unused here but kept for compatibility)
    coords : np.ndarray
        (N, 3) array with atomic coordinates
    cut_off : float
        Cutoff radius for neighbor determination
    pbc : bool
        Whether to use periodic boundary conditions

    Returns
    -------
    np.ndarray
        Array of integer structure IDs per atom
    """
    # Compute CNA info
    cna = cna_peratom(1, coords, cut_off, pbc=pbc)
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
    for i in tqdm(range(n_atoms), desc="Processing CNA patterns"):
        sigs = np.array(cna[i][0])
        counts = np.array(cna[i][1]).flatten()

        if len(sigs) == 0:
            continue

        cna_atom[i] = match_pattern(sigs, counts)

    return cna_atom
