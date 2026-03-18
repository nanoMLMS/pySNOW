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


from snow.descriptors.utils import (
    adjacency_matrix,
    nearest_neighbours,
    pair_list,
)

from snow.descriptors.coordination import coordination_number

### start Sofia test
def fortran_like_chain_length(neigh_common, neigh_list, a, b):
    """
    CNA t-signature (Honeycutt–Andersen), Fortran-equivalent.

    Parameters
    ----------
    neigh_common : array-like
        Common nearest neighbours of atoms a and b
    neigh_list : dict or list
        Neighbour list for each atom
    a, b : int
        Indices of the central atom pair
    """

    neigh_common = list(neigh_common)
    n = len(neigh_common)
    if n < 2:
        return 0

    pair = {a, b}

    # adjacency restricted to common neighbours ONLY
    adj = {
        u: set(
            v for v in neigh_list[u]
            if v in neigh_common and v not in pair
        )
        for u in neigh_common
    }

    max_len = 0

    def backtrack(path, used):
        nonlocal max_len

        # number of consecutive bonds
        max_len = max(max_len, len(path) - 1)

        # pruning
        if len(path) + (n - len(used)) - 1 <= max_len:
            return

        last = path[-1]
        for nxt in adj[last]:
            if nxt not in used:
                used.add(nxt)
                backtrack(path + [nxt], used)
                used.remove(nxt)

    # try all starting points
    for start in neigh_common:
        backtrack([start], {start})

    # check cycle closure ONLY if full chain
    if max_len == n - 1:
        for u in neigh_common:
            if neigh_common[0] in adj[u]:
                max_len = n
                break

    return max_len

def calculate_cna_sofia(coords, cut_off, return_pair=False):
    """
    Common Neighbour Analysis (Honeycutt–Andersen)

    Returns
    -------
    n_pairs : int
    cna : ndarray (Npairs, 3) with (r, s, t)
    pairs : list of tuples (optional)
    """

    neigh_list = nearest_neighbours(coords, cut_off)
    pairs = pair_list(coords=coords, cut_off=cut_off)

    r = np.zeros(len(pairs), dtype=int)
    s = np.zeros(len(pairs), dtype=int)
    t = np.zeros(len(pairs), dtype=int)

    if return_pair:
        ret_pair = []

    for i, (a, b) in enumerate(pairs):

        # common neighbours (exclude central pair explicitly)
        neigh_common = np.intersect1d(neigh_list[a], neigh_list[b])
        neigh_common = neigh_common[
            (neigh_common != a) & (neigh_common != b)
        ]

        if return_pair:
            ret_pair.append((a, b))

        # r: number of common neighbours
        r[i] = len(neigh_common)

        # s: number of bonds between common neighbours
        s_i = 0
        for idx, j in enumerate(neigh_common):
            for k in neigh_common[idx + 1:]:
                if k in neigh_list[j]:
                    s_i += 1
        s[i] = s_i

        # t: longest CNA chain (Fortran-like)
        t[i] = fortran_like_chain_length(
            neigh_common, neigh_list, a, b
        )

    cna = np.column_stack((r, s, t))

    if return_pair:
        return len(pairs), cna, ret_pair

    return len(pairs), cna





import numpy as np

def cna_percentages_sofia(coords, cut_off, r_bulk_threshold=12):
    """
    CNA con percentuali, separando coppie bulk e surface.
    
    Parameters
    ----------
    coords : np.ndarray
        (N,3) array con le coordinate atomiche
    cut_off : float
        Cutoff per nearest neighbours
    r_bulk_threshold : int
        Numero minimo di vicini per considerare un atomo "bulk"
    
    Returns
    -------
    dict
        'total' : dict[signature tuple] -> percentuale
        'bulk'  : dict[signature tuple] -> percentuale
        'surface': dict[signature tuple] -> percentuale
    """
    # Frame dummy (0) perché calculate_cna_sofia richiede index_frame
    frame = 0

    # CNA totale
    n_pairs, cna, pairs = calculate_cna_sofia(frame, coords, cut_off, return_pair=True)

    # Signature uniche totali
    unique_sigs, counts = np.unique(cna, axis=0, return_counts=True)
    percentages_total = 100 * counts / n_pairs
    total_dict = {tuple(sig): perc for sig, perc in zip(unique_sigs, percentages_total)}

    # Ora dividiamo bulk vs surface
    bulk_counts = {}
    surface_counts = {}
    
    # Precalcolo numero di vicini per ogni atomo
    neigh_list = nearest_neighbours(frame, coords, cut_off)
    num_neigh = np.array([len(n) for n in neigh_list])

    for sig, pair in zip(cna, pairs):
        # Atomi coinvolti nella coppia
        a, b = pair
        # Se entrambi hanno num_neigh >= r_bulk_threshold → bulk
        if num_neigh[a] >= r_bulk_threshold and num_neigh[b] >= r_bulk_threshold:
            bulk_counts[tuple(sig)] = bulk_counts.get(tuple(sig), 0) + 1
        else:
            surface_counts[tuple(sig)] = surface_counts.get(tuple(sig), 0) + 1

    # Trasformiamo in percentuali
    n_bulk = sum(bulk_counts.values())
    n_surface = sum(surface_counts.values())

    bulk_dict = {sig: 100*cnt/n_bulk for sig, cnt in bulk_counts.items()} if n_bulk>0 else {}
    surface_dict = {sig: 100*cnt/n_surface for sig, cnt in surface_counts.items()} if n_surface>0 else {}

    return {
        'total': total_dict,
        'bulk': bulk_dict,
        'surface': surface_dict
    }




### end Sofia test
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
    coords, cut_off, return_pair=False
) -> tuple[int, np.ndarray]:
    """_summary_

    Parameters
    ----------
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

    neigh_list = nearest_neighbours(coords, cut_off)

    pairs = pair_list(coords=coords, cut_off=cut_off)

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
    coords, cut_off=None, return_pair=False, pbc=False,display_progress=False
):
    """
    Faster version of calculate_cna that precomputes neighbor sets.

    Parameters
    ----------
    
        _description_
    coords : ndarray
        3xNatoms array containing the coordinates of each atom
    cut_off : float
        cutoff radius for the determination of nearest neighbors
    return_pair : bool, optional
        Whether to return an ordered list of the indices of the atoms forming a given pair, by default False
    display_progress: bool
        Wheter to display a progress bar

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
        coords=coords, cut_off=cut_off, pbc=pbc
    )
    pairs = pair_list(
        coords=coords, cut_off=cut_off, pbc=pbc
    )

    # Precompute neighbor sets for fast membership tests
    neigh_sets = [set(neigh) for neigh in neigh_list]

    # Initialize result arrays
    r = np.empty(len(pairs), dtype=int)
    s = np.empty(len(pairs), dtype=float)
    t = np.empty(len(pairs), dtype=float)
    ret_pair = [] if return_pair else None

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

    if cna_unique == True:
        with open(file_path + "cna_unique.csv", "a") as f:
            f.write(f"\n{frame}\n")

            for i, p in enumerate(perc):
                f.write(
                    #f"{np.unique(cna, axis=0, return_counts=True)[0][i]}, {np.unique(cna, axis=0, return_counts=True)[1][i]},{p}\n"
                    f"{p}, {np.unique(cna, axis=0, return_counts=True)[0][i]},\n" #percentage and signature
                )


def cna_peratom(
    coords: np.ndarray,
    cut_off: float = None,
    pbc: bool = False,
):
    """
    Optimized per-atom CNA calculation by precomputing a mapping from atom indices
    to pair indices. This avoids scanning the entire pair list for every atom.

    Parameters
    ----------
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
    coords: np.ndarray,
    cut_off: float = None,
    pbc: bool = False,
    display_progress: bool = False,
) -> np.ndarray:
    """
    Computes the CNA patterns per atom and assigns an integer structure ID
    (see README for mapping).

    Parameters
    ----------
        Frame index (unused here but kept for compatibility)
    coords : np.ndarray
        (N, 3) array with atomic coordinates
    cut_off : float
        Cutoff radius for neighbor determination
    pbc : bool
        Whether to use periodic boundary conditions
    display_progress: bool
        Wheter to display a progress bar

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
    iterator = tqdm(range(n_atoms), desc="Processing CNA patterns") if display_progress \
            else range(n_atoms)
    for i in iterator:
        sigs = np.array(cna[i][0])
        counts = np.array(cna[i][1]).flatten()

        if len(sigs) == 0:
            continue

        cna_atom[i] = match_pattern(sigs, counts)

    return cna_atom
