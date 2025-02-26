from os import write

import numpy as np
from snow.lodispp.utils import nearest_neighbours, adjacency_matrix, coordination_number, pair_list
import os



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
                current_length, current_is_cycle = dfs(neighbor, start, visited, path_length + 1)
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





def calculate_cna(index_frame, coords, cut_off, return_pair=False) -> (int, np.ndarray):
    neighbors_list = nearest_neighbours(index_frame, coords, cut_off)



    pairs = pair_list(index_frame=index_frame, coords=coords, cut_off=cut_off)
    neigh_list, coord_numb = coordination_number(index_frame=index_frame, coords=coords, cut_off=cut_off,
                                                 neigh_list=True)


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




    cna=np.column_stack((r, s, t))




    if return_pair:
        return  len(pairs), cna, ret_pair

    return len(pairs), cna





def write_cna(frame, len_pair, cna, pair_list, file_path=None, signature=True, pattern=True):


    if frame == 0 and os.path.exists(file_path+'signatures.csv'):
        os.remove(file_path+'signatures.csv')

    if frame == 0 and os.path.exists(file_path + 'pattern.csv'):
        os.remove(file_path + 'pattern.csv')

    perc = 100 * np.unique(cna, axis=0, return_counts=True)[1] / len_pair

    if signature==True:

        with open(file_path + 'signatures.csv', "a") as f:
            f.write(f"\n{frame}\n")

            for i, p in enumerate(pair_list):
                f.write(f"{p[0]}, {p[1]}, {cna[i]}\n")


    if pattern == True:
        with open(file_path + 'pattern.csv', "a") as f:
            f.write(f"\n{frame}\n")

            for i, p in enumerate(perc):
                f.write(f"{np.unique(cna, axis=0, return_counts=True)[0][i]}, {np.unique(cna, axis=0, return_counts=True)[1][i]},{p}\n")




