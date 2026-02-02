import numpy as np
from snow.descriptors.utils import distance_matrix, adjacency_matrix

def test_two_atoms():
    coords = np.array([[0, 0, 0], [1, 0, 0]])  # two atoms 1 unit apart
    dist_mat, dist_max, dist_min = distance_matrix(0, coords)
    expected_matrix = np.array([[0, 1],
                                [1, 0]])
    assert np.allclose(dist_mat, expected_matrix)
    assert dist_max == 1
    assert dist_min == 0

def test_three_atoms():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    dist_mat, dist_max, dist_min = distance_matrix(0, coords)
    expected_matrix = np.array([[0, 1, 1],
                                [1, 0, np.sqrt(2)],
                                [1, np.sqrt(2), 0]])
    assert np.allclose(dist_mat, expected_matrix)
    assert dist_max == np.sqrt(2)
    assert dist_min == 0

def test_identical_atoms():
    coords = np.array([[0, 0, 0], [0, 0, 0]])
    dist_mat, dist_max, dist_min = distance_matrix(0, coords)
    expected_matrix = np.array([[0, 0],
                                [0, 0]])
    assert np.allclose(dist_mat, expected_matrix)
    assert dist_max == 0
    assert dist_min == 0

def test_single_atom():
    coords = np.array([[1, 2, 3]])
    dist_mat, dist_max, dist_min = distance_matrix(0, coords)
    expected_matrix = np.array([[0]])
    assert np.allclose(dist_mat, expected_matrix)
    assert dist_max == 0
    assert dist_min == 0

### Testing adjacency #####

def test_two_atoms():
    coords = np.array([[0, 0, 0], [2, 0, 0]])
    adj_1 = adjacency_matrix(1, coords, cutoff=2)
    assert adj_1[1,0] == 1
    adj_2 = adjacency_matrix(1, coords, cutoff=1)
    assert adj_2[1,0] == 0
