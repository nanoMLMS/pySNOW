from snow.lodispp.utils import coordination_number
import numpy as np

def test_cn_isolated():
    coord = np.array([[0,0,0]])
    el = np.array(["Au"])
    
    assert coordination_number(1, coord,2.98) == 0
    
def test_cn_twoatoms():
    coords = np.array([
        [0, 1, 0],
        [0, -1, 0]
    ])
    
    els = np.array(["Au", "Au"])
    assert coordination_number(1, coords=coords, cut_off = 3)[0] == 1
    assert coordination_number(1, coords=coords, cut_off = 1.5)[1] == 0

