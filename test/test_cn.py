from snow.lodispp.utils import coordination_number
import numpy as np

def test_cn_isolated():
    coord = np.array([0,0,0])
    el = "Au"
    
    assert coordination_number(1, coord,2.98)