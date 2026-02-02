from snow.descriptors.coordination import coordination_number, agcn_calculator
import numpy as np
from snow.io import read_xyz
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

import os

def test_fcc_bulk():
    el, coords = read_xyz("cube_test.xyz")
    assert coordination_number(1, coords=coords , cut_off = 3.0)[142] == 12

"""
def test_bcc():
    el, coords = read_xyz("bcccell.xyz")
    assert coordination_number(1, coords=coords , cut_off = 1.83)[4] == 8

def test_cn_111():
    el, coords = read_xyz("cut_fcc.xyz")
    assert coordination_number(1, coords=coords , cut_off = 1.83)[25] == 9

def test_gcn_111():
    el, coords = read_xyz("cut_fcc.xyz")
    assert agcn_calculator(1, coords=coords , cut_off = 1.83, gcn_max=12.0)[25] == 7.5 """