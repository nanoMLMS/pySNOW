from snow.descriptors.shape_descriptors import compute_gyration_descriptors
import numpy as np
def test_shape_descriptors():
    """
    Tests the calculation of gyration tensor derived shape descriptors. Does not test the aspect ratio yet
    """
    from ase.cluster import Decahedron as dh, Icosahedron as ih, Octahedron as oh
    from ase.build import bulk

    pos = ih('Cu', 100).get_positions()    #this should be very spherical
    b, c, k = compute_gyration_descriptors(pos)

    assert(np.allclose( np.array([b,c,k]), np.array([0,0,0])))

    pos = dh('Cu', 3, 100, 0).get_positions() #this should be very cylindrical and elongated
    b, c, k = compute_gyration_descriptors(pos)

    assert(c<1e-10)
    assert(1-k<0.1)

    #test translations
    at = bulk('Cu', 'fcc', 3.6)
    cube = at.repeat([10,10,10])
    pos = cube.get_positions()
    b1, c1, k1 = compute_gyration_descriptors(pos)
    pos+= 1000*np.ones(pos.shape)
    b2, c2, k2 = compute_gyration_descriptors(pos)
    assert(np.allclose( np.array([b1,c1,k1]), np.array([b2,c2,k2]) ))

    #would be nice to find some non-cylindrical (and non prismical) clsuter to test acylindricity>>0
    #possibly add checks that b, c>0 and 0<k<1 always
    #something like
    # eps = 1e-12
    # assert b >= -eps
    # assert c >= -eps
    # assert -eps <= k <= 1. + eps