from snow.descriptors.shape_descriptors import compute_gyration_descriptors
from snow.io import read_xyz
import numpy as np
def test_ih_100_spherical():
    """
    Test that the gyration tensor descriptors for a highly spherical cluster are all zero.
    """
    el, pos = read_xyz("ih_40_test.xyz")  # very spherical
    rg, b, c, k = compute_gyration_descriptors(pos)
    assert np.allclose([b, c, k], [0, 0, 0])


def test_dh_100_flat():
    """
    Test the descriptors for a slightly elongated/dh cluster.
    """
    el, pos = read_xyz("dh_100_test.xyz")
    rg, b, c, k = compute_gyration_descriptors(pos)
git rm --cached test/ih_100_test.xyz
    assert 1 - k < 0.1


def test_cube_translation_invariance():
    """
    Test that the descriptors are invariant under translation.
    """
    el, pos = read_xyz("cube_test.xyz")
    rg, b1, c1, k1 = compute_gyration_descriptors(pos)

    # apply a large translation
    pos += 1000 * np.ones(pos.shape)
    rg, b2, c2, k2 = compute_gyration_descriptors(pos)

    assert np.allclose([b1, c1, k1], [b2, c2, k2])

    #would be nice to find some non-cylindrical (and non prismical) clsuter to test acylindricity>>0
    #possibly add checks that b, c>0 and 0<k<1 always
    #something like
    # eps = 1e-12
    # assert b >= -eps
    # assert c >= -eps
    # assert -eps <= k <= 1. + eps