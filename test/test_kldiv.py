from snow.descriptors.utils import kl_div
import numpy as np

dist1= np.array([1,2,3,4,5,6,7,8,9,19],dtype=float)
dist2= dist1*2

assert kl_div(dist1,dist1) == 0.0
assert kl_div(dist2,dist2) == 0.0
assert kl_div(dist1,dist2) == np.sum(dist1)*np.log(0.5)
