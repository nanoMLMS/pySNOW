import numpy as np
from snow.descriptors.distributions import pddf_calculator, pddf_calculator_by_elements
from snow.io import read_xyz_movie

BIN_WIDTH=0.05

elements,coordinates = read_xyz_movie("test/AlAu_ih13.xyz")
bincenters,total_pddf = pddf_calculator(coordinates[0],bin_width=BIN_WIDTH,use_lattice_units=False)

_,alal_pddf= pddf_calculator_by_elements(elements[0],coordinates[0],["Al","Al"],bin_width=BIN_WIDTH,use_lattice_units=False)
_,alau_pddf = pddf_calculator_by_elements(elements[0],coordinates[0],["Al","Au"],bin_width=BIN_WIDTH,use_lattice_units=False,cutoff=5.7)


#Sanity check : the sum of all chemical PDDFs MUST be the total PDDF

assert np.all(alal_pddf+alau_pddf == total_pddf) , "Sum of chemical PDDFs != total PDDF"

#TODO Assert that the result is the same as in the PDDF file here
