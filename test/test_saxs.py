from snow.descriptors.distributions import pddf_calculator,pddf_calculator_by_elements
from snow.io import read_xyz_movie
from snow.descriptors.utils import distance_matrix
from snow.descriptors.saxs import *

elements,coordinates = read_xyz_movie("test/AlAu_ih13.xyz")

#Tests that distance matrix and pddf method give the same SAXS spectra up to RATIO_TOLERANCE error

#We accept a discordance of +/- 5% between distance_matrix and PDDF method
RATIO_TOLERANCE=0.05

#Test using distance matrix
distmat =  distance_matrix(coordinates[0])
qs=np.logspace(-1,1,100)
i_qs = [iq_from_dist_mat(species = elements[0],q=q,dist_mat=distmat) for q in qs]
i_qs_distmat = np.array(i_qs)

#TODO: this could become a function
#Read pddf from file
alal_pddf=np.empty(0)
alau_pddf=np.empty(0)
auau_pddf=np.empty(0)
bincenters=np.empty(0)
lines= open("test/pddf_AlAu_ih13_bins_0.05.dat",'r').readlines()
for l in lines:
    if l.startswith("#"):
        continue
    newbin,newalal,newalau,_= [float(x) for x in l.strip().split()]
    bincenters = np.hstack((bincenters,newbin))
    alal_pddf  = np.hstack((alal_pddf,newalal))
    alau_pddf  = np.hstack((alau_pddf,newalau))
    auau_pddf  = np.hstack((auau_pddf,0.0)) #fictitious auau PDDF

i_qs_pddf = np.empty(0)
for q in qs:
    iq=0
    iq += iq_from_pddf("Al","Al",q,bincenters,alal_pddf,nat=12)
    iq += iq_from_pddf("Au","Au",q,bincenters,auau_pddf,nat=1)
    iq += iq_from_pddf("Al","Au",q,bincenters,alau_pddf)
    i_qs_pddf = np.hstack((i_qs_pddf,iq))

error_ratio = np.abs(i_qs_pddf/i_qs_distmat - 1)
assert max(error_ratio) < RATIO_TOLERANCE , f"Results between distance matrix method and PDDF method disagree worse than {RATIO_TOLERANCE}"
