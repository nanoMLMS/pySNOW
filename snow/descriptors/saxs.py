# Module to compute Small Angle X-Ray Scattering spectra of nanoparticles
# using either their atomic coordinates or their PDDF

# date: Jan 5th, 2026
# author: davide.alimonti@unimi.it

import numpy as np
from snow.misc.constants import cm_coeffs
from snow.utils import distance_matrix

cm_coeffs={
        "Al" : {
                "as" : [6.4202, 1.9002, 1.5936, 1.9646 ] ,
                "bs" : [3.0387, 0.7426, 31.5472 ],
                "c" : 1.1151,
                },
        "Au" : {
                "as" : [16.8819, 18.5913, 25.5582, 5.86 ] ,
                "bs" : [0.4611, 8.6216, 1.4826 ],
                "c" : 12.0658
                }
         }

def thomson(element,q):
    """
    Computes the Thomson part of the atomic form factor at given exchanged
    momentum using the Cromer-Mann approximation
    """
    coeffs=cm_coeffs[element]
    f = coeffs['c']
    for a,b in zip(coeffs["as"],coeffs["bs"]):
        f += a*np.exp(-b*(q/4/np.pi)**2)
    return f

def iq_from_dist_mat(element_i,element_j,q,dist_mat):
    """
    Computes the SAXS spectrum from a distance matrix.
    The distance matrix needs to be computed beforehand!
    """
    fi = thomson(element_i,q)
    fj = fi if element_j == element_i else thomson(element_j)
    intensity = 0.0
    ds = dist_mat.flatten()
    for d in ds:
        intensity += np.sinc(d*q/np.pi)
    return intensity * fi*fj

def iq_pddf(element_i,element_j,nat=0,q,dists,counts,dr):
    """
    Computes thes SAXS spectrum from a PDDF.
    Carfeul: if the bin at distance 0 is not included, nat must be
    manually set, otherwise set at 0
    """
    fi = thomson(element_i,q)
    fj = fi if element_j == element_i else thomson(element_j)
    intensity = nat
    for dist,count in zip(dists,counts):
        intensity += count * np.sinc(q*dist/np.pi)
    intensity *=  fi*fj
    #intensity += nat * f**2
    return intensity
