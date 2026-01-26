# Module to compute Small Angle X-Ray Scattering spectra of nanoparticles
# using either their atomic coordinates or their PDDF

# date: Jan 15th, 2026
# author: davide.alimonti@unimi.it

import numpy as np
from snow.misc.constants import cm_coeffs
from snow.descriptors.utils import distance_matrix

def thomson(element: str ,q : float):
    """
    Computes the Thomson part of the atomic form factor at given exchanged
    momentum using the Cromer-Mann approximation
        Parameters
    ----------
    element : str
        Name of the chemical species
    q : float
        Magnitude of exchanged momentum

    Returns
    -------
    f : float 
        Atomic form factor for the element at given exchanged momentum
    """
    coeffs=cm_coeffs[element]
    f = coeffs['c']
    for a,b in zip(coeffs["as"],coeffs["bs"]):
        f += a*np.exp(-b*(q/4/np.pi)**2)
    return f

def iq_from_dist_mat(element_i: str,element_j : str,q : float ,dist_mat : np.ndarray):
    """
    Computes the SAXS spectrum for a set of points from its distance matrix.
        Parameters
    ----------
    element_i : str
        Name of the chemical species of atoms i
    element_j : str
        Name of the chemical species of atoms j
    q : float
        Magnitude of exchanged momentum
    dist_mat : float
        Matrix of distances of the atoms as computed in snow.utils.distance_matrix

    Returns
    -------
    iq : float 
        Intensity of SAXS for the considered system, at given exchanged momentum
    """
    fi = thomson(element_i,q)
    fj = fi if element_j == element_i else thomson(element_j)
    intensity = 0.0
    ds = dist_mat.flatten()
    for d in ds:
        intensity += np.sinc(d*q/np.pi)
    iq = intensity * fi*fj
    return iq

def iq_pddf(element_i: str,element_j : str,q :float,
            dists:list ,counts:list ,nat:int =0):
    """
    Computes thes SAXS spectrum from a PDDF.
    Carfeul: if the bin at distance 0 is not included, nat must be
    manually set, otherwise set at 0
        Parameters
    ----------
    element_i : str
        Name of the chemical species of atoms i
    element_j : str
        Name of the chemical species of atoms j
    nat : int
        Number of atoms (only set if element_i == element_j)
    q : float
        Magnitude of exchanged momentum
    dists : list[float]
        Central values of the bins of the PDDF
    counts: list[float]
        Values of PDDF at the bins

    Returns
    -------
    intensity : float 
        Intensity of SAXS for the considered system, at given exchanged momentum

    """
    fi = thomson(element_i,q)
    fj = fi if element_j == element_i else thomson(element_j)
    intensity = nat
    for dist,count in zip(dists,counts):
        intensity += count * np.sinc(q*dist/np.pi)
    intensity *=  fi*fj
    intensity += nat * f**2
    return intensity
