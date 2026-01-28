from typing import Tuple
import numpy as np
import os
import inspect

def read_rgl(filepot: str) -> dict:
    """Reads parameter for the RGL potential from a file

    Parameters
    ----------
    filepot : str
        _description_

    Returns
    -------
    dict
        _description_
    """
    with open(filepot, 'r') as file:
        lines = file.readlines()
    
    # Preprocess lines to replace 'd' with 'e' for compatibility with Python floats
    lines = [line.replace('d', 'e').replace('D', 'E') for line in lines]
    
    #get the two elements from the file
    element_a, element_b = lines[2].split()
    
    #potential parameters
    p_line = lines[5].split()
    p_a = float(p_line[0])
    p_b = float(p_line[1])
    p_ab = float(p_line[2])
    
    q_line = lines[6].split()
    q_a = float(q_line[0])
    q_b = float(q_line[1])
    q_ab = float(q_line[2])
    
    a_line = lines[7].split()
    a_a = float(a_line[0])
    a_b = float(a_line[1])
    a_ab = float(a_line[2])
        
    qsi_line = lines[8].split()
    qsi_a = float(qsi_line[0])
    qsi_b = float(qsi_line[1])
    qsi_ab = float(qsi_line[2])
    
    #metal properties
    coh_energies = lines[11].split()[0:2]
    coh_energy_a = float(coh_energies[0])
    coh_energy_b = float(coh_energies[1])
    
    
    atom_radiuses = lines[12].split()[0:2]
    atom_radius_a = float(atom_radiuses[0])
    atom_radius_b = float(atom_radiuses[1])
    
    masses = lines[13].split()[0:2]
    mass_a = float(masses[0])
    mass_b = float(masses[1])
    
    #cut off 
    cutoffs = lines[16].split()[0:2]
    cut_start = float(cutoffs[0])
    cut_end = float(cutoffs[1])   
        
    #bulk nearest enighbour distances
    dist = np.array([atom_radius_a * np.sqrt(8.0), 
                      atom_radius_b * np.sqrt(8.0) ,
                      (atom_radius_a + atom_radius_b) * np.sqrt(2.0)])
    #organizing params in array for iteration 
    a = np.array([a_a, a_b, a_ab])
    p = np.array([p_a, p_b, p_ab])
    q = np.array([q_a, q_b, q_ab])
    qsi = np.array([qsi_a, qsi_b, qsi_ab])
    
    x3 = np.zeros(3)
    x4 = np.zeros(3)
    x5 = np.zeros(3)
    
    a3 = np.zeros(3)
    a4 = np.zeros(3)
    a5 = np.zeros(3)
    for i in range(3):
        d_ik_0 = dist[i]
        
        ar = -a[i] * np.exp(-p[i]* (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start ) ** 3
        br = -(p[i] / d_ik_0) * a[i] * np.exp(-p[i] * (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start) ** 2
        cr = -((p[i] / d_ik_0) ** 2) * a[i] * np.exp(-p[i] * (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start)

        ab = -qsi[i] * np.exp( -q[i] * (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start) ** 3
        bb= -(q[i] / d_ik_0) * qsi[i] * np.exp( -q[i] * (cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start) ** 2
        cb= -((q[i] / d_ik_0) ** 2) * qsi[i] * np.exp( -q[i]*(cut_start / d_ik_0 - 1.0)) / (cut_end - cut_start)
        
        x5[i] = (12.0 * ab - 6.0 * bb + cb) / (2.0 * (cut_end - cut_start) ** 2) 
        x4[i] = (15.0 * ab - 7.0 * bb + cb) / (cut_end - cut_start)
        x3[i] = (20.0 * ab - 8.0 * bb + cb) / 2.0
        
        a5[i] = (12.0 *ar - 6.0 * br + cr) / (2.0 * (cut_end - cut_start) ** 2)
        a4[i] = (15.0 *ar - 7.0 * br + cr) / (cut_end - cut_start)
        a3[i] = (20.0 *ar - 8.0 * br + cr) / 2.0
        
    return {
        "el_a": element_a,
        "el_b": element_b,
        
        "p": p,
        
        "q": q, 
        
        "a": a,
        
        "qsi": qsi,
        
        "coh_a": coh_energy_a,
        "coh_b": coh_energy_b,
        
        "r_a": atom_radius_a,
        "r_b": atom_radius_b,
        
        "m_a": mass_a,
        "m_b": mass_b,
        
        "cut_start": cut_start,
        "cut_end": cut_end,
        
        "dist": dist,
        
        "x3": x3,
        "x4": x4,
        "x5": x5,
        
        "a3": a3,
        "a4": a4,
        "a5": a5,
        
        
    }
       
        
            
        




