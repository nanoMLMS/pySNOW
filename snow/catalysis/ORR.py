import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist
from snow.io.xyz import read_xyz
from snow.descriptors.coordination import agcn_calculator, nearest_neighbours

# --- COSTANTI FISICHE ---
ATOMIC_R = 1.385e-10          # m
MASS_PT = 3.2394457e-19       # mg
KB_T = 8.6173303e-5 * 298     # eV
V_REVERSIBLE = 1.23           # V (Potenziale teorico ORR)



def get_geometry_properties(coords):
    natoms = len(coords)
    rij_max = np.max(pdist(coords))
    return {"natoms": natoms, "rij_max": rij_max}

def get_physical_and_surface_props(agcns, num_nn, natoms):
    mass_np_mg = MASS_PT * natoms
    surf_mask = (num_nn < 11)
    counter_surf = np.sum(surf_mask)
    
    # Area Attiva (m^2)
    active_surf_m2 = np.sum(4 * np.pi * (ATOMIC_R**2) * ((12.0 - agcns[surf_mask]) / 12.0))
    
    return {
        "mass_np_mg": mass_np_mg,
        "active_surf_m2": active_surf_m2,
        "active_surf_cm2": active_surf_m2 * 1e4,
        "counter_surf": counter_surf,
        "surf_mask": surf_mask
    }

def calculate_volcano_dg_fortran(gcn_norm):
    """Calcola DG usando il GCN normalizzato (0-1) come da riga 243 Fortran."""
    return np.where(gcn_norm * 12.0 < 8.33,
                    0.192 * (gcn_norm * 12.0) - 0.724,
                    -0.178 * (gcn_norm * 12.0) + 2.345)

def perform_activity_analysis(unique_gcns, occurrences, phys_props):
    """Simulazione IV fedele alle righe 260-300 del Fortran."""
    u_bins = np.arange(1, 1299) * 0.001 # j*0.001
    total_current = np.zeros_like(u_bins)

    gcn_norm = unique_gcns / 12.0
    
    dg_values = np.where(unique_gcns < 8.33,
                         0.192 * unique_gcns - 0.724,
                         -0.178 * unique_gcns + 2.345)
    
    freq = occurrences / np.sum(occurrences)
    
    for j_idx, u in enumerate(u_bins):

        delta = (dg_values - u) / KB_T
        j_lim = unique_gcns * 12.65
        
        site_currents = np.exp(delta) * freq * j_lim
        total_current[j_idx] = np.sum(site_currents)
    
    # SA a 0.9V (indice 899 perché u = 900 * 0.001)
    sa_09 = total_current[899]
    
    ma_09 = sa_09 * (phys_props['active_surf_m2'] * 10.0) / phys_props['mass_np_mg']
    
    eta1 = 0.0
    for j_idx in range(len(u_bins)-1, -1, -1):
        if total_current[j_idx] > 1.0:
            eta1 = V_REVERSIBLE - u_bins[j_idx]
            break
            
    return sa_09, ma_09, eta1
    
    
    

file_xyz = '../traj.xyz'
el, coords = read_xyz(file_xyz)

cutoff = 1.2e10 * 2.0 * ATOMIC_R
sites, agcns = agcn_calculator(coords, cutoff, thr_gcn=11)
neigh_list = nearest_neighbours(coords=coords, cut_off=cutoff)
num_nn = np.array([len(n) for n in neigh_list])

geo = get_geometry_properties(coords)
phys = get_physical_and_surface_props(agcns, num_nn, geo['natoms'])

counts = Counter(np.round(agcns[phys['surf_mask']], 3))
unique_gcns = np.array(list(counts.keys()))
occurrences = np.array(list(counts.values()))

dg_list = np.where(unique_gcns < 8.33, 0.192 * unique_gcns - 0.724, -0.178 * unique_gcns + 2.345)
sa_09, ma_09, eta_val = perform_activity_analysis(unique_gcns, occurrences, phys)


print("\n" + "="*45)
print("   ANALISI ELETTROCATALITICA COMPLETA")
print("="*45)
print(f"N. Atomi:           {geo['natoms']}")
print(f"D [nm]:             {geo['rij_max']/10.0:.4f}")
print(f"Massa [mg]:         {phys['mass_np_mg']:.4e}")
print(f"Agcn [cm2]:         {phys['active_surf_cm2']:.4e}")
print("-" * 45)
print(f"eta1 [V]:           {eta_val:.4f}")
print(f"SA* [mA/cm2]:       {sa_09:.4f}")
print(f"MA* [A/mg]:         {ma_09:.4f}")
print("="*45)



import numpy as np
from collections import Counter

# ... (il resto del tuo codice precedente: definizioni funzioni e calcoli) ...

# 1. Preparazione dati per gcn_genome.dat
# Creiamo un dizionario per mappare i GCN calcolati sulle occorrenze
# Arrotondiamo a 2 decimali come nel tuo file sorgente
gcn_range = np.arange(0.0, 12.3, 0.1)
occ_map = {np.round(k, 2): v for k, v in counts.items()}

with open('gcn_genome.dat', 'w') as f:
    f.write("# frame num., gcn value, gcn occurrance, number atoms, number surface atoms, diameter [nm]\n")
    for gval in gcn_range:
        val = np.round(gval, 2)
        count = occ_map.get(val, 0)
        # Usiamo i dati estratti da geo e phys
        line = f"1 {val:.2f} {count} {geo['natoms']} {phys['n_surf']} {geo['rij_max']/10.0:.15f}\n"
        f.write(line)

# 2. Preparazione dati per eta.dat
with open('eta.dat', 'w') as f:
    f.write("# natoms \t diameter [nm] \t gcn area [m2] \t gcn area [Ang 2] \t mass NP [mg] \t eta1 [V] \t SA@0.9V (mA/cm2) \t MA@0.9V (mA/mg) (gcn area based)\n")
    
    # Trasformiamo l'area da cm2 (usata nel print) a m2 e Angstrom^2 per il file
    area_m2 = phys['active_surf_cm2'] / 1e4
    area_ang2 = phys['active_surf_cm2'] * 1e16
    
    line = (f"{geo['natoms']}\t"
            f"{geo['rij_max']/10.0:.16f}\t"
            f"{area_m2:.18e}\t"
            f"{area_ang2:.16f}\t"
            f"{phys['mass_np_mg']:.6e}\t"
            f"{eta_val:.16f}\t"
            f"{sa_09:.16f} \t "
            f"{ma_09:.16f}\n")
    f.write(line)

print("\nFile 'gcn_genome.dat' e 'eta.dat' salvati correttamente.")


# 3. Salvataggio gcn_pure.dat (Occorrenze senza binning/arrotondamento)
# Recuperiamo i GCN solo per gli atomi di superficie
surf_gcns = agcns[phys['surf_mask']]

# Usiamo Counter sui valori originali (non arrotondati)
pure_counts = Counter(surf_gcns)

# Ordiniamo i GCN dal più piccolo al più grande
sorted_pure_gcns = sorted(pure_counts.items())

with open('gcn_pure.dat', 'w') as f:
    f.write("# GCN_value (pure) \t Occurrence\n")
    for gcn_val, count in sorted_pure_gcns:
        # Scriviamo il valore puro con alta precisione e la sua occorrenza
        f.write(f"{gcn_val:.15f}\t{count}\n")

print("File 'gcn_pure.dat' salvato con le occorrenze esatte.")
