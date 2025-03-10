# SNOW

SNOW is a mostly standalone (only a few required dependencies: numpy and scipy) characterization tool for trajectories generated during MD simulations

## Features

SNOW allows for the computation of a number of properties and descriptors for atomistic structures, ranging from the coordination number to more complex parameters such as Steinhardt's parameters.

### Input - Output

The structures can be obtained using external modules, such as [ASE](https://wiki.fysik.dtu.dk/ase/index.html) or read fromthe popular xyz format or from LAMMPS data files using the funtion provided by SNOW. 

If you use SNOW, the functions will genrally provide a tuple: a list of elements for each of the atoms in the system and a 3d array containg the coordinates of each atom. This will have an extra dimension if you read a trajectory consisting of multiple snapshot as generated for instance from an MD simulation.

As an example, if we want to read the coordinates from a file named "Au561_Ih.xyz", containing a single snaplshot, we would call:
```python
from snow.lodispp.pp_io import read_xyz
el, sys_coords = read_xyz(filename = "Au561_Ih.xyz")
```
since the structure only contains gold atoms (561 of them) we would get **el** as a list of 561 elements, all "Au" and coords a 3x561 array with the coordinates of the atoms.

If, on the other hand, we had a file named, for instance "Au561_md_300k.xyz", containing let's say 100 snapshot of a molecular dynamics simulation for the same structure, we would call:
```python
from snow.lodispp.pp_io import read_xyz_movie
el, sys_coords = read_xyz_movie(filename = "Au561_Ih.xyz")
```
now el would be the same as before, coords, on the other hand, would be a higher dimensional array (100x3x561), and we could isolate a single snapshot as, for instance
```python
coord_5 = sys_coords[5]
```

Structures can also be written to an xyz file, with any additional data, using the write_xyz function. It is necessary to provide a path to the file where the structure will be stored, the array with the elements and the array with the coordinates, as an example (some of the calculated features will be explained below):
```python
from snow.lodispp.pp_io import read_xyz, write_xyz
elements, coords = read_xyz("Au561_Ih.xyz")

# Define parameters
cutoff_radius = 4.079 * 0.95  # Cutoff radius for neighbor detection
dist_0 = 1.66 * 2       # Reference distance for strain calculation

# Compute coordination number, strain, and other properties
cn = coordination_number(1, coords=coords, cut_off=cutoff_radius)
strain_syst = strain_mono(index_frame=1, coords=coords, dist_0=dist_0, cut_off=cutoff_radius)
fnn = nearest_neighbours(1, coords, cutoff_radius)
snn = second_neighbours(1, coords, cutoff_radius)
agcn = agcn_calculator(index_frame=1, coords=coords, cutoff=cutoff_radius, gcn_max=12)

# Determine surface atoms
is_surface = agcn < 10.0

# Compute Steinhardt parameters
stein = peratom_steinhardt(1, coords, [4, 6, 8, 12], cutoff_radius)

# Prepare additional data
additional_data = np.column_stack((cn, agcn, strain_syst, is_surface, *stein))
print("q12_avg = {:.3f} +/- {:.3f}".format(np.mean(stein[3]), np.std(stein[3])))

# Write updated XYZ file
write_xyz("output_test.xyz", elements=elements, coords=coords, additional_data=additional_data)
```

will write the coordinates, elements and all other computed parameters and descriptors to an xyz file *output_test.xyz* which can be then opened, for instance, with [ovito](https://www.ovito.org/) for a visual representation using some form of [color coding](https://www.ovito.org/manual/reference/pipelines/modifiers/color_coding.html)
### Neighbours Lists

SNOW includes the determination of neighbours lists, it does so efficiently exploiting [KDTrees](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) as implemented in the popular scipy python package, this neighbours lists can be extracted by the user for further analysis and are used for the computation of most descriptors such as the coordination and the generalized coordination numbers, the identification of bridge sites etc.

The neighbours are defined, for each atom, as the atoms lying within a sphere of a given radius known as the **cut off radius**.

With SNOW you can obtain the neighbour list as a list of neighbours for each atom, the i-th entry of the list is itself a list containing the indeces of the atoms neighbouring the i-th atom within a given cut-off. In SNOW we can obtain said list with: 
```python
from snow.lodispp.utils import nearest_neighbours
neigh_list = nearest_neighbours(index_frame = 1, coords = sys_coords , cut_off = 2.89)
```

### Cordination Number

The coordination number is one of the simplest descriptors of the environment of each atom, it is simply the number of neighbouring atoms for each atom in the system. It is a first indicator of a possible crystal structure, for instance it is equal to 12 in FCC crystals, 8 in BCC crystals, 9 on 111 surfaces of FCC crystals and so on.

When computed by SNOW it returns a one dimensional array of coordination numbers for each atom with the same ordering as in the coordinates array.
```python
from snow.lodispp.utils import coordination_number
cn = coordination_number(index_frame = 1, coords = sys_coords, cut_off = 2.89, neigh_list = False)
```
The neigh_list flag (set to false by default) can be switched to true to also output the neighbour list:
```python
from snow.lodispp.utils import coordination_number
neigh_list, cn = coordination_number(index_frame = 1, coords = sys_coords, cut_off = 2.89, neigh_list = True)
```

### Generalized Coordination Number

Introduce by [Federico Calle-Vallejo](https://doi.org/10.1002/advs.202207644) the generalized coordination number (GCN) is a more refined descriptors for the environment of atoms, taking into account, for each atom, the neighbours of the neighbouring atoms. It is defined as a weighted average (weighted by a maximum coordination number associated to the refernece bulk structure, so, if the system is composed of gold atoms the crystal reference is FCC and CN<sub>max</sub>=12) of the coordination numbers of the neighbours of each atom.

What has been described is referred to **atop** GCN (a-GCN), to differentiate with the **bridge** GCN (b-gcn) which computes the same weighted average for bridge sites (identified by a pair of nieghbouring atoms and counting the CN of neihgbours for each atom counted only once) and the **hollow** GCN (for triplets and fourplets, still to implement).

Note that while the a-GCN function returns a list of values for each atom, the b-gcn returns values for each pair, for output and representation purposes is possible to obtain also the coordinates of the midpoints of each pair so that a "phantom" atom can be written to an xyz file to represent the GCN for those bridge sites.

```python
from snow.descriptors.gcn import agcn_calculator, bridge_gcn
agcn = agcn_calculator(index_frame = 1, coords= sys_coords, cut_off = 2.89, gcn_max = 12.0)
phant_xyz, pairs, bgcn = bridge_gcn(index_frame = 1, coords = sys_coords, cut_off = 2.89, gcn_max=18.0, phantom=True)

for p in pairs:
    elements.append("H")

# Combine AGCN and phantom GCN
a_b_gcn = np.concatenate((agcn, b_gcn))
a_b_gcn = a_b_gcn.reshape(-1, 1)  # Shape will be (843, 1)

# Concatenate coordinates with phantom coordinates
out_xyz = np.concatenate((sys_coords, phant_xyz))    

# Write updated XYZ file with phantom data
write_xyz("phantom.xyz", elements, out_xyz, additional_data=a_b_gcn)
```
in this example we compute both the atop and bridge gcn, generate coordinates for each pair midpoint, assign an H atom at those coordinates, concatenate it to the original coordinates and output everything (agcn for each atom and bgcn for the "phantom" hydrogen atoms at the midpoints to and xyz file).
### Strain

The strain, as computed by SNOW, is a measure of the deformation of interatomic separations compared with typical equilibrium separations in the bulk. It provides a percentage for each atom which will be positive (negative) if the interatomic distances with negihbours is larger (smaller) than what is typically observed in the bulk and thus the atom is on average in a expanded (compressed) state.

### Steinhardt Parameters

Steinhardt parameters are order parameters for atomistic structures 


### CNA Patterns

pySNOW is capable of performing a simple identification of what type of structure each atom belon based on the 
study of the number of singatures each atom partecipates to.

The following code is a simple CLI tool that allows users to input an xyz file with a given name and return an xyz file
having as an extra column an integer whihch cathegorize the atom based on the structure to which the atom belongs to:

```python
import argparse
import numpy as np
from snow.lodispp.cna import cnap_peratom
from snow.lodispp.pp_io import read_xyz, write_xyz

parser = argparse.ArgumentParser(
    prog="CNAPatterner",
    description="It patterns the CNA",
    epilog="Text at the bottom of help",
)

parser.add_argument("-i", "--infile", help="Input xyz file")
parser.add_argument("-o", "--outfile", help="To what xyz file write CNAp")

args = parser.parse_args()

infile = args.infile
outfile = args.outfile

el, coords = read_xyz(infile)


cna_atom = cnap_peratom(1, coords, 4.08 * 0.85)
write_xyz(outfile, el, coords, cna_atom.reshape(-1, 1))
```
#### **Usage**  
```bash
python cnapatterner.py -i input.xyz -o output.xyz
```

#### **CNAp Index Descriptions**  

| **CNAp** | **Description**                                      | **CNAp composition** |
|:--------:|------------------------------------------------------|:--------:|
| 1        | Vertex between two (111) facets and a (100) facet    |[(1, (100)), (2, (211)), (1, (322)), (1, (422))] |
| 2        | Edge between (100) and a slightly distorted (111)    |[(1, (200)), (2, (211)), (2, (311)), (1, (421))] |
| 3        | Atoms lying on a (555) symmetry axis                 |[(10, (422)), (2, (555))] |
| 4        | FCC bulk                                             |[(12, (421))] |
| 5        | Intersection of six five-fold axes                   |[(12, (555))] |
| 6        | Edge between (100) facets                            |[(2, (100)), (2, (211)), (2, (422))] |
| 7        | Vertex on twinning planes shared by (111) facets     |[(2, (200)), (1, (300)), (2, (311)), (1, (322)), (1, (422))] |
| 8        | Edge between (111) re-entrances and (111) facets     |[(2, (200)), (4, (311)), (1, (421))] |
| 9        | Re-entrance delimited by (111) facets                |[(2, (300)), (4, (311)), (2, (421)), (2, (422))] |
| 10       | Edge between (100) and (111) facets                  |[(3, (211)), (2, (311)), (2, (421))] |
| 11       | Vertex shared by (100) and (111) facets              |[(4, (211)), (1, (421))] |
| 12       | (100) facet                                          |[(4, (211)), (4, (421))] |
| 13       | Five-fold symmetry axis (without center)             |[(4, (311)), (2, (322)), (2, (422))] |
| 14       | Five-fold vertex                                     |[(5, (322)), (1, (555))] |
| 15       | (111) face                                           |[(6, (311)), (3, (421))] |
| 16       | Twinning plane                                       |[(6, (421)), (6, (422))] |

<div align="center">
  <img src="https://github.com/user-attachments/assets/46d18b9e-3829-4cf1-82c4-a8204f4da237" width="60%">
</div>
