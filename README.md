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
el, coords = read_xyz(filename = "Au561_Ih.xyz")
```
since the structure only contains gold atoms (561 of them) we would get **el** as a list of 561 elements, all "Au" and coords a 3x561 array with the coordinates of the atoms.

If, on the other hand, we had a file named, for instance "Au561_md_300k.xyz", containing let's say 100 snapshot of a molecular dynamics simulation for the same structure, we would call:
```python
from snow.lodispp.pp_io import read_xyz_movie
el, coords = read_xyz_movie(filename = "Au561_Ih.xyz")
```
now el would be the same as before, coords, on the other hand, would be a higher dimensional array (100x3x561), and we could isolate a single snapshot as, for instance
```python
coord_5 = coords[5]
```


### Neighbours Lists

SNOW includes the determination of neighbours lists, it does so efficiently exploiting [KDTrees](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) as implemented in the popular scipy python package, this neighbours lists can be extracted by the user for further analysis and are used for the computation of most descriptors such as the coordination and the generalized coordination numbers, the identification of bridge sites etc.

With SNOW you can obtain the neighbour list as a list of neighbours for each atom, the i-th entry of the list is itself a list containing the indeces of the atoms neighbouring the i-th atom within a given cut-off. In SNOW we can obtain said list with: 
```python
from snow.lodispp.utils import nearest_neighbours
neigh_list = nearest_neighbours(index_frame = 1, coords = sys_coords , cut_off = 2.89)
```

### Cordination Number

### Generalized Coordination Number

### Strain

### Steinhardt Parameters