# 23/03/2026

 - Snow descriptors now work with pbc
 - Modified PDDF by element: now you have to pass two elements (provide same element for homo pddf , different elements for hetero-pddf) - chemical pddf and hetero_pddf are now wrapped in this unique function, so they will not exist anymore.
 - fixed a few bugs mainly in saxs and pddf.
 - io.xyz functions are now able to deal with movies, even with a varying number of atoms per configuration (e.g., for databases with configurations from different systems)

# 30/04/2026

 - refactoring of add_molecule, now any kind of molecule can be added with any kind of direction and angle on a given site. A few functions return the 'locally normal' direction wrt to the surface in a given site for different use cases.
 - usual bug fixing and docs writing

# 14/05/2026

## coordination.py
- Standardized phantom atom handling across all functions.
- Unified the default format for returned sites.

## distributions.py
- Standardized argument ordering to `(element, coordination)` across all functions.

## Package restructuring
- Removed the `build` package.
- Moved `add_molecule.py` to the `catalysis` package.
- Moved `rototranslation.py` from `build` to `misc`.
- Merged the `lbl.py` subpackage into `distributions.py`.

## New adsorption tools (`catalysis/add_molecule.py`)
- Added support for placing adsorbed atoms and molecules on user-selected sites defined by XYZ coordinates.
- Added molecular orientation control through azimuthal and polar angles.
- Added a function to cover a specified percentage of a nanoparticle surface with adsorbates.
- Expanded tutorials with new examples and usage instructions.

## New ORR module
- Added `Pt_ORR.py` to the `catalysis` package.
- Implemented tools to calculate:
  - current,
  - mass activity,
  - specific activity
  for Pt nanoparticles following the methodology of Rossi, Baletto et al.

## reax_xyz
- Added support for reading additional columns from input files.

## io.py
- Extended `read_ordered_LAMMPS_dump` with:
  - support for custom coordinate columns for non-standard dump styles,
  - optional reordering of dump files when atomic indices are available.

## utils
- Added a function to retrieve coordinates by element.

## Bug fixes
- Fixed a bug in the PDDF implementation.

## References
- Added literature references for the CNA and CNAP methods.

## Cleanup
- Removed all index frames.

## Documentation
- Expanded and added several tutorials.
