# 23/03/2026

## Snow descriptors
- Added support for periodic boundary conditions (PBC).

## PDDF
- Refactored PDDF-by-element calculations:
  - the function now requires two elements as input,
  - homo-PDDF calculations can be obtained by passing the same element twice,
  - hetero-PDDF calculations can be obtained by passing two different elements.
- Merged `chemical_pddf` and `hetero_pddf` into a single unified function.
- Removed the standalone `chemical_pddf` and `hetero_pddf` functions.

## Bug fixes
- Fixed several bugs, mainly affecting SAXS and PDDF.

## io.xyz
- Added support for reading XYZ movies.
- Added support for trajectories with a varying number of atoms per configuration.

# 30/04/2026

## add_molecule
- Refactored `add_molecule`.
- Added support for placing arbitrary molecules with custom orientations and angles on selected adsorption sites.
- Added functions to compute locally normal surface directions for different use cases.

## Documentation and maintenance
- General bug fixes.
- Expanded documentation.

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

## read_xyz
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
