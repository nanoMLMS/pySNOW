23/03/2026

 - Snow descriptors now work with pbc
 - Modified PDDF by element: now you have to pass two elements (provide same element for homo pddf , different elements for hetero-pddf) - chemical pddf and hetero_pddf are now wrapped in this unique function, so they will not exist anymore.
 - fixed a few bugs mainly in saxs and pddf.
 - io.xyz functions are now able to deal with movies, even with a varying number of atoms per configuration (e.g., for databases with configurations from different systems)

30/04/2026 and running

 - refactoring of add_molecule, now any kind of molecule can be added with any kind of direction and angle on a given site. A few functions return the 'locally normal' direction wrt to the surface in a given site for different use cases.
 - usual bug fixing and docs writing


