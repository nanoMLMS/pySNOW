---
title:  'pySNOW: a Python Suite for the NanO-World'
tags:
  - Python
  - atomistic systems
  - molecular dynamics
  - nanoparticles physics
  - computational material design
authors:
  - name: Sofia Zinzani
    orcid: 0009-0009-6961-7766
    equal-contrib: true
    affiliation: 1
    corresponding author: true
  - name: Gilberto Nardi
    orcid: 0009-0003-3276-8372
    equal-contrib: true
    affiliation: 1
    corresponding author: true
  - name: Giacomo Becatti
    orcid: 0009-0004-1686-7984
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Davide Alimonti
    orcid: 0009-0008-4530-1559
    affiliation: 1
  - name: Letícia F. Basso
    orcid: 0009-0002-5108-3101
    affiliation: 1
  - name: Kevin Rossi
    orcid: 0000-0001-8428-5127
    affiliation: 3
  - name: Francesca Baletto
    orcid: 0000-0003-1650-0010
    affiliation: 1
affiliations:
  - name: Department of Physics, University of Milan, Via Celoria 16, Milan, 20133, Italy
    index: 1
  - name: Scientific Computing Center, Karlsruhe Institute of Technology, Hermann-von-Helmholtz-Platz 1, 76344 Eggenstein-Leopoldshafen
    index: 2
  - name: Materials Science and Engineering Department, Delft University of Technology, Delft, 2623CD, Netherlands
    index: 3
date: 22 May 2026
bibliography: paper.bib
---



# Summary

[A description of the high-level functionality and purpose of the software for a diverse, non-specialist audience.]: #

In computational materials science, numerical simulations are indispensable tools for revealing atomic-scale processes. For building blocks of the nanoworld, such as nanoparticles and nanoalloys, atomistic simulations provide detailed insight into their behaviour under diverse conditions. These simulations enable the study of formation and growth mechanisms, transport phenomena, chemophysical stability, and chemical reactions, including catalytic processes.

To unravel the complex structure–property relationships that characterise nanoobjects, it is crucial to develop robust and insightful representations of their atomistic structure at global and local scales. Such descriptions enhance our understanding of nanoparticle behaviour and play a key role in guiding their rational design in silico for targeted applications. 

# Statement of need
`pySNOW` -a Python Suite for the NanO-World- offers a complete Python workflow designed for the morphological characterisation of metallic nanoparticles: from input/output operations, through extended characterisation, up to connection with computational catalysis calculations and experimentally observable quantities. The main focus of pySNOW is the morphological characterisation of mono- and bi-metallic atomistic systems, e.g., from classical and ab initio molecular dynamics, Monte Carlo simulations, or eventually 3D tomography reconstruction. To this end, it provides the user with a suite of tools and routines along with a full list of tutorials. The only requirement is the coordinates or the trajectory of the nanosystem under study, in a simple, human-readable, standard .xyz format. From such inputs,
`pySNOW` allows the computation of several common descriptors, widely used in computational nanomaterials science [@baletto2019structural], as well as more advanced observables that mimic the results of experimental techniques, such as SAXS spectra and chemical analyses.

Apart from morphological characterisation, it also enables catalytic investigation. `pySNOW` allows mapping the diversity of surface sites and adding one or more adsorbed molecules to them, providing input for further electronic structure calculations. Furthermore, it offers, within the CHE model [@Calle-Vallejo2014], [@Rossi2020], a microkinetic model for estimating specific current and mass activities during the oxygen reduction reaction (ORR) on Pt nanoparticles.

The motivation for developing the `pySNOW` package is the need for a unified and comprehensive toolkit for these analyses.  Thanks to its modular, functional and simple structure, it should be easily adopted by broad sections of both the computational and experimental materials science communities.

[FB: secondo me qui è da espandere con una frase, gli spettri SAXS e anche le chemical analysis che si interfacciano a EDX non sono standard.]: #


Apart from morphological characterisation, it also enables catalytic investigation. `pySNOW` allows mapping the diversity of surface sites and adding one or more asorbed structures to them, providing input for further electronic structure calculations. Furthermore, it offers, within the CHE model [@Calle-Vallejo2014], [@Rossi2020], a microkinetic model for estimating specific current and mass activities during the oxygen reduction reaction (ORR) on Pt nanoparticles.

The motivation for developing the `pySNOW` package is the need for a unified and comprehensive toolkit for these analyses.  Thanks to its modular, functional and simple structure, it should be easily adopted by broad sections of both the computational and experimental materials science communities.

[To decide: The latters can also benefit from the presence of calculation that can be compared with experimental observables. comparing experimental data with computed quantites of directly observable experimental variables]: #

[The goal of  pysnow is to provide a comprehensive and complete list of descriptors tailored for the full charaterization of metallic nanoparticles.]: #


[A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work.]: #


# State of the field     

Several tools exist to analyse atomistic simulation results and MD trajectories. Here, we provide a quick overview of the main packages the authors are familiar with, along with their differences from pySNOW. MDTraj [@MDtraj] enables fast calculations of root-mean-square displacement (RMSD) and the extraction of common order parameters. Together with MDAnalysis [@MDAnalysis_gowers2016] [@MDAnalysis_Agrawal2011], these packages are more oriented towards biomolecular systems. `pySCAL` [@pySCAL] is a Python module for the calculation of descriptors of local atomic environments, e.g. Steinhardt’s parameters, and with the possibility of adaptive cutoffs.
Freud [@freud:2020] is a capable and efficient Python/C++ tool that enables the computation of a wide range of quantities, including correlation functions and diffraction patterns.
ASE [@ase-paper]: is a general library capable of building and modifying structures and directly performing calculations. Along with these tools, it provides several post-processing tools, e.g. to obtain derived quantities such as phonon curves and vibration spectra, diffusion coefficients, or fitting equations of state.

While other packages provide characterisation tools, `pySNOW` covers a broad and extensive range of quantities. Being tailored for nanoparticles and bimetallic nanoalloys, it pays close attention to element-wise chemical analysis and also provides relevant experimentally accessible observables.

[- While we are not only local and have more descriptors.]: #
[eventualmente aggiungere pymatgen
While other packages provide characterization tools, `pySNOW` covers a broad and extensive range of quantities. Being tailored for nanoparticles and bimetallic nanoalloys, it comes with attention with respect to element-wise chemical analysis, and also provides relevant experimentally accessible observables.]: #


# Software design

`pySNOW` has been developed with a few key design principles in mind: namely, (1) ease of use - to enable fast learning for a wide range of scientists with different backgrounds, (2) ease of modification, (3) independence from many complex packages, with the only required depdendencies being numpy and scipy, in order to maintain a high staibility over time and the reduce possible conflicts for future versions, (4) relying exclusively on open and freely available development environments and programming languages, and (5) integrability with other simulation and analysis codes.

We adopted a functional approach to implementing the software, avoiding the use of classes. This choice has been made to provide greater flexibility and easier access for users with a wide variety of programming skills.  We used a uniform style for function definitions and documented all relevant functions with complete docstrings to make them as user-friendly as possible.

![pySNOW workflow](workflow.png)

The code provides four main subpackages, as schematized in figure \ref, including tools for reading/writing, analysing, and modifying atomistic configurations. These are:

* descriptors: the main pySNOW subpackage. It includes modules to compute several different descriptors. A comprehensive but non-exhaustive list includes: descriptors relevant to coordination (coordination numbers (CNs) and (strained) generalized CNs), common neighbour analysis (CNA) and CNA patterns, chemical element-wise atomic distributions (Pair Distance Distribution Function - PDDF, center of mass radial distribution function (RDF), pair radial density function, layer-by-layer density), nanoparticles’ general shape descriptors (including inertia tensor- and gyration tensor-based descriptors), chemical bond-based local atomic environment analysis, and Steinhardt parameters. It also includes a module to compute small-angle X-ray scattering (SAXS) spectra from both atomic coordinates and precomputed PDDFs.
* io: the input/output subpackage. It includes modules for reading from and writing to file. Currently supported formats are .xyz, .lammps-data, and .lammps-dump. It supports both trajectory and single-frame files.
* catalysis: a subpackage with functions oriented to the preparation and analysis of catalysis simulations. It includes the add_molecule module, which allows adding a number of adsorbed molecules on the surface(s) of a nanoparticle, and the ORR module, which computes the current, the mass activity and the specific activity at an applied potential of a nanoparticle based on its atop generalised coordination number as in [@Rossi2020].
* misc: a subpackage with some utilities (the rototranslation module, to perform rototranslations of atomic coordinates) and tabulated constants (the constants module, which contains atomic masses and SAXS-related constants).

For some functions, a fast version is provided and recommended for situations with a large number of atoms to avoid time-calculation bottlenecks.

These modules are documented in the provided documentation. Together with the code documentation, we provide a list of Jupyter Notebook tutorials that cover all the relevant modules, their use in common use cases, tips and tricks.
Finally, along with the main modules, `pySNOW` also provides several unit tests that can be run automatically with the `pytest` package (which constitues an optional dependency of the package).




[An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description.]: #


# Research Application (impact statement)
`pySNOW` is applicable to morphological analysis  during the coalescence of nanoalloys [@Zinzani_Baletto:2026]; catalysis mapping [@Leti2025]; and catalytic performance [@Zinzani_Rossi:2025].

`pySNOW` is taught at the Physics Department, Universita' degli Studi di Milano in a Master course, and it was used during undergraduate projects.



# AI usage disclosure

[Transparent disclosure of any use of generative AI in the software creation, documentation, or paper authoring. If no AI tools were used, state this explicitly. If AI tools were used, describe how they were used and how the quality and correctness of AI-generated content was verified.]: #

While we consider our work original, we have used generative AI tools (OpenAI’s ChatGPT 5.3 and Anthropic’s Claude Opus 4.6) to compare solutions and check algorithms in parts of the code.
All AI-assisted outputs were reviewed, edited, and
validated by the human authors, who designed the overall code architecture and take full
responsibility for the accuracy and originality of the software and all submitted materials.

[All AI-assisted outputs were reviewed, edited, and
validated by the human authors, who designed the overall code architecture and take full
responsibility for the accuracy and originality of the software and all submitted materials.]: #

[No generative AI tools were used in the development of this software, the writing
of this manuscript, or the preparation of supporting materials.]: #


# Author contributions
FB contributed to conceptualization, supervision, and financial support.
SZ, GB, and GN equally contributed to conceptualization, development, and documentation.
DA contributed to the SAXS part.
LFB contributed to GCN and molecule addition.
All authors conitrbuted to test the code, the tutorials, and contribute to the manuscript.

# Acknowledgements
SZ, DA, GN, LFB acknowledge the Università degli Studi di Milano and the PhD programme at the Physics department. Furthermore, SZ thanks the financial support from ISC-SrL (D.M. 117/2023 PNRR).
DA thanks the CNR-ICCOM financial support for his PhD studentship (D.M. 630/2024 PNRR).
LFB thanks the CNR-Unimi collaboration for her PhD studentship and the PNRR (D.M. 118/TD/2023).
GB, FB, and SZ acknowledge financial support from the European Commission under the EIC Pathfinder project CHIRALFORCE, contract number 101046961. GN and FB thank MONSTER, a project of the Italian National Centre for HPC, Big Data and Quantum computing (ICSC, CUP B93C22000620006), funded by the European Union - NextGenerationEU, through PNNR, for its financial support.
FB and LFB thank the financial support from the FWO grant number G076625N on “Design of plasmonic photocatalysts for CO2 conversion based on multi-metallic clusters".

All authors thank the constructive discussion with R. M. Jones (Alan Turing Institute) and Mirko Vanzan (Unimi).

# Conflict of interest
The authors declare no potential conflict of interests.

# References
