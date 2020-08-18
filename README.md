# Ultrafast Spectroscopy Suite (UFSS)
Code for simulating nonlinear optical spectroscopies of closed and open systems

Contains 4 separate modules:
1. Diagram Generator (DG) - tool for automatically generating all Feynman diagrams for arbitrary order nonlinear optical spectroscopies
2. Ultrafast Ultrafast (UF2) - fast algorithm for calculating individual Feynman diagrams including arbitrary pulse shapes
  - for closed systems: class Wavepackets in UF2_core.py
  - for open systems: class DensityMatrices in UF2_open_core.py
3. Runga-Kutta-Euler (RKE) - alternative algorithm for calculating individual Feynman diagrams including arbitrary pulse shapes
  - for closed systems: currently broken
  - for open systems: class RKE_DensityMatrices in RKE_open_core.py
4. Hamiltonian/Liouvillian Generator (HLG) - tool for generating vibronic Hamiltonians and (optionally) Liouvillians from simple parameter inputs

This code depends upon the following packages:
numpy,matplotlib,pyfftw,scipy>=1,pyyaml,pyx

To get started with this code, clone this repo and have a look at the four jupyter notebooks contained within. They give examples of working with the diagram generator and generating spectroscopic signals. They also show how to generate figures in the manuscripts describing UFSS.

Installation instructions coming soon

[1] Peter A. Rose and Jacob J. Krich, "Automatic Feynman diagram generation for nonlinear optical spectroscopies", arXiv:2008.05081

[2] Peter A. Rose and Jacob J. Krich, "Efficient numerical method for predicting nonlinear optical spectroscopies of open systems", arXiv:2008.05082