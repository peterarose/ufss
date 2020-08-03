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
numpy,matplotlib,pyfftw,scipy>=1,pyyaml

In order to try this code out, you can download this repo and run the jupyter notebooks contained within.

Installation instructions coming soon