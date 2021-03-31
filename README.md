# Ultrafast Spectroscopy Suite (UFSS)
Code for simulating nonlinear optical spectroscopies of closed and open systems

## Installation instructions
pip install ufss  

(Note: ufss is only written for python 3, so you may need to run
pip3 install ufss if pip points to python 2 on your machine)  

Visualizing Feynman diagrams requires a TeX distribution

(If you would like to try ufss out without cloning this repository or installing, see below for links to Google Colaboratory notebooks)

## Overview

Contains 4 separate modules:
1. Diagram Generator (DG) - tool for automatically generating all Feynman diagrams for arbitrary order nonlinear optical spectroscopies
  - class DiagramGenerator in ufss/diagram_automation.py
2. Ultrafast Ultrafast (UF2) - fast algorithm for calculating individual Feynman diagrams including arbitrary pulse shapes
  - for closed systems: class Wavepackets in ufss/UF2_core.py
  - for open systems: class DensityMatrices in ufss/UF2_open_core.py
3. Runga-Kutta-Euler (RKE) - alternative algorithm for calculating individual Feynman diagrams including arbitrary pulse shapes
  - for closed systems: to be included in this repository eventually. For a working version see https://github.com/peterarose/ultrafastultrafast
  - for open systems: class RKE_DensityMatrices in ufss/RKE_open_core.py
4. Hamiltonian/Liouvillian Generator (HLG) - tool for generating vibronic Hamiltonians and (optionally) Liouvillians from simple parameter inputs
  - contained in vibronic_eigenstates sub-directory


## Taking a test drive in Google's Colaboratory
To try ufss without installing or downloading the repository,
follow this link to see examples using the Diagram Generator:  
https://colab.research.google.com/github/peterarose/ufss/blob/master/DiagramGeneratorExample_Colab.ipynb  
and this link to see examples using the Hamiltonian/Liouvillian Generator and open UF2:  
coming soon

(Note: Google's Coloaboratory gives a warning message about running Jupyter
notebooks not authored by Google. When prompted by the warning, select
"RUN ANYWAY", and then click "YES" when it asks you if you would like to
reset all runtimes)

## Examples
There are four jupyter notebooks included with this repository that give examples of working with the diagram generator and generating spectroscopic signals. Some notebooks show how to generate figures in the manuscripts describing UFSS.

## Dependencies  
This code depends upon the following packages:  
numpy, matplotlib, pyfftw, scipy>=1, pyyaml, pyx

To get started with this code, clone this repo and have a look at the four jupyter notebooks contained within. They give examples of working with the diagram generator and generating spectroscopic signals. They also show how to generate figures in the manuscripts describing UFSS.

[1] Peter A. Rose and Jacob J. Krich, "Automatic Feynman diagram generation for nonlinear optical spectroscopies and application to fifth-order spectroscopy with pulse overlaps", [J. Chem. Phys. 154, 034109 (2021)](https://doi.org/10.1063/5.0024105)


[2] Peter A. Rose and Jacob J. Krich, "Efficient numerical method for predicting nonlinear optical spectroscopies of open systems", [J. Chem. Phys. 154, 034108 (2021)](https://doi.org/10.1063/5.0024104)
