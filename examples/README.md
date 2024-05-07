# Examples
This folder contains a number of notebooks that illustrate features of UFSS and show how figures in some of our references were produced.

## Les Houches Summer School Course
Dr. James Green and Dominik Brey developed a short course about nonlinear optical spectroscopy (NLOS) that uses UFSS to simulate linear absorption, transient absorption, and 2D spectra for a vibronic system coupled to a Markovian bath. The course is a jupyter notebook called "LesHouches_Spectroscopy.ipynb" and is included in this repository. It is a great introduction to both NLOS and how to use UFSS, and we are very grateful to Dr. Green for sharing it with us. This Jupyter notebook will be published as a chapter in a book that will be released sometime in the next year, and we will provide a link to that book once it is published. 

## Taking a test drive in Google's Colaboratory
To try ufss without installing or downloading the repository,
follow this link to see examples using the Diagram Generator:  
https://colab.research.google.com/github/peterarose/ufss/blob/master/examples/DiagramGeneratorExample_Colab.ipynb

You can also open any of the other example notebooks in google colab. You just need to add a cell with the command
"!pip install ufss"
above the cell that imports ufss

(Note: Google's Coloaboratory gives a warning message about running Jupyter
notebooks not authored by Google. When prompted by the warning, select
"RUN ANYWAY", and then click "YES" when it asks you if you would like to
reset all runtimes)

## Figures in UFSS papers
Several of these notebooks document the figures in the UFSS papers.
- DiagramGeneratorPaperFigure3.ipynb and DiagramGeneratorPaperFigure4.ipynb show how those figures in Ref. 1 are made.
- Smallwood2017Comparison.ipynb is used to produce Fig. 5 in Ref. 2.
- ThesisChapter4.ipynb shows how to make the figures in chapter 4 of my thesis (http://dx.doi.org/10.20381/ruor-27645).

## Other examples
- DiabaticLindblad_example.ipynb shows how to set up a model system using Lindblad formalism in the diabatic basis, and how to simulate 2D photon echo and transient absorption spectra, including rotational averaging.
- FullRedfield_example.ipynb shows how to set up a model system using non-secular Redfield theory, and how to simulate all of the same spectra as in DiabaticLindblad_example.ipynb. Also shows how to use the included adaptive step-size Runge-Kutta (RK45) direct propagation to calculate TA spectra, with a quantitative comparison of the results using RK45 and UF2.
- CircularlyPolarizedLight.ipynb and CircularlyPolarizedLight-OpenSystems.ipynb show how to compute spectra using closed and open systems, respectively, and circularly polarized light (all other examples assume linearly polarized light). Currently requires manually creating the Hamiltonian/Liovillian, since the HLG does not currently support complex dipole moments
- CircularlyPolarizedLight_inhomogeneous_random_sampling.py and CircularlyPolarizedLight_inhomogeneous_trapezoidal.py show how to do inhomogenous broadening over a single parameter, using random sampling or grid sampling. Random sampling is more efficient when broadening over many parameters, whereas grid sampling is more efficient when broadening over only one or two parameters. Grid sampling requires a careful choice of the grid, whereas random sampling does not.

## Convergence Testing
- efield_convergence_testing.py shows how to run convergence testing over the electric field parameters, given a set model system
- truncation_size_convergence_testing.py shows how to run convergence testing over the number of vibrational basis states included in the model system. Convergence testing is done for a particular electric field shape. Longer pulse durations (smaller frequency bandwidth) require fewer vibrational basis states

## References


[1] Peter A. Rose and Jacob J. Krich, "Automatic Feynman diagram generation for nonlinear optical spectroscopies and application to fifth-order spectroscopy with pulse overlaps", [J. Chem. Phys. 154, 034109 (2021)](https://doi.org/10.1063/5.0024105)


[2] Peter A. Rose and Jacob J. Krich, "Efficient numerical method for predicting nonlinear optical spectroscopies of open systems", [J. Chem. Phys. 154, 034108 (2021)](https://doi.org/10.1063/5.0024104)
