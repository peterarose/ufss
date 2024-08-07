from ufss.diagram_automation import DiagramGenerator
from ufss.composite_diagrams import CompositeDiagrams
#from ufss.UF2 import Wavepackets
#from ufss.UF2 import DensityMatrices
#from ufss.RKE import RKE_DensityMatrices
from ufss.signals import UF2Wavefunctions, UF2DensityMatrices, RKWavefunctions, RKDensityMatrices
import ufss.signals as signals
import ufss.HLG as HLG
from ufss.efield_shapes import *
from ufss.convergence_tester import efieldConvergence, Convergence

## Backwards compatibility
Wavepackets = UF2Wavefunctions
DensityMatrices = UF2DensityMatrices
RKE_DensityMatrices = RKDensityMatrices