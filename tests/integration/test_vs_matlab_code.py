import unittest
import numpy as np
import os
from scipy.io import loadmat

import ufss

def calculate_spectrum_with_UF2():
    folder = os.path.join('fixtures','so_comparison_dimer')
    ufss.HLG.run(folder)

    ta = ufss.Wavepackets(os.path.join(folder,'closed'),
                              detection_type='integrated_polarization')

    # defining the optical pulses in the RWA
    M = 9 # number of points used to resolve optical pulses
    Delta = 6 # pulse interval
    t = np.linspace(-Delta/2,Delta/2,num=M)
    dt = t[1] - t[0]

    sigma = 1
    ef = ufss.gaussian(t,sigma)

    ta.set_polarization_sequence(['x','x'])

    ta.set_efields([t,t],[ef,ef],[0,0],[(1,1),(1,0)])

    # ta.set_t(gamma_dephasing)

    T = np.arange(0.6,112,0.6)

    ta.set_pulse_delays([T])

    sig = ta.calculate_signal_all_delays()

    return sig

def L2_norm(a,b):
    """Computes L2 norm of two nd-arrays, taking the b as the reference"""
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2))
    

class test_against_matlab(unittest.TestCase):
    """This test verifies that UFSS produces the same frequency-integrated
transient absorption spectrum as is generated using the code released with
the Ultrafast Spectroscopy book 
(https://iopscience.iop.org/book/978-0-750-31062-8, see supplementary 
material). The spectrum contained in the file 
fixtures/so_comparison_dimer/MATLAB_SOP_comparison.mat was obtained by 
running the downloaded code and using the parameters contained in the file
fixtures/so_comparison_dimer/MATLAB_SOP_comparison_params.mat.

This test serves to simultaneously test the diagram generator, the 
Hamiltonian generator, and UF2, since the comparison method uses
a different method for representing the Hamiltonian, and a different method
for propagating the wavefunctions and including the pulse shapes. The other
code Feynman diagrams that are written by hand. This does NOT test the 
Liouvillian generator or the open UF2 algorithm, since the MATLAB code only
works for closed systems.
"""
    def test(self):
        uf2_sig = calculate_spectrum_with_UF2()
        matlab_sig_path = os.path.join('fixtures','so_comparison_dimer',
                                       'MATLAB_SOP_comparison.mat')
        matlab_sig = loadmat(matlab_sig_path)['signal_T'][0,:]
        diff = L2_norm(-2*uf2_sig,matlab_sig)
        self.assertTrue(diff < 0.01)

if __name__ == '__main__':
    unittest.main()
