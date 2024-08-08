import unittest
import numpy as np
import os

import ufss

def calculate_spectrum_with_UF2(interaction_picture_shift):
    folder = os.path.join('fixtures','copolymer_P1')

    uf2 = ufss.DensityMatrices(os.path.join(folder,'open'),
                               conserve_memory=True,
                               detection_type='complex_polarization')

    uf2.interaction_picture_shift = interaction_picture_shift

    # defining the optical pulses in the RWA
    sigma = 1
    t = np.arange(-20,21)/20*sigma*5
    ef = ufss.efield_shapes.gaussian(t,sigma)

    uf2.set_efields([t,t,t,t],[ef,ef,ef,ef],[1,1,1,1],[(0,1),(1,0),(1,0)])
    uf2.set_polarization_sequence(['x','x','x','x'])
    uf2.set_t(0.1,dt=3)

    tau = uf2.t.copy() #dtau is the same as the dt for local oscillator
    T = np.array([0,100])
    uf2.set_pulse_delays([tau,T])

    sig = uf2.calculate_signal_all_delays()

    return sig

def calculate_2Q_spectrum_with_UF2(interaction_picture_shift):
    folder = os.path.join('fixtures','copolymer_P1')

    uf2 = ufss.DensityMatrices(os.path.join(folder,'open'),
                               conserve_memory=True,
                               detection_type='complex_polarization')

    uf2.interaction_picture_shift = interaction_picture_shift

    # defining the optical pulses in the RWA
    sigma = 1
    t = np.arange(-20,21)/20*sigma*5
    ef = ufss.efield_shapes.gaussian(t,sigma)

    uf2.set_efields([t,t,t,t],[ef,ef,ef,ef],[1,1,1,1],[(0,2),(2,0),(1,0)])
    uf2.set_polarization_sequence(['x','x','x','x'])
    uf2.set_t(0.1,dt=3)

    tau = uf2.t.copy() #dtau is the same as the dt for local oscillator
    T = np.array([0,100])
    uf2.set_pulse_delays([tau,T])

    sig = uf2.calculate_signal_all_delays()

    return sig

def calculate_integrated_spectrum_with_UF2(interaction_picture_shift):
    folder = os.path.join('fixtures','copolymer_P1')

    uf2 = ufss.DensityMatrices(os.path.join(folder,'open'),
                               conserve_memory=True,
                               detection_type='integrated_polarization')

    uf2.interaction_picture_shift = interaction_picture_shift

    # defining the optical pulses in the RWA
    sigma = 1
    t = np.arange(-20,21)/20*sigma*5
    ef = ufss.efield_shapes.gaussian(t,sigma)

    uf2.set_efields([t,t,t,t],[ef,ef,ef,ef],[1,1,1,1],[(0,1),(1,0),(1,0)])
    uf2.set_polarization_sequence(['x','x','x','x'])
    uf2.set_t(0.1,dt=3)

    tau = uf2.t.copy() #dtau is the same as the dt for local oscillator
    T = np.array([0,100])
    uf2.set_pulse_delays([tau,T])

    sig = uf2.calculate_signal_all_delays()

    return sig

def L2_norm(a,b):
    """Computes L2 norm of two nd-arrays, taking the b as the reference"""
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2))
    

class test_shift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = os.path.join('fixtures','copolymer_P1')
        ufss.HLG.run(folder,conserve_memory=False)

    def test_1(self):
        sig1 = calculate_spectrum_with_UF2(True)
        sig2 = calculate_spectrum_with_UF2(False)
        # print(np.max(np.abs(sig1-sig2)))
        self.assertTrue(np.allclose(sig1,sig2))

    def test_2(self):
        sig1 = calculate_integrated_spectrum_with_UF2(True)
        sig2 = calculate_integrated_spectrum_with_UF2(False)
        # print(np.max(np.abs(sig1-sig2)))
        self.assertTrue(np.allclose(sig1,sig2))

    def test_3(self):
        sig1 = calculate_2Q_spectrum_with_UF2(True)
        sig2 = calculate_2Q_spectrum_with_UF2(False)
        # print(np.max(np.abs(sig1-sig2)))
        self.assertTrue(np.allclose(sig1,sig2))
    
if __name__ == '__main__':
    unittest.main()
