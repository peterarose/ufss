import unittest
import numpy as np
import os
import yaml
from scipy.io import loadmat

import ufss

folder = 'fixtures/dimer_d0.2'

def make_L():
    site_energies = [0,0.75]
    site_couplings = [0.3307]
    dipoles = [[1,0,0],[0,1,0]]

    site_bath = {'cutoff_frequency':1,
                 'coupling':0.05,
                 'temperature':1,
                 'cutoff_function':'lorentz-drude',
                 'spectrum_type':'ohmic'}

    vibration_bath = site_bath
    trunc_size = 1

    vibraations = []

    vibrations = [{'displacement':0,'site_label':0,'omega_g':1},
                  {'displacement':0,'site_label':1,'omega_g':1.01}]

    os.makedirs(folder,exist_ok=True)


    secular_bath = {'secular':True,
            'site_bath':site_bath,
            'vibration_bath':vibration_bath}

    secular_params = {
        'site_energies':site_energies,
        'site_couplings':site_couplings,
        'dipoles':dipoles,
        'truncation_size':trunc_size,
        'num_eigenvalues':'full',
        'eigenvalue_precision':1,
        'vibrations':vibrations,
        'bath':secular_bath}

    with open(os.path.join(folder,'simple_params.yaml'),'w+') as new_file:
        yaml.dump(secular_params,new_file)

    ufss.HLG.run(folder,conserve_memory=False)

    return None

def gaussian(t,sigma):
    """t is time.  Gaussian pulse, with time-domain standard deviation sigma, 
    normalized to behave like a delta function as sigma -> 0"""
    pre = 1/(np.sqrt(2*np.pi)*sigma)
    return pre * np.exp(-t**2/(2*sigma**2))

def setup(uf2_flag=True,conserve_memory=True,open=True):
    open_folder = os.path.join(folder,'open')
    closed_folder = os.path.join(folder,'closed')
    if open:
        if uf2_flag:
            obj = ufss.DensityMatrices(open_folder,detection_type='polarization',conserve_memory=conserve_memory)
            M = 41
        else:
            obj = ufss.RKE_DensityMatrices(open_folder,detection_type='polarization',conserve_memory=conserve_memory)
            M = 801
    else:
        if uf2_flag:
            obj = ufss.Wavepackets(closed_folder,detection_type='polarization')
            M = 41
        else:
            obj = ufss.RKWavefunctions(closed_folder,detection_type='polarization')
            M = 801


    sigma = 1
    Delta = 10
    t = np.linspace(-Delta/2,Delta/2,num=M)*sigma
    ef = gaussian(t,sigma)

    obj.set_efields([t,t,t,t],[ef,ef,ef,ef],[0,0,0,0],[(0,1),(1,0),(1,0)])

    obj.set_polarization_sequence(['x','x','x','x'])
    obj.set_t(0.137,dt=0.5)

    tau = np.array([0])

    T = np.arange(10)

    obj.set_pulse_delays([tau,T])

    return obj

def L2_norm(a,b):
    """Computes L2 norm of two nd-arrays, taking the b as the reference"""
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2))
    

class test_UF2_vs_RKE(unittest.TestCase):
    """This is to test that UF2 and RKE gives the same result, to
        within the expected tolerance. This tests the UF2 and RKE
        modules, both for open and closed calculations, but does
        not test the HLG, since both UF2 and RKE are relying on it
        for this test.
"""
    def test_1(self):
        make_L()
        uf2 = setup(uf2_flag = True)
        uf2_sig = uf2.calculate_signal_all_delays()

        rke = setup(uf2_flag = False)
        rke_sig = rke.calculate_signal_all_delays()

        diff = L2_norm(uf2_sig,rke_sig)
        self.assertTrue(diff < 0.01)

    def test_2(self):
        make_L()
        uf2 = setup(uf2_flag = True,conserve_memory=False)
        uf2_sig = uf2.calculate_signal_all_delays()

        rke = setup(uf2_flag = False,conserve_memory=False)
        rke_sig = rke.calculate_signal_all_delays()

        diff = L2_norm(uf2_sig,rke_sig)
        self.assertTrue(diff < 0.01)

    def test_3(self):
        make_L()
        uf2 = setup(uf2_flag = True, open=False)
        uf2_sig = uf2.calculate_signal_all_delays()

        rke = setup(uf2_flag = False, open=False)
        rke_sig = rke.calculate_signal_all_delays()

        diff = L2_norm(uf2_sig,rke_sig)
        self.assertTrue(diff < 0.01)

if __name__ == '__main__':
    unittest.main()
