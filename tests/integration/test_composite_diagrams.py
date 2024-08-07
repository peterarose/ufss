import unittest
import numpy as np
import os
import yaml
from scipy.io import loadmat

import ufss

folder = 'fixtures/2LS'

def make_L():
    site_energies = [1]
    site_couplings = []
    dipoles = [[1,0,0],]

    site_bath = {'cutoff_frequency':1,
                 'coupling':0.05,
                 'temperature':0,
                 'cutoff_function':'lorentz-drude',
                 'spectrum_type':'ohmic'}

    vibration_bath = site_bath
    trunc_size = 1

    vibrations = []

    os.makedirs(folder,exist_ok=True)

    relax_bath = {'dephasing_rate':0.1, # additonal "pure" dephasing
              'relaxation_rate':0.05,
               'temperature':0,
                'spectrum_type':'white-noise'}

    secular_bath = {'secular':True,
        'site_bath':site_bath,
        'vibration_bath':vibration_bath,
        'site_internal_conversion_bath':relax_bath}

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

def setup(uf2_flag=True,conserve_memory=True,open=True):
    if open:
        calc_folder = os.path.join('/Users/prose/fossil_ufss/Examples/2LS','open')
    else:
        calc_folder = os.path.join('/Users/prose/fossil_ufss/Examples/2LS','closed')
    if uf2_flag:
        engine_name = 'UF2'
    else:
        engine_name = 'RK' 
    obj = ufss.signals.TransientAbsorption(calc_folder,include_linear=True,
                                           engine_name=engine_name,
                                           conserve_memory=conserve_memory,
                                           detection_type='polarization')

    center = 1
    sigma = 1
    highest_order = 7
    gamma = 0.1
    T = np.arange(0,1)*np.pi/4 + 1E-2 + sigma*10
    dt = 0.5
    obj.set_identical_gaussians(sigma,center,M=101)
    #obj.set_impulsive_pulses(center)
    obj.set_hightest_order(highest_order)
    obj.set_t(gamma,dt=dt)
    obj.set_pulse_delays([T])

    return obj

def L2_norm(a,b):
    """Computes L2 norm of two nd-arrays, taking the b as the reference"""
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2))
    

class test_composite_diagrams(unittest.TestCase):
    """This is to test that the diagrams and composite diagrams give the same
        result, up to 7th order
"""
    def test_1(self):
        make_L()
        ta = setup(uf2_flag = True)
        ta.calculate_signal_all_delays(composite_diagrams=True)

        ta2 = setup(uf2_flag = True)
        ta2.calculate_signal_all_delays(composite_diagrams=False)

        for i in range(1,4,2):
            y = ta.get_signal_order(i)[0,:]
            y2 = ta2.get_signal_order(i)[0,:]
            res = np.allclose(y,y2,atol=1E-3)
            with self.subTest(res=res):
                self.assertTrue(res)

    def test_2(self):
        make_L()
        ta = setup(uf2_flag = True, open = False)
        ta.calculate_signal_all_delays(composite_diagrams=True)

        ta2 = setup(uf2_flag = True, open = False)
        ta2.calculate_signal_all_delays(composite_diagrams=False)

        for i in range(1,4,2):
            y = ta.get_signal_order(i)[0,:]
            y2 = ta2.get_signal_order(i)[0,:]
            res = np.allclose(y,y2,atol=1E-3)
            with self.subTest(res=res):
                self.assertTrue(res)

if __name__ == '__main__':
    unittest.main()
