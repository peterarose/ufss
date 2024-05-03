"""
This doesn't work yet for things like TA, because the core is designed to treat the pump as a single pulse, whereas averaging 
needs to split the pump into two pulses, and the probe into two separate pulses
"""

#Standard python libraries
import os
import time

#Dependencies
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq


"""The following definitions of I4_mat and kdelvec are based
upon the formulas given in Appendix B of Molecular Quantum 
Electrodynamics, by Akbar Salam
"""

I4_mat = np.array([[4,-1,-1],[-1,4,-1],[-1,-1,4]])/30

def kdel(x,y):
    """Kronecker Delta"""
    if x == y:
        return 1
    else:
        return 0

def kdel2(a,b,c,d):
    """Product of 2 Kronecker Deltas"""
    return kdel(a,b)*kdel(c,d)

def kdelvec(i,j,k,l):
    """Length 3 vector of Kronecker Delta products, as defined in """
    vec = [kdel2(i,j,k,l),
           kdel2(i,k,j,l),
           kdel2(i,l,j,k)]
    return np.array(vec)

class FWMIsotropicAverage(object):
    """This class performs the isotropic average of the 4th order tensor
        which is the material response produced by 4-wave mixing process"""

    def __init__(self,spectra_calculator,lab_polarization,*,diagrams='all'):
        """Takes as input a ufss object that calculates 4-wave mixing spectra,
        and calculates the isotropically averaged signal, given a lab-frame polarization"""
        self.sc = spectra_calculator
        self.pol = lab_polarization
        self.diagrams = diagrams

    def molecular_frame_signal(self,mol_polarization):
        self.sc.set_polarization_sequence(mol_polarization)
        if self.diagrams == 'all':
            signal = self.sc.calculate_signal_all_delays()
        else:
            signal = self.sc.calculate_diagrams_all_delays(self.diagrams)
        return signal

    def averaged_signal(self,*,return_signal=False):

        t0 = time.time()

        left_vec = kdelvec(*self.pol)

        xyz = ['x','y','z']

        pol_options = []
        for i in range(3):
            # Check to see if the dipole operator has any non-zero components along the given
            # molecular frame axis, if the dipole exists only in the x-y plane, for example,
            # then we can avoid doing quite a few unnecessary calculations!
            try:
                random_mu_key = list(self.sc.mu.keys())[0]
                test_mu = self.sc.mu[random_mu_key]
            except AttributeError:
                random_mu_key = list(self.sc.H_mu.keys())[0]
                test_mu = self.sc.H_mu[random_mu_key]
            if type(test_mu) is np.ndarray:
                if not np.allclose(test_mu[:,:,i],0):
                    pol_options.append(xyz[i])
            elif type(test_mu) is list:
                if not np.allclose(test_mu[i][:,:],0):
                    pol_options.append(xyz[i])

        new_signal = True
        for i in pol_options:
            for j in pol_options:
                for k in pol_options:
                    for l in pol_options:
                        # generate the vector of kronecker delta products
                        right_vec = kdelvec(i,j,k,l)
                        if np.allclose(right_vec,0):
                            # If the vector is 0, don't bother!
                            pass
                        else:
                            # If not, set the polarization sequence, do the calculation, and
                            # add the weight given by the isotropic weight matrix, I4_mat
                            # Note the the polarization sequences are not the lab frame
                            # polarization sequence of the pulses.
                            mol_frame_signal = self.molecular_frame_signal([i,j,k,l])
                            weight = I4_mat.dot(right_vec)
                            weight = np.dot(left_vec,weight)
                            if new_signal:
                                signal = weight * mol_frame_signal
                                new_signal = False
                            else:
                                signal += weight * mol_frame_signal

        self.signal = signal

        self.calculation_time = time.time() - t0
        if return_signal:
            return signal

    def save(self,save_name,pulse_delay_names=[],*,use_base_path=True):
        if save_name == 'auto':
            save_name = 'FWM_IsotropicAverage_'+''.join(self.pol)
        self.sc.signal = self.signal
        self.sc.calculation_time = self.calculation_time
        self.sc.save(save_name,pulse_delay_names,use_base_path=use_base_path)

    
