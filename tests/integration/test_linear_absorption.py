import unittest
import numpy as np
import os
import yaml

import ufss

def setup_params():
    """This function creates the file 'simple_params.yaml' inside the 
        specified folder. The Hamiltonian/Liouvillian Generator (HLG) part of
        ufss needs 'simple_params.yaml' to run.
"""
    truncation_size = 1 # irrelevant since no vibrations are included
    folder = os.path.join('fixtures','vibrationless_dimer')
    os.makedirs(folder,exist_ok=True)

    ### Define Hamiltonian
    site_energies = [0,0.75]
    site_couplings = [0.3307]

    vibs = []

    ### Define electronic dipoles
    # dipoles are input as a list of cartesian vectors [mu_x, mu_y, mu_z]
    mu = [[1,0,0],[1,0,0]]

    # Use a closed system for this calculation, which is acheived by
    # using 'coupling':0. This creates a closed folder and an open folder,
    # which should produce identical results, if all is working correctly
    # all other bath paramters are irrelevant in this case
    site_bath = {'cutoff_frequency':1,# in units of omega_0
                 'coupling':0,# system-bath coupling in units of omega_0
                 'temperature':1,# kT in units of omega_0
                 'cutoff_function':'lorentz-drude',
                 'spectrum_type':'ohmic'}

    Redfield_bath = {'secular':False, 
                     'site_bath':site_bath,
                     'vibration_bath':site_bath}

    params = {
        'site_energies':site_energies,
        'site_couplings':site_couplings,
        'dipoles':mu,
        'truncation_size':truncation_size,
        'vibrations':vibs,
        'bath':Redfield_bath}

    with open(os.path.join(folder,'simple_params.yaml'),'w+') as file_stream:
        yaml.dump(params,file_stream)

    ufss.HLG.run(folder)
    return None

def calculate_linear_absorption_with_UF2(closed_system=True):
    """Calculate the linear absorption spectrum using either the closed- or
        open-systems methods of UF2.
"""
    folder = os.path.join('fixtures','vibrationless_dimer')

    if closed_system == True:
        la = ufss.Wavepackets(os.path.join(folder,'closed'),
                              detection_type='polarization')
    else:
        la = ufss.DensityMatrices(os.path.join(folder,'open'),
                              detection_type='polarization')

    # defining the optical pulses in the RWA
    M = 41 # number of points used to resolve optical pulses
    Delta = 10 # pulse interval
    t = np.linspace(-Delta/2,Delta/2,num=M)
    dt = t[1] - t[0]

    sigma = 1
    ef = np.zeros(t.size,dtype='float')
    ef[ef.size//2] = 1/dt

    la.set_polarization_sequence(['x'])

    la.set_efields([t],[ef],[0],[(1,0)])

    gamma_dephasing = 0.2/sigma
    la.gamma_res = 20
    if closed_system == True:
        la.set_t(gamma_dephasing,set_gamma=False)
    else:
        la.set_t(gamma_dephasing)
    la.set_pulse_delays([])

    sig = la.calculate_signal_all_delays()

    return la.w, sig

def calculate_peak_ratio(w,sig):
    """Calculate the integrated peak ratios of a 1D signal, assuming that
        the peaks are centered at 0 and 1 omega0, respectively.
        Args:
            w (np.array) : 1D array of frequencies
            sig (np.array) : 1D array of the signal
        Returns:
            ratio of the integrated peak areas
"""
    peak0_inds = np.where((w>-0.5) & (w<0.5))[0]
    peak0_area = np.trapz(sig[peak0_inds],x=w[peak0_inds])

    peak1_inds = np.where((w>0.5) & (w<1.5))[0]
    peak1_area = np.trapz(sig[peak1_inds],x=w[peak1_inds])

    return peak0_area/peak1_area
    

class test_against_squared_dipoles(unittest.TestCase):
    """Check to make sure that the calculated linear absorption spectrum
        for a vibration-less dimer gives two peaks at 0 and 1 omega_0, and 
        that the integrated peak ratios match the ratio mu_0^2/mu_1^2. Note
        that this does not really check the HLG, since we are not asserting 
        what the values of mu_0 and mu_1 are, but are simply loading them
        from the file mu.npz
"""
    def test_open(self):
        # Use the HLG to set up the system
        setup_params()
        # Calculate the linear absorption spectrum
        w, sig = calculate_linear_absorption_with_UF2(closed_system=False)
        # Load the dipole operator that is fed to UF2
        mu_path = os.path.join('fixtures','vibrationless_dimer',
                               'closed','mu.npz')
        mu = np.load(mu_path)
        expected_peak_ratio = mu['0_to_1'][0,0,0]**2/mu['0_to_1'][1,0,0]**2
        actual_peak_ratio = calculate_peak_ratio(w,sig)
        abs_err = np.abs(expected_peak_ratio-actual_peak_ratio)
        rel_err = np.abs(abs_err/actual_peak_ratio)
        # I have found that the relative error is roughly 0.000497
        # I do not expect perfect agreement, since I am using a discrete
        # Fourier transform. Increasing the frequency resolution of the
        # signal should diminish the error
        self.assertTrue(rel_err<0.0005)

    def test_closed(self):
        # Use the HLG to set up the system
        setup_params()
        # Calculate the linear absorption spectrum
        w, sig = calculate_linear_absorption_with_UF2(closed_system=True)
        self.open_sig = sig.copy()
        # Load the dipole operator that is fed to UF2
        mu_path = os.path.join('fixtures','vibrationless_dimer',
                               'closed','mu.npz')
        mu = np.load(mu_path)
        expected_peak_ratio = mu['0_to_1'][0,0,0]**2/mu['0_to_1'][1,0,0]**2
        actual_peak_ratio = calculate_peak_ratio(w,sig)
        abs_err = np.abs(expected_peak_ratio-actual_peak_ratio)
        rel_err = np.abs(abs_err/actual_peak_ratio)
        # I have found that the relative error is roughly 0.000497
        # I do not expect perfect agreement, since I am using a discrete
        # Fourier transform. Increasing the frequency resolution of the
        # signal should diminish the error
        self.assertTrue(rel_err<0.0005)

    def test_open_vs_closed(self):
        """For linear signals, the open and closed case should be identical
            in the case of 0 coupling to the bath. Even when the bath 
            coupling is 0, this is not quite the case for nonlinear signals.
"""
        w, open_sig = calculate_linear_absorption_with_UF2(closed_system=False)
        w, closed_sig = calculate_linear_absorption_with_UF2(closed_system=True)
        self.assertTrue(np.allclose(closed_sig,open_sig))
        
if __name__ == '__main__':
    unittest.main()
