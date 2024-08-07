import unittest
import numpy as np
import os
import yaml

import ufss

def run_HLG():
    cm_to_eV = 1/8065.54429    # cm^-1 to eV ; easier to just spell cm
    ev_to_THz = 1/4.1357E-15/1E12  # self-explanatory


    e_b = 13500*cm_to_eV

    e_a = 14700*cm_to_eV

    # transition coupling energies

    J_AB = -420*cm_to_eV  # one exciton coupling

    # transition dipole magnitudes

    # 0 -> 1 transition
    mu_a = np.array([1.2,0,0])

    mu_b = np.array([1,0,0])


    # bath parameters

    nu_a = 1.0

    nu_b = 1.66

    # v_f_average = (v_af + v_bf)/2

    lam = 180*cm_to_eV    # lower-case lambda
    gamma = 300*cm_to_eV  # capital lambda

    S = 0.18              # S, the Huang-Rhys factor
    Omega = 1280*cm_to_eV # Bath oscillation frequency
    Gamma = 10*cm_to_eV   # damping lower-case gamma


    # relaxation parameters are in fs or ps ; 
    # for the purposes of the HLG, we convert them to rates, in fs^-1, and then to eV radians

    # the 1->0 relaxation is dependent upon the polymer length, so is defined in the next cell

    def make_2LS_dimer():
        num_pairs = 1
        params_folder = os.path.join('fixtures','2LS_dimer_b')
        os.makedirs(params_folder,exist_ok=True)

        ### Making closed system parameters ###
        num_sites = 2*num_pairs # total number of sites

        site_energies = []

        for i in range(num_pairs):
            # site energies e_a and f_a for SQA monomer
            energies_a = e_a
            site_energies.append(energies_a)
            # site energies e_b and f_b for SQB monomer
            energies_b = e_b
            site_energies.append(energies_b)

        J_list = []

        for i in range(num_sites):
            for j in range(i+1,num_sites):
                # there's only coupling of adjacent units ; if i and j differ by more than 1 (i.e. if for J_i,j we have j-i>1,)
                if j-i > 1:
                    J = 0
                else : 
                    J = J_AB
                J_list.append(J)

        site_couplings = [J_list]

        dipoles = np.zeros((num_sites,3))
        # identical dipoles for all sites
        dipoles[0,:] = mu_a
        dipoles[1,:] = mu_b

        for i in range(2,num_sites-1,2):
            # dipole operator for SQA unit
            dipoles[i,...] = dipoles[0,...]

            # dipole operator for SQB unit
            dipoles[i+1,...] = dipoles[1,...]

        dipoles_list = dipoles.tolist()

        ### open system parameters ###

        # e -> g relaxation rate
        tau_r = (1710.2 - 69.1*num_pairs) # in ps

        k_r = 1/tau_r  # in ps^-1

        k_r = k_r/ ev_to_THz / (2*np.pi) # in eV radians ; no need for the *1000 factor since k_r is in ps^-1

        kT = 0.025  # assuming room temperature

        # site-bath coupling of singly-excited state, A-type monomer
        site_bath_a = {'cutoff_frequency':gamma,
                    'coupling': lam,
                    'temperature':kT,
                    'cutoff_function':'brownian',
                    'S': S,
                    'Gamma' : Gamma,          
                    'Omega' : Omega,
                    'nu': nu_a,
                    'spectrum_type':'ohmic'}

        # site-bath coupling of singly-excited states, B-type monomer
        site_bath_b = {'cutoff_frequency':gamma,
                    'coupling': lam,
                    'temperature':kT,
                    'cutoff_function':'brownian',
                    'S': S,
                    'Gamma' : Gamma,          
                    'Omega' : Omega,
                    'nu': nu_b,
                    'spectrum_type':'ohmic'}

        # relaxation bath for 1->0 processes
        relax_bath = {'dephasing_rate':0.01, # maybe does nothing!
                    'relaxation_rate':k_r,
                    'temperature':kT,
                        'spectrum_type':'white-noise'}

        Redfield_bath = {'secular':True,
                'site_bath':[site_bath_a,site_bath_b],
                'site_bath_option' : [0,1]*num_pairs,
                'vibration_bath':site_bath_a,
                'site_internal_conversion_bath':relax_bath}

        params = {
            'site_energies':site_energies,
            'site_couplings':site_couplings,
            'dipoles':dipoles_list,
            'bath':Redfield_bath,
            'truncation_size':1,
            'vibrations':[]}

        with open(os.path.join(params_folder,'simple_params.yaml'),'w+') as new_file:
            yaml.dump(params,new_file)

        ufss.HLG.run(params_folder,conserve_memory=False)

    make_2LS_dimer()

def calculate_PPP(order,*,include_wtau_DC_terms=True,include_linear=True):
    cm_to_eV = 1/8065.54429    # cm^-1 to eV ; easier to just spell cm
    e_b = 13500*cm_to_eV
    center = e_b #center on squaraine B's transition energy
    open_folder = os.path.join('fixtures','2LS_dimer_b','open')
    td = ufss.signals.PumpPumpProbe(open_folder,conserve_memory=False,
                                    include_linear=include_linear,
                                    include_wtau_DC_terms=include_wtau_DC_terms)

    #impulsive or finite pulses
    td.set_impulsive_pulses(center)
    td.engine.efields = [np.array([0.5]),np.array([0.5]),np.array([1])]
    #td.set_identical_gaussians(1,center,M=101,delta=10)

    # highest deisred order in perturbation theory
    td.set_highest_order(order)

    # set the time-grid, 'auto' only works for open systems, otherwise
    # specify an exponential decay rate as a float imposed on the decay of P(t)
    td.set_t(0.05,dt=3)

    # inter-pulse delay times
    # coherence times
    tau = np.array([0])
    # populations times
    T = np.array([1E-2])
    td.set_pulse_delays([tau,T])
    td.calculate_signal_all_delays()
    return td

def calculate_TA(order,include_linear=True):
    cm_to_eV = 1/8065.54429   # cm^-1 to eV ; easier to just spell cm
    e_b = 13500*cm_to_eV
    center = e_b #center on squaraine B's transition energy
    open_folder = os.path.join('fixtures','2LS_dimer_b','open')
    ta = ufss.signals.TransientAbsorption(open_folder,conserve_memory=False,
                                          include_linear=include_linear)

    #impulsive or finite pulses
    ta.set_impulsive_pulses(center)
    #ta.set_identical_gaussians(1,center,M=101,delta=10)

    # highest deisred order in perturbation theory
    ta.set_highest_order(order)

    # set the time-grid, 'auto' only works for open systems, otherwise
    # specify an exponential decay rate as a float imposed on the decay of P(t)
    ta.set_t(0.05,dt=3)

    # populations times
    T = np.array([1E-2])
    ta.set_pulse_delays([T])
    ta.calculate_signal_all_delays()
    return ta

class test_TA_vs_PPP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run_HLG()

    def test_1(self):
        order = 25
        ppp = calculate_PPP(order)
        ta = calculate_TA(order)
        for i in range((order+1)//2):
            y1 = ppp.get_signal_order(2*i+1,tau_dft=False)[0,0,:]
            y2 = ta.get_signal_order(2*i+1)[0,:]
            res = np.allclose(y1,y2,atol=1E-20)
            with self.subTest(res=res):
                self.assertTrue(res)

    def test_2(self):
        order = 25
        ppp = calculate_PPP(order,include_linear=False,include_wtau_DC_terms=False)
        ta = calculate_TA(order,include_linear=False)
        for i in range(1,(order+1)//2):
            y1 = ppp.get_signal_order(2*i+1,tau_dft=False)[0,0,:]
            y2 = ta.get_signal_order(2*i+1)[0,:]
            res = np.allclose(y1,y2,atol=1E-20)
            with self.subTest(res=res):
                self.assertFalse(res)

if __name__ == '__main__':
    unittest.main()