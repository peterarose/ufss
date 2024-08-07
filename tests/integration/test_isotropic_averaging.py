import numpy as np
import ufss
import os
import yaml
import unittest

def direction_cosine_matrix(phi,th,chi):
    c = np.cos
    s = np.sin
    M = np.array([
        [c(phi)*c(th)*c(chi)-s(phi)*s(chi), s(phi)*c(th)*c(chi)+c(phi)*s(chi), -s(th)*c(chi)],
        [-c(phi)*c(th)*s(chi)-s(phi)*c(chi), -s(phi)*c(th)*s(chi)+c(phi)*c(chi), s(th)*s(chi)],
        [c(phi)*s(th), s(phi)*s(th), c(th)]
        ])
    return M

def manual_average(sc,lab_polarizations,num_angles):
    sc.reset()
    sc.polarization_sequence = lab_polarizations
    signal = sc.calculate_signal_all_delays()
    signal[...] = 0
    dtheta = 2*np.pi/num_angles
    for i in range(0,num_angles//2):
        theta = dtheta*i
        for j in range(0,num_angles):
            phi = dtheta*j
            for k in range(0,num_angles):
                chi = dtheta * k
                dcm = direction_cosine_matrix(phi,theta,chi)

                new_polarizations = [dcm.dot(pol) for pol in lab_polarizations]
                sc.polarization_sequence = new_polarizations
                sc.reset()
                signal += sc.calculate_signal_all_delays()*np.sin(theta)*dtheta**3

    return signal/(8*np.pi**2)

def run_HLG():
    site_energies = [100,100.5]#*5
    site_couplings = [0.5]#[0.5,0,0.2,0.1,0]*9
    dipoles = [[1,0,0],
            [0,1,0]]#*5

    trunc_size = 1

    folder = 'fixtures/OrthogonalDimer'
    os.makedirs(folder,exist_ok=True)

    site_bath = {'cutoff_frequency':1,
                'coupling':0.05,
                'temperature':0,
                'cutoff_function':'lorentz-drude',
                    'spectrum_type':'ohmic'}

    vibration_bath = site_bath # use same bath for sites and vibrations

    relax_bath = {'dephasing_rate':0.05, # additonal "pure" dephasing
                'relaxation_rate':0.05,
                'temperature':0,
                    'spectrum_type':'white-noise'}


    Redfield_bath = {'secular':True,
            'site_bath':site_bath,
            'vibration_bath':vibration_bath}

    #include if you wish to model inter-manifold relaxation
    Redfield_bath['site_internal_conversion_bath'] = relax_bath

    vibrations = []

    params = {
        'site_energies':site_energies,
        'site_couplings':site_couplings,
        'dipoles':dipoles,
        'truncation_size':trunc_size,
        'vibrations':vibrations}

    params['bath'] = Redfield_bath

    with open(os.path.join(folder,'simple_params.yaml'),'w+') as new_file:
        yaml.dump(params,new_file)

    ufss.HLG.run(folder,conserve_memory=False)
    return folder

class test_iso_ave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = run_HLG()

        TA4 = ufss.signals.UF2Wavefunctions(os.path.join(folder,'closed'))
        TA4_pdc = ((1,0),(0,1),(1,0))
        TA4_times = [np.array([0])]*4
        TA4_efields = [np.array([1])]*4
        TA4_centers = [100]*4
        TA4.set_efields(TA4_times,TA4_efields,TA4_centers,TA4_pdc)

        TA4.gamma_res = 10
        TA4.set_t(0.1)

        T = np.arange(1,10)

        TA4.set_pulse_delays([np.array([0]),T])

        TA4.polarization_sequence= [np.array([1,0,0])]*4

        cls.TA4 = TA4

        cls.TA4_iso = ufss.signals.IsotropicAverage(cls.TA4)
        # running the below calculates all the necessary molecular frame 
        # polarizations and saves them, so that they don't have to be
        # redone for each test below
        cls.TA4_iso.averaged_signal(['x','x','x','x'],return_signal=True)[0,...]

        cls.rel_tol = 0.01

    def test_xx(self):
        sig_averaged_xx = self.TA4_iso.averaged_signal(['x','x','x','x'],
                                                       return_signal=True)[0,...]
        xxxx = [np.array([0,0,1])]*4
        sig_xx_manual_averaging = manual_average(self.TA4,xxxx,8)[0,:,:]
        abs_diff = np.max(np.abs(sig_xx_manual_averaging-sig_averaged_xx))
        rel_diff = abs_diff/np.max(np.abs(sig_averaged_xx))
        self.assertTrue(rel_diff<self.rel_tol)

    def test_zy(self):
        sig_averaged_xy = self.TA4_iso.averaged_signal(['x','x','y','y'],
                                                       return_signal=True)[0,...]
        #zzyy converges much faster than xxyy, but converges to the same result
        zzyy = [np.array([0,0,1])]*2 + [np.array([0,1,0])]*2
        sig_zy_manual_averaging = manual_average(self.TA4,zzyy,10)[0,:,:]
        abs_diff = np.max(np.abs(sig_zy_manual_averaging-sig_averaged_xy))
        rel_diff = abs_diff/np.max(np.abs(sig_averaged_xy))
        self.assertTrue(rel_diff<self.rel_tol)
    

class test_iso_ave_5th_order(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = run_HLG()

        TA6 = ufss.signals.UF2Wavefunctions(os.path.join(folder,'closed'))
        TA6_pdc = ((1,0),(1,0),(0,1),(0,1),(1,0))
        TA6_times = [np.array([0])]*6
        TA6_efields = [np.array([1])]*6
        TA6_centers = [100]*6
        TA6.set_efields(TA6_times,TA6_efields,TA6_centers,TA6_pdc)

        TA6.gamma_res = 10
        TA6.set_t(0.1)

        T = np.arange(1,10)

        TA6.set_pulse_delays([np.array([0]),np.array([0]),np.array([0]),T])

        TA6.polarization_sequence= [np.array([1,0,0])]*4

        cls.TA6 = TA6

        cls.TA6_iso = ufss.signals.IsotropicAverage(cls.TA6)
        # running the below calculates all the necessary molecular frame 
        # polarizations and saves them, so that they don't have to be
        # redone for each test below
        cls.TA6_iso.averaged_signal(['x','x','x','x','x','x'],return_signal=True)[0,0,0,...]

        cls.rel_tol = 0.01

    def test_zz(self):
        sig_averaged_xx = self.TA6_iso.averaged_signal(['x','x','x','x','x','x'],
                                                       return_signal=True)[0,0,0,...]
        z6 = [np.array([0,0,1])]*6
        sig_xx_manual_averaging = manual_average(self.TA6,z6,8)[0,0,0,:,:]
        abs_diff = np.max(np.abs(sig_xx_manual_averaging-sig_averaged_xx))
        rel_diff = abs_diff/np.max(np.abs(sig_averaged_xx))
        print(rel_diff)
        self.assertTrue(rel_diff<self.rel_tol)

    def test_zy(self):
        sig_averaged_xy = self.TA6_iso.averaged_signal(['x','x','x','x','y','y'],
                                                       return_signal=True)[0,...]
        #zzyy converges much faster than xxyy, but converges to the same result
        zzzzyy = [np.array([0,0,1])]*4 + [np.array([0,1,0])]*2
        sig_zy_manual_averaging = manual_average(self.TA6,zzzzyy,10)[0,:,:]
        abs_diff = np.max(np.abs(sig_zy_manual_averaging-sig_averaged_xy))
        rel_diff = abs_diff/np.max(np.abs(sig_averaged_xy))
        print(rel_diff)
        self.assertTrue(rel_diff<self.rel_tol)
if __name__ == '__main__':
    unittest.main()
