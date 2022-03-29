import unittest
import numpy as np
import os

import ufss

def g(t,sigma):
    """t is time.  Gaussian pulse, with time-domain standard deviation sigma, 
        normalized to behave like a delta function as sigma -> 0"""
    pre = 1/(np.sqrt(2*np.pi)*sigma)
    return pre * np.exp(-t**2/(2*sigma**2))

def calculate_spectrum_with_UF2():
    folder = os.path.join('fixtures','v_3LS')

    re = ufss.DensityMatrices(os.path.join(folder,'uf2'),
                              detection_type='complex_polarization')

    # defining the optical pulses in the RWA
    M = 25 # number of points used to resolve optical pulses
    Delta = 6 # pulse interval
    t = np.linspace(-Delta/2,Delta/2,num=M)
    dt = t[1] - t[0]

    sigma = 1
    ef = g(t,sigma)

    # Smallwood et al. use a delta function for the local oscillator
    lo_dt = 0.25 #### This must never change, because it defines t and tau, by default
    lo_t =  np.arange(-5,5.2,lo_dt)    
    lo_dt = lo_t[1] - lo_t[0]
    lo = np.zeros(lo_t.size,dtype='float')
    lo[lo.size//2] = 1/lo_dt


    re.set_polarization_sequence(['x','x','x','x'])

    re.set_efields([t,t,t,lo_t],[ef,ef,ef,lo],[0,0,0,0],[(0,1),(1,0),(1,0)])

    gamma_dephasing = 0.2/sigma
    re.gamma_res = 20
    re.set_t(gamma_dephasing)
    re.pulse_times = [0,0,0]

    tau = re.t.copy() #dtau is the same as the dt for local oscillator
    T = np.arange(0,1,1)
    re.set_pulse_delays([tau,T])

    time_ordered_diagrams = [(('Bu', 0), ('Ku', 1), ('Ku', 2)), (('Bu', 0), ('Ku', 1), ('Bd', 2)), (('Bu', 0), ('Bd', 1), ('Ku', 2))]

    sig = re.calculate_diagrams_all_delays(time_ordered_diagrams)

    ift = ufss.signals.SignalProcessing.ift1D
    
    wtau, sig_ft = ift(tau,sig,zero_DC=False,axis=0)

    return sig_ft

def L2_norm(a,b):
    """Computes L2 norm of two nd-arrays, taking the b as the reference"""
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2))
    

class test_against_smallwood(unittest.TestCase):
    def test(self):
        sig_ft = calculate_spectrum_with_UF2()
        analytical_sig_path = os.path.join('fixtures','v_3LS',
                                           'analytical_signal.npy')
        analytical_sig = np.load(analytical_sig_path)
        diff = L2_norm(sig_ft[:,0,:],analytical_sig[:,0,:])
        self.assertTrue(diff < 0.007)
    


if __name__ == '__main__':
    unittest.main()
