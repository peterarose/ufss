import unittest
import numpy as np
import numpy.polynomial.chebyshev as npch
import numpy.polynomial.hermite as nphe
import os

import ufss

def g(t,sigma):
    """t is time.  Gaussian pulse, with time-domain standard deviation sigma,
        normalized to behave like a delta function as sigma -> 0"""
    pre = 1/(np.sqrt(2*np.pi)*sigma)
    return pre * np.exp(-t**2/(2*sigma**2))

def calculate_spectrum(method='UF2',*,M=25):
    """Runs the Smallwood et al. comparison calculation, representing the
        three interacting pulses using one of three methods: 'UF2' (the
        default, FFT-based Heaviside convolution), 'chebyshev', or
        'hermite' (both of which replace that convolution with a spectral
        indefinite integral -- see ChebPoly/HermitePoly in containers.py).
        The local oscillator/detection time grid (lo_t, lo) is left
        untouched across all three: it's a discretized delta function on a
        uniform grid, which defines tau and t and must stay directly
        comparable to the stored analytical_signal reference regardless of
        how the interacting pulses themselves are represented.

    Keyword Args:
        M (int) : number of points used to resolve the three interacting
            pulses (Chebyshev/Gauss-Hermite node count, or uniform grid
            size for 'UF2'). See test_hermite_convergence.py for a sweep
            over this parameter for the 'hermite' method.
"""
    folder = os.path.join('fixtures','v_3LS')

    re = ufss.DensityMatrices(os.path.join(folder,'open'),
                              detection_type='complex_polarization')
    re.method = method

    # defining the optical pulses in the RWA
    Delta = 6 # pulse interval
    sigma = 1

    # Smallwood et al. use a delta function for the local oscillator
    lo_dt = 0.25 #### This must never change, because it defines t and tau, by default
    lo_t =  np.arange(-5,5.2,lo_dt)
    lo_dt = lo_t[1] - lo_t[0]
    lo = np.zeros(lo_t.size,dtype='float')
    lo[lo.size//2] = 1/lo_dt

    if method == 'chebyshev':
        # nodes/domain for cheb_perturbative_container, replacing the
        # uniform grid -- dom is shared by all three interacting pulses
        # (index 0,1,2); the local oscillator (index 3) never goes through
        # the spectral path since it's on a uniform grid (see
        # get_local_oscillator in calculate_signals.py)
        t = npch.chebpts1(M) * Delta/2
        dom = np.array([-Delta/2,Delta/2])
        re.doms = [dom,dom,dom,dom]
        # pre-existing gap shared by 'chebyshev' and 'hermite': exp_cutoff
        # guards against overflowing the open-system decaying-eigenvalue
        # exponentials, but has no default set anywhere in the repo
        re.exp_cutoff = 700
    elif method == 'hermite':
        # scale = sigma*sqrt(2) makes exp(-t**2/(2*sigma**2)) exactly
        # exp(-x**2) in the dimensionless Hermite variable x = t/scale
        scale = sigma * np.sqrt(2)
        t = nphe.hermgauss(M)[0] * scale
        re.herm_centers = [0.0,0.0,0.0,0.0]
        re.herm_scales = [scale,scale,scale,scale]
        re.exp_cutoff = 700
    else:
        t = np.linspace(-Delta/2,Delta/2,num=M)

    ef = g(t,sigma)

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

def calculate_spectrum_with_UF2():
    """Kept for backwards compatibility with any external callers."""
    return calculate_spectrum(method='UF2')

def L2_norm(a,b):
    """Computes L2 norm of two nd-arrays, taking the b as the reference"""
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2))


class test_against_smallwood(unittest.TestCase):
    def _check(self,method):
        sig_ft = calculate_spectrum(method=method)
        analytical_sig_path = os.path.join('fixtures','v_3LS',
                                           'analytical_signal.npy')
        analytical_sig = np.load(analytical_sig_path)
        # signal definition has changed since this test was first made
        # signals in UFSS are now a factor of pi smaller, and complex
        # signals have a factor of -1 relative to the original implementation
        diff = L2_norm(-np.pi*sig_ft[:,0,:],analytical_sig[:,0,:])
        self.assertTrue(diff < 0.007)

    def test(self):
        self._check('UF2')

    def test_chebyshev(self):
        self._check('chebyshev')

    def test_hermite(self):
        self._check('hermite')



if __name__ == '__main__':
    unittest.main()
