"""Checks the Schrodinger-picture spectral solvers used by method='chebyshev'
and method='hermite' when engine.damped_spectral is True (the default):
each must reproduce rho_j(t) = int_{-inf}^t exp(lam_j (t-t')) f(t') dt' for
decay rates ranging from 0 to many times the inverse pulse width, which the
interaction-picture algorithm cannot do."""
import unittest
import warnings
import numpy as np
import numpy.polynomial.hermite as nphe
import numpy.polynomial.chebyshev as npch
from scipy.integrate import quad
from ufss.perturbative_calculations.containers import (HermiteDampedTerm,
    ChebDampedTerm, HermitePoly)

W0 = 0.5
def src(t):
    return (np.exp(-(t-3)**2/(2*4**2))*(1 + 0.03*t + 0.2j*np.cos(0.4*t))
            * np.exp(-1j*W0*t))

GAMMA = np.array([0., 1E-3, 0.05, 1., 5., 40.])   # kappa = gamma*scale up to ~226
OMEGA = np.full(GAMMA.size, W0)
T_REF = 1.
T_EVAL = np.linspace(-30, 70, 41)

def reference():
    warnings.simplefilter('ignore')
    out = np.zeros((GAMMA.size, T_EVAL.size), dtype=complex)
    for j, (gam, om) in enumerate(zip(GAMMA, OMEGA)):
        lam = -gam + 1j*om
        for i, T in enumerate(T_EVAL):
            a = max(-60., T - 40/gam) if gam > 0 else -60.
            if a >= T:
                continue
            for part, unit in ((np.real, 1), (np.imag, 1j)):
                f = lambda tp: part(np.exp(lam*(T-tp))*src(tp))
                out[j, i] += unit*quad(f, a, T, limit=400, epsabs=1E-15,
                                       epsrel=1E-12)[0]
    return out

class test_damped_spectral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = reference()

    def _err(self, rho):
        return np.max(np.abs(rho - self.ref), axis=1)/np.max(np.abs(self.ref), axis=1)

    def test_hermite(self):
        scale = 4*np.sqrt(2)
        t = nphe.hermgauss(100)[0]*scale + 2
        u = src(t)[None, :]*np.exp(-1j*OMEGA[:, None]*(t - T_REF))
        term = HermiteDampedTerm.from_samples(np.arange(GAMMA.size), t, u, 2.,
                                              scale, T_REF, GAMMA, OMEGA)
        self.assertLess(np.max(self._err(term(T_EVAL))), 1E-6)

    def test_chebyshev(self):
        dom = np.array([-35., 45.])
        t = npch.chebpts1(120)*40 + 5
        u = src(t)[None, :]*np.exp(-1j*OMEGA[:, None]*(t - T_REF))
        term = ChebDampedTerm.from_samples(np.arange(GAMMA.size), t, u, dom,
                                           T_REF, GAMMA, OMEGA)
        self.assertLess(np.max(self._err(term(T_EVAL))), 1E-6)

    def test_hermite_gamma0_matches_ramp(self):
        scale = 4*np.sqrt(2)
        t = nphe.hermgauss(40)[0]*scale + 2
        u = src(t)[None, :]
        hp = HermitePoly(t, u, center=2., scale=scale)
        hp.integrate()
        term = HermiteDampedTerm.from_samples([0], t, u, 2., scale, 0.,
                                              np.array([0.]), np.array([0.]))
        self.assertLess(np.max(np.abs(term(T_EVAL) - hp(T_EVAL))), 1E-13)

if __name__ == '__main__':
    unittest.main()
