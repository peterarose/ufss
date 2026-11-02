"""Checks the frequency-matched secular approximation in
SecularRedfieldConstructor (params.yaml: bath.secular = True):

 - L does not depend on which eigenvectors are chosen inside a degenerate
   subspace (a ring trimer with a degenerate exciton pair);
 - the coherence-transfer terms of a harmonic mode are kept, so <a> decays at
   a single rate whichever ladder coherence it starts from;
 - models with no matched Bohr frequencies give the same L and Liouvillian
   eigenvalues as the stored fixtures made with the previous secular
   construction."""
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import scipy.sparse as sp
import yaml
from scipy.linalg import expm

import ufss
from ufss.HLG import SecularRedfieldConstructor, DiagonalizeLiouvillian

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'integration', 'fixtures')


def load_npz(folder, name):
    """Dict of arrays (sparse matrices stored as objects are unwrapped)."""
    out = {}
    with np.load(os.path.join(folder, 'open', name), allow_pickle=True) as arch:
        for key in arch.keys():
            x = arch[key]
            out[key] = x.item() if x.dtype == np.dtype('O') else x
    return out


def make_system(folder, simple_params):
    """Writes simple_params.yaml and runs the full HLG pipeline."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'simple_params.yaml'), 'w') as fh:
        yaml.dump(simple_params, fh)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ufss.HLG.run(folder, conserve_memory=True)


def ohmic(coupling, cutoff, temperature):
    return {'coupling': coupling, 'cutoff_frequency': cutoff,
            'cutoff_function': 'lorentz-drude', 'spectrum_type': 'ohmic',
            'temperature': temperature}


class TestSecularRedfield(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_degenerate_basis_invariance(self):
        """Rotating the eigenvectors inside each degenerate subspace by a
        random orthogonal matrix U must transform L covariantly,
        L' = W L W^T with W = U_ket^T (x) U_bra^T, and leave it otherwise
        unchanged."""
        J = -0.05
        params = {'site_energies': [1.0, 1.0, 1.0],
                  'site_couplings': [J, J, J],          # ring: J12, J13, J23
                  'dipoles': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  'truncation_size': 1,
                  'vibrations': [],
                  'bath': {'secular': True,
                           'site_bath': ohmic(0.05, 0.1, 0.025)}}
        folder_a = os.path.join(self.tmp, 'ring')
        make_system(folder_a, params)

        # same system, with rotated eigenvectors inside degenerate subspaces
        folder_b = os.path.join(self.tmp, 'ring_rotated')
        os.makedirs(folder_b)
        shutil.copy(os.path.join(folder_a, 'params.yaml'), folder_b)
        shutil.copytree(os.path.join(folder_a, 'closed'),
                        os.path.join(folder_b, 'closed'))
        rng = np.random.default_rng(1)
        with np.load(os.path.join(folder_a, 'closed', 'eigenvalues.npz')) as e_arch:
            evals = {k: e_arch[k] for k in e_arch.keys()}
        with np.load(os.path.join(folder_a, 'closed', 'eigenvectors.npz')) as v_arch:
            evecs = {k: v_arch[k] for k in v_arch.keys()}
        U = {}
        num_degenerate = 0
        for key, e in evals.items():
            U[key] = np.eye(e.size)
            groups = np.split(np.arange(e.size),
                              np.where(np.diff(e) > 1E-9 * max(1, np.abs(e).max()))[0] + 1)
            for g in groups:
                if g.size > 1:
                    num_degenerate += 1
                    Q, _ = np.linalg.qr(rng.normal(size=(g.size, g.size)))
                    U[key][np.ix_(g, g)] = Q
            evecs[key] = evecs[key].dot(U[key])
        self.assertGreater(num_degenerate, 0, 'test system has no degeneracy')
        np.savez(os.path.join(folder_b, 'closed', 'eigenvectors.npz'), **evecs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            SecularRedfieldConstructor(folder_b, conserve_memory=True)

        L_a = load_npz(folder_a, 'L.npz')
        L_b = load_npz(folder_b, 'L.npz')
        for L_key in L_a:
            if L_key == 'all_manifolds':
                ket_key = bra_key = 'all_manifolds'
            else:
                ket_key, bra_key = L_key.split(',')
            W = np.kron(U[ket_key].T, U[bra_key].T)
            La = sp.csr_matrix(L_a[L_key]).toarray()
            Lb = sp.csr_matrix(L_b[L_key]).toarray()
            scale = max(np.abs(La).max(), 1E-300)
            np.testing.assert_allclose(Lb, W.dot(La).dot(W.T), rtol=0,
                                       atol=1E-10 * scale,
                                       err_msg='L block ' + L_key)

    def test_harmonic_ladder_coherence_transfer(self):
        """A displaced-free harmonic mode coupled to a zero-temperature bath:
        <a>(t)/<a>(0) must be the same single damped oscillation whether the
        initial state is the coherence |1><0| or |3><2|. Without the
        frequency-matched transfer terms rho_21 -> rho_10, |3><2| decays
        three times faster than |1><0|."""
        params = {'site_energies': [1.0],
                  'site_couplings': [],
                  'dipoles': [[1, 0, 0]],
                  'truncation_size': 8,
                  'vibrations': [{'displacement': 0, 'omega_g': 0.2,
                                  'site_label': 0}],
                  'bath': {'secular': True,
                           'site_bath': ohmic(0.05, 0.1, 0.0),
                           'vibration_bath': ohmic(0.02, 0.5, 0.0)}}
        folder = os.path.join(self.tmp, 'ho')
        make_system(folder, params)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rc = SecularRedfieldConstructor(folder, conserve_memory=True,
                                            do_nothing=True)
        # no internal conversion, so the manifolds are separable; the
        # vibrational ladder is the ground manifold '0'
        self.assertIn('0', rc.eigenvalues)
        e = rc.eigenvalues['0']
        v = rc.eigenvectors['0']
        ad = rc.PolyVib.extract_vibronic_manifold(rc.PolyVib.total_ups[0], 0)
        a = v.T.dot(ad.T).dot(v)                 # annihilation op, eigenbasis
        n = e.size
        order = np.argsort(e)                    # number states n = 0, 1, ...
        L = sp.csr_matrix(load_npz(folder, 'L.npz')['0,0']).toarray()
        weights = a.T.ravel()                    # Tr(a rho) on row-major rho
        times = np.linspace(0, 40, 21)
        ratios = []
        for upper in (1, 3):
            i, j = order[upper - 1], order[upper]
            rho0 = np.zeros(n * n, dtype='complex')
            rho0[j * n + i] = 1.0                # |upper><upper-1|
            a0 = weights.dot(rho0)
            self.assertGreater(abs(a0), 0.5)
            r = np.array([weights.dot(expm(L * t).dot(rho0)) / a0 for t in times])
            ratios.append(r)
        np.testing.assert_allclose(ratios[1], ratios[0], rtol=0, atol=1E-8)

        # and that common decay is a single exponential
        mag = np.abs(ratios[0])
        rate = -np.polyfit(times, np.log(mag), 1)[0]
        self.assertGreater(rate, 0)
        np.testing.assert_allclose(mag, np.exp(-rate * times), rtol=1E-8)

    def test_unchanged_without_matched_frequencies(self):
        """Models whose Bohr frequencies never match reproduce the stored
        fixtures (built with the previous secular construction)."""
        for name in ['2LS', 'dimer_d0.2']:
            src = os.path.join(FIXTURES, name)
            folder = os.path.join(self.tmp, name)
            os.makedirs(folder)
            shutil.copy(os.path.join(src, 'params.yaml'), folder)
            shutil.copytree(os.path.join(src, 'closed'),
                            os.path.join(folder, 'closed'))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                SecularRedfieldConstructor(folder, conserve_memory=True)
                DiagonalizeLiouvillian(folder, conserve_memory=True, secular=True)
            for fname in ['L.npz', 'eigenvalues.npz']:
                new = load_npz(folder, fname)
                old = load_npz(src, fname)
                self.assertEqual(sorted(new.keys()), sorted(old.keys()))
                for key in old:
                    a = old[key]
                    b = new[key]
                    if sp.issparse(a) or sp.issparse(b):
                        a = sp.csr_matrix(a).toarray()
                        b = sp.csr_matrix(b).toarray()
                    scale = max(np.abs(a).max(initial=0), 1.0)
                    np.testing.assert_allclose(b, a, rtol=0, atol=1E-12 * scale,
                                               err_msg=f'{name} {fname} {key}')


if __name__ == '__main__':
    unittest.main()
