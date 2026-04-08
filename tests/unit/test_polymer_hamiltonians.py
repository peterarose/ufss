"""
Unit tests for the Polymer class in ufss/HLG/Hamiltonians.py

Organisation
------------
Test3LS  - regression tests for existing 2-site 3LS behaviour.
           These MUST continue to pass after the 4LS extension is added.
Test3LSCouplingK1  - tests for the new K1 (b†b) coupling added to 3LS.
                     These will FAIL until K1 is implemented.
Test4LS  - specification / correctness tests for the new 4LS support.
           These will FAIL until 4LS is implemented.

Operator naming (user's convention)
-------------------------------------
For a single 3LS site (states g, e, f):
  a†  = |e><g|  (+1 excitation, 0→1)
  b†  = |f><g|  (+2 excitation, 0→2)
  c†  = |f><e|  (+1 excitation, 1→2)

  J   coupling  a†_i a_j + h.c.  (singly excited manifold)
  K   coupling  c†_i a_j + h.c.  (doubly excited manifold)
  K1  coupling  b†_i b_j + h.c.  (doubly excited manifold) ← NEW, missing from 3LS
  L   coupling  c†_i c_j + h.c.  (triply excited manifold)

For a single 4LS site (states g, e, f, h):
  d†  = |h><g|  (+3 excitation, 0→3)
  p†  = |h><e|  (+2 excitation, 1→3)
  q†  = |h><f|  (+1 excitation, 2→3)

  L1  coupling  q†_i a_j + h.c.  (triply excited)
  L2  coupling  p†_i b_j + h.c.  (triply excited)
  L3  coupling  d†_i d_j + h.c.  (triply excited)
  M0  coupling  q†_i c_j + h.c.  (quadruply excited)
  M1  coupling  p†_i p_j + h.c.  (quadruply excited)
  N0  coupling  q†_i q_j + h.c.  (quintuply excited)

Dipole ordering for 4LS (6-vector form per site):
  index 0: g<->e (d_ge)
  index 1: e<->f (d_ef)
  index 2: f<->h (d_fh)
  index 3: g<->f (d_gf)
  index 4: g<->h (d_gh)
  index 5: e<->h (d_eh)

Dipole shortcut (3-vector form per site = ladder-only):
  index 0: g<->e
  index 1: e<->f
  index 2: f<->h
  (d_gf, d_gh, d_eh are all zero)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ufss.HLG.Hamiltonians import Polymer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _3LS_single(e1=1.0, e2=2.5, d01=None, d12=None, d02=None):
    """Build a single-site 3LS Polymer."""
    d01 = d01 if d01 is not None else [1, 0, 0]
    d12 = d12 if d12 is not None else [1, 0, 0]
    d02 = d02 if d02 is not None else [0, 0, 0]
    site_energies = [[e1, e2]]
    site_couplings = [[], [], []]
    dipoles = np.array([[d01, d12, d02]])   # shape (1, 3, 3)
    return Polymer(site_energies, site_couplings, dipoles)


def _3LS_dimer(e1=1.0, e2=2.5, J=0.0, K=0.0, L=0.0):
    """Build a degenerate 2-site 3LS dimer using the matrix coupling format."""
    site_energies = [[e1, e2], [e1, e2]]
    J_mat = [[0, J], [J, 0]]
    K_mat = [[0, K], [K, 0]]
    L_mat = [[0, L], [L, 0]]
    site_couplings = [J_mat, K_mat, L_mat]
    dipoles = np.array([
        [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
    ])
    return Polymer(site_energies, site_couplings, dipoles)


def _4LS_single(e1=1.0, e2=2.5, e3=5.0, dipoles=None):
    """Build a single-site 4LS Polymer (ladder dipoles by default)."""
    site_energies = [[e1, e2, e3]]
    site_couplings = {}
    if dipoles is None:
        # 3-vector shortcut = ladder-only
        dipoles = np.array([[[1, 0, 0], [1, 0, 0], [1, 0, 0]]])  # shape (1, 3, 3)
    return Polymer(site_energies, site_couplings, dipoles)


def _4LS_dimer(e1=1.0, e2=2.5, e3=5.0, couplings=None):
    """Build a degenerate 2-site 4LS dimer."""
    site_energies = [[e1, e2, e3], [e1, e2, e3]]
    site_couplings = couplings if couplings is not None else {}
    dipoles = np.array([
        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
    ])
    return Polymer(site_energies, site_couplings, dipoles)


# ---------------------------------------------------------------------------
# 3LS regression tests
# ---------------------------------------------------------------------------

class Test3LS:
    """Regression tests – must pass before and after 4LS is added."""

    def test_detection(self):
        p = _3LS_single()
        assert p.N == 3

    def test_single_site_hilbert_dim(self):
        p = _3LS_single()
        assert p.H_dim == 3

    def test_dimer_hilbert_dim(self):
        p = _3LS_dimer()
        assert p.H_dim == 9

    def test_single_site_hamiltonian(self):
        """H is diagonal in site basis with energies [0, e1, e2]."""
        e1, e2 = 1.0, 2.5
        p = _3LS_single(e1=e1, e2=e2)
        np.testing.assert_allclose(
            p.electronic_hamiltonian, np.diag([0.0, e1, e2]), atol=1e-12
        )

    def test_hamiltonian_hermitian(self):
        p = _3LS_dimer(J=0.1, K=0.2, L=0.3)
        H = p.electronic_hamiltonian
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_J_coupling_eigenvalues(self):
        """J coupling splits the singly-excited doublet by 2J."""
        e1, J = 1.0, 0.15
        p = _3LS_dimer(e1=e1, J=J)
        e, _ = p.make_manifold_eigensystem(1)
        np.testing.assert_allclose(np.sort(e), [e1 - J, e1 + J], atol=1e-12)

    def test_K_coupling_doubly_excited_structure(self):
        """K coupling produces correct off-diagonal structure in doubly-excited H.

        For a degenerate dimer with only K coupling the doubly-excited 3x3
        Hamiltonian (states |g,f>, |e,e>, |f,g> in Kron order) is:
            [[e2,   K,   0 ],
             [K,  2*e1,  K ],
             [0,    K,  e2 ]]
        """
        e1, e2, K = 1.0, 2.5, 0.2
        p = _3LS_dimer(e1=e1, e2=e2, K=K)
        H2 = p.extract_manifold(p.electronic_hamiltonian, 2)
        expected = np.array([
            [e2,    K,   0.0],
            [K,  2*e1,   K  ],
            [0.0,   K,   e2 ],
        ])
        np.testing.assert_allclose(H2, expected, atol=1e-12)

    def test_L_coupling_triply_excited_structure(self):
        """L (c†c) coupling splits the |e,f>/<|f,e> doublet in manifold 3.

        For a degenerate dimer the triply-excited 2x2 block is:
            [[e1+e2,  L ],
             [L,   e1+e2]]
        with eigenvalues (e1+e2) ± L.
        """
        e1, e2, L = 1.0, 2.5, 0.3
        p = _3LS_dimer(e1=e1, e2=e2, L=L)
        H3 = p.extract_manifold(p.electronic_hamiltonian, 3)
        assert H3.shape == (2, 2), f"Expected 2x2, got {H3.shape}"
        expected = np.array([[e1 + e2, L], [L, e1 + e2]])
        np.testing.assert_allclose(H3, expected, atol=1e-12)

    def test_L_coupling_eigenvalues(self):
        e1, e2, L = 1.0, 2.5, 0.3
        p = _3LS_dimer(e1=e1, e2=e2, L=L)
        e, _ = p.make_manifold_eigensystem(3)
        np.testing.assert_allclose(
            np.sort(e), [e1 + e2 - L, e1 + e2 + L], atol=1e-12
        )

    def test_no_K1_coupling_in_doubly_excited(self):
        """Documents that b†b (K1) is NOT in the existing 3LS code.

        With e2 != 2*e1, |g,f> and |f,g> are degenerate at e2 but
        |e,e> is at 2*e1.  Without K1 the |g,f>/<|f,g> pair must be
        uncoupled (off-diagonal element H2[0,2] = 0).
        """
        e1, e2 = 1.0, 2.5          # 2*e1 = 2.0 ≠ e2 = 2.5
        p = _3LS_dimer(e1=e1, e2=e2)   # no couplings
        H2 = p.extract_manifold(p.electronic_hamiltonian, 2)
        # States: |g,f>(0), |e,e>(1), |f,g>(2)
        # K1 would appear in H2[0,2] and H2[2,0]
        np.testing.assert_allclose(H2[0, 2], 0.0, atol=1e-12,
            err_msg="K1 coupling unexpectedly present in existing 3LS code")
        np.testing.assert_allclose(H2[2, 0], 0.0, atol=1e-12)

    def test_dipole_hermitian(self):
        p = _3LS_dimer(J=0.1, K=0.1, L=0.1)
        for pol in ['x', 'y', 'z']:
            mu = p.mu_dict[pol]
            np.testing.assert_allclose(mu, mu.conj().T, atol=1e-12,
                err_msg=f"mu_{pol} is not Hermitian")

    def test_dipole_up_connects_correct_manifolds(self):
        """mu_up (raising) has non-zero elements only for n→n+1 transitions."""
        e1, e2 = 1.0, 2.5
        p = _3LS_single(e1=e1, e2=e2,
                        d01=[1, 0, 0], d12=[0, 1, 0], d02=[0, 0, 0])
        mu_up_x = p.mu_up_dict['x']
        mu_up_y = p.mu_up_dict['y']

        m01_x = p.extract_coherence(mu_up_x, 1, 0)   # |e> <- |g|, x-pol
        m12_y = p.extract_coherence(mu_up_y, 2, 1)   # |f> <- |e|, y-pol
        m02_x = p.extract_coherence(mu_up_x, 2, 0)   # direct, should be 0

        np.testing.assert_allclose(m01_x, [[1.0]], atol=1e-12)
        np.testing.assert_allclose(m12_y, [[1.0]], atol=1e-12)
        np.testing.assert_allclose(m02_x, [[0.0]], atol=1e-12)


# ---------------------------------------------------------------------------
# 3LS K1 (b†b) coupling tests – FAIL until K1 is implemented
# ---------------------------------------------------------------------------

class Test3LSCouplingK1:
    """Tests for the new K1 (b†b) coupling to be added to 3LS.

    These tests will fail until the K1 coupling is implemented.
    After implementation they serve as regression tests alongside Test3LS.
    """

    def _dimer_with_K1(self, K1, e1=1.0, e2=2.5):
        """3LS dimer using new dict coupling format with only K1 nonzero."""
        site_energies = [[e1, e2], [e1, e2]]
        site_couplings = {
            'J':  [[0, 0.0], [0.0, 0]],
            'K':  [[0, 0.0], [0.0, 0]],
            'K1': [[0, K1 ], [K1,  0]],
            'L':  [[0, 0.0], [0.0, 0]],
        }
        dipoles = np.array([
            [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
        ])
        return Polymer(site_energies, site_couplings, dipoles)

    def test_dict_coupling_accepted_for_3LS(self):
        """Polymer accepts dict coupling input for 3LS."""
        p = self._dimer_with_K1(K1=0.1)
        assert p.N == 3

    def test_K1_doubly_excited_structure(self):
        """K1 coupling adds H2[0,2] = H2[2,0] = K1 (|g,f> <-> |f,g>).

        With only K1, the doubly-excited 3x3 block is:
            [[e2,   0,   K1],
             [0,  2*e1,  0 ],
             [K1,  0,   e2 ]]
        """
        e1, e2, K1 = 1.0, 2.5, 0.15
        p = self._dimer_with_K1(K1=K1, e1=e1, e2=e2)
        H2 = p.extract_manifold(p.electronic_hamiltonian, 2)
        expected = np.array([
            [e2,    0.0,  K1 ],
            [0.0, 2*e1,   0.0],
            [K1,   0.0,   e2 ],
        ])
        np.testing.assert_allclose(H2, expected, atol=1e-12)

    def test_K1_eigenvalues(self):
        """K1 splits |g,f>/<|f,g> by 2*K1; |e,e> is unaffected."""
        e1, e2, K1 = 1.0, 2.5, 0.15
        p = self._dimer_with_K1(K1=K1, e1=e1, e2=e2)
        e, _ = p.make_manifold_eigensystem(2)
        expected = np.sort([2 * e1, e2 - K1, e2 + K1])
        np.testing.assert_allclose(np.sort(e), expected, atol=1e-12)

    def test_old_list_format_unchanged(self):
        """Old [J, K, L] list format still works; K1 defaults to zero."""
        p = _3LS_dimer(e1=1.0, e2=2.5, J=0.1, K=0.2, L=0.3)
        H2 = p.extract_manifold(p.electronic_hamiltonian, 2)
        # K1 term is H2[0,2]; must still be zero
        np.testing.assert_allclose(H2[0, 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(H2[2, 0], 0.0, atol=1e-12)

    def test_extended_list_format(self):
        """[J, K, L, K1] 4-element list format sets K1."""
        e1, e2, K1 = 1.0, 2.5, 0.15
        site_energies = [[e1, e2], [e1, e2]]
        site_couplings = [
            [[0, 0.0], [0.0, 0]],   # J
            [[0, 0.0], [0.0, 0]],   # K
            [[0, 0.0], [0.0, 0]],   # L
            [[0, K1 ], [K1,  0]],   # K1
        ]
        dipoles = np.array([
            [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
        ])
        p = Polymer(site_energies, site_couplings, dipoles)
        H2 = p.extract_manifold(p.electronic_hamiltonian, 2)
        np.testing.assert_allclose(H2[0, 2], K1, atol=1e-12)
        np.testing.assert_allclose(H2[2, 0], K1, atol=1e-12)


# ---------------------------------------------------------------------------
# 4LS tests – FAIL until 4LS is implemented
# ---------------------------------------------------------------------------

class Test4LS:
    """Specification tests for the new FourLevelPolymer / 4LS support.

    These tests will fail until 4LS is implemented and then serve as
    regression tests.
    """

    # --- detection and basic structure ---

    def test_detection(self):
        """N=4 when site_energies elements have length 3."""
        p = _4LS_single()
        assert p.N == 4

    def test_single_site_hilbert_dim(self):
        p = _4LS_single()
        assert p.H_dim == 4

    def test_dimer_hilbert_dim(self):
        """2-site 4LS: 4^2 = 16 states, all have total occupation ≤ 6."""
        p = _4LS_dimer()
        assert p.H_dim == 16

    def test_maximum_manifold_auto(self):
        """Auto maximum_manifold for 4LS is 3 * num_sites."""
        p = _4LS_dimer()          # 2 sites
        assert p.maximum_manifold == 6

    # --- single-site Hamiltonian ---

    def test_single_site_hamiltonian(self):
        """H is diagonal with energies [0, e1, e2, e3]."""
        e1, e2, e3 = 1.0, 2.5, 5.0
        p = _4LS_single(e1=e1, e2=e2, e3=e3)
        expected = np.diag([0.0, e1, e2, e3])
        np.testing.assert_allclose(p.electronic_hamiltonian, expected, atol=1e-12)

    # --- Hamiltonian is Hermitian ---

    def test_hamiltonian_hermitian_all_couplings(self):
        """H is Hermitian when all coupling types are nonzero."""
        couplings = {
            'J':  [[0, 0.10], [0.10, 0]],
            'K':  [[0, 0.09], [0.09, 0]],
            'K1': [[0, 0.08], [0.08, 0]],
            'L':  [[0, 0.07], [0.07, 0]],
            'L1': [[0, 0.06], [0.06, 0]],
            'L2': [[0, 0.05], [0.05, 0]],
            'L3': [[0, 0.04], [0.04, 0]],
            'M0': [[0, 0.03], [0.03, 0]],
            'M1': [[0, 0.02], [0.02, 0]],
            'N0': [[0, 0.01], [0.01, 0]],
        }
        p = _4LS_dimer(couplings=couplings)
        H = p.electronic_hamiltonian
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    # --- J coupling (singly excited, same physics as 3LS) ---

    def test_J_coupling_eigenvalues(self):
        """J coupling splits the singly-excited doublet by 2J."""
        e1, J = 1.0, 0.15
        p = _4LS_dimer(couplings={'J': [[0, J], [J, 0]]})
        e, _ = p.make_manifold_eigensystem(1)
        np.testing.assert_allclose(np.sort(e), [e1 - J, e1 + J], atol=1e-12)

    # --- K1 coupling (doubly excited, new to 4LS via 3LS extension) ---

    def test_K1_default_zero(self):
        """No K1 entry in couplings → H2[|g,f>, |f,g>] = 0."""
        e1, e2 = 1.0, 2.5
        p = _4LS_dimer()
        H2 = p.extract_manifold(p.electronic_hamiltonian, 2)
        # doubly-excited manifold must be diagonal for zero couplings
        np.testing.assert_allclose(H2, np.diag(np.diag(H2)), atol=1e-12)

    def test_K1_coupling_eigenvalues(self):
        """K1 (b†b) splits |g,f>/<|f,g> by 2*K1; |e,e> unchanged."""
        e1, e2, K1 = 1.0, 2.5, 0.15
        p = _4LS_dimer(e1=e1, e2=e2, couplings={'K1': [[0, K1], [K1, 0]]})
        e, _ = p.make_manifold_eigensystem(2)
        expected = np.sort([2 * e1, e2 - K1, e2 + K1])
        np.testing.assert_allclose(np.sort(e), expected, atol=1e-12)

    # --- L coupling (triply excited, same as 3LS c†c) ---

    def test_L_coupling_triply_excited(self):
        """L (c†c) splits |e,f>/<|f,e> by 2L; |g,h>/<|h,g> unchanged."""
        e1, e2, e3, L = 1.0, 2.5, 5.0, 0.2
        # e1+e2 = 3.5 ≠ e3 = 5.0, so the two pairs are well separated
        p = _4LS_dimer(e1=e1, e2=e2, e3=e3, couplings={'L': [[0, L], [L, 0]]})
        e, _ = p.make_manifold_eigensystem(3)
        expected = np.sort([e1+e2-L, e1+e2+L, e3, e3])
        np.testing.assert_allclose(np.sort(e), expected, atol=1e-12)

    # --- L3 coupling (d†d, triply excited, new for 4LS) ---

    def test_L3_coupling_triply_excited(self):
        """L3 (d†d) splits |g,h>/<|h,g> by 2*L3; |e,f>/<|f,e> unchanged."""
        e1, e2, e3, L3 = 1.0, 2.5, 5.0, 0.2
        p = _4LS_dimer(e1=e1, e2=e2, e3=e3, couplings={'L3': [[0, L3], [L3, 0]]})
        e, _ = p.make_manifold_eigensystem(3)
        expected = np.sort([e1+e2, e1+e2, e3-L3, e3+L3])
        np.testing.assert_allclose(np.sort(e), expected, atol=1e-12)

    # --- M1 coupling (p†p, quadruply excited) ---

    def test_M1_coupling_quadruply_excited(self):
        """M1 (p†p) splits |e,h>/<|h,e> by 2*M1.

        |e,h> and |h,e> both have energy e1+e3.
        With e1=1, e2=2.5, e3=5: e1+e3=6, e2+e2=5, e1+e3 ≠ 2*e2.
        M1 only couples the |e,h>/<|h,e> pair.
        """
        e1, e2, e3, M1 = 1.0, 2.5, 5.0, 0.12
        p = _4LS_dimer(e1=e1, e2=e2, e3=e3, couplings={'M1': [[0, M1], [M1, 0]]})
        e, _ = p.make_manifold_eigensystem(4)
        e = np.sort(e)
        # Find the pair that splits: eigenvalues near e1+e3 should be
        # (e1+e3)-M1 and (e1+e3)+M1; the pair near 2*e2 should be unsplit
        idx_low  = np.argmin(np.abs(e - (e1 + e3 - M1)))
        idx_high = np.argmin(np.abs(e - (e1 + e3 + M1)))
        np.testing.assert_allclose(e[idx_low],  e1 + e3 - M1, atol=1e-12)
        np.testing.assert_allclose(e[idx_high], e1 + e3 + M1, atol=1e-12)

    # --- N0 coupling (q†q, quintuply excited) ---

    def test_N0_coupling_quintuply_excited(self):
        """N0 (q†q) splits |f,h>/<|h,f> by 2*N0."""
        e1, e2, e3, N0 = 1.0, 2.5, 5.0, 0.08
        p = _4LS_dimer(e1=e1, e2=e2, e3=e3, couplings={'N0': [[0, N0], [N0, 0]]})
        e, _ = p.make_manifold_eigensystem(5)
        # Only states in manifold 5 are |f,h> (e2+e3=7.5) and |h,f> (e3+e2=7.5)
        expected = np.sort([e2 + e3 - N0, e2 + e3 + N0])
        np.testing.assert_allclose(np.sort(e), expected, atol=1e-12)

    # --- eigensystem completeness ---

    def test_eigensystem_covers_all_manifolds(self):
        """electronic_eigenvectors covers all manifolds up to maximum_manifold."""
        p = _4LS_single()
        # For single site, maximum_manifold = 3; eigenvectors should be 4x4 identity
        V = p.electronic_eigenvectors
        assert V.shape == (4, 4)
        np.testing.assert_allclose(V @ V.T, np.eye(4), atol=1e-12)

    # --- dipole operators ---

    def test_dipole_hermitian(self):
        p = _4LS_single()
        for pol in ['x', 'y', 'z']:
            mu = p.mu_dict[pol]
            np.testing.assert_allclose(mu, mu.conj().T, atol=1e-12,
                err_msg=f"mu_{pol} not Hermitian")

    def test_dipole_ladder_shortcut(self):
        """3-vector (ladder-only) input → direct transitions d_gf, d_gh, d_eh are zero."""
        p = _4LS_single()
        mu_up = p.mu_up_dict['x']

        # Ladder transitions should be nonzero (all d=1 x-polarised)
        m01 = p.extract_coherence(mu_up, 1, 0)   # e <- g
        m12 = p.extract_coherence(mu_up, 2, 1)   # f <- e
        m23 = p.extract_coherence(mu_up, 3, 2)   # h <- f
        np.testing.assert_allclose(np.abs(m01[0, 0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(m12[0, 0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(m23[0, 0]), 1.0, atol=1e-12)

        # Direct transitions should be zero
        m02 = p.extract_coherence(mu_up, 2, 0)   # f <- g  (d_gf)
        m03 = p.extract_coherence(mu_up, 3, 0)   # h <- g  (d_gh)
        m13 = p.extract_coherence(mu_up, 3, 1)   # h <- e  (d_eh)
        np.testing.assert_allclose(m02, [[0.0]], atol=1e-12)
        np.testing.assert_allclose(m03, [[0.0]], atol=1e-12)
        np.testing.assert_allclose(m13, [[0.0]], atol=1e-12)

    def test_dipole_full_six_transitions(self):
        """All 6 per-site dipole transitions can be set independently.

        Dipole array shape (1, 6, 3):
          [0] g<->e  x-only
          [1] e<->f  y-only
          [2] f<->h  z-only
          [3] g<->f  x, magnitude 0.5
          [4] g<->h  x, magnitude 0.3
          [5] e<->h  x, magnitude 0.2
        """
        dipoles = np.array([[
            [1.0, 0,   0],   # d_ge
            [0,   1.0, 0],   # d_ef
            [0,   0,   1.0], # d_fh
            [0.5, 0,   0],   # d_gf
            [0.3, 0,   0],   # d_gh
            [0.2, 0,   0],   # d_eh
        ]])
        p = Polymer([[1.0, 2.5, 5.0]], {}, dipoles)

        mux = p.mu_up_dict['x']
        muy = p.mu_up_dict['y']
        muz = p.mu_up_dict['z']

        np.testing.assert_allclose(np.abs(p.extract_coherence(mux, 1, 0)[0,0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(p.extract_coherence(muy, 2, 1)[0,0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(p.extract_coherence(muz, 3, 2)[0,0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(p.extract_coherence(mux, 2, 0)[0,0]), 0.5, atol=1e-12)
        np.testing.assert_allclose(np.abs(p.extract_coherence(mux, 3, 0)[0,0]), 0.3, atol=1e-12)
        np.testing.assert_allclose(np.abs(p.extract_coherence(mux, 3, 1)[0,0]), 0.2, atol=1e-12)

    # --- make_4LS convenience function ---

    def test_make_4LS_runs(self):
        """make_4LS() creates the output files without error (requires tmp dir)."""
        import tempfile
        from ufss.HLG.make_simple_systems import make_4LS
        with tempfile.TemporaryDirectory() as tmpdir:
            make_4LS(tmpdir, omega0=1.0, tau_deph=100.0, kT=0.1)
            assert os.path.exists(os.path.join(tmpdir, 'closed', 'H.npz'))
            assert os.path.exists(os.path.join(tmpdir, 'closed', 'mu.npz'))
