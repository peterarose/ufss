import unittest
import numpy as np
import os

from test_uf2 import calculate_spectrum, L2_norm

# Node counts to sweep over. hermite_perturbative_container now stores the
# closed-form result of each pulse interaction exactly (no spline), so there
# is no minimum node count beyond M >= 1; the sweep starts at 5 simply
# because smaller M is too coarse to be interesting.
M_VALUES = [5, 7, 9, 13, 19, 25, 35, 49]


def run_convergence_sweep(M_values=M_VALUES):
    """Runs the Smallwood et al. comparison calculation (see test_uf2.py)
        with method='hermite', sweeping only the Gauss-Hermite node count M
        -- not the pulse width parameter `scale`.

        Why scale doesn't need its own convergence sweep: in this test the
        electric field is g(t,sigma), an exact Gaussian, and HermitePoly
        represents f(x) as exp(-x**2)*P(x) with x = (t-center)/scale. When
        scale is set to sigma*sqrt(2) (as calculate_spectrum does), that
        Gaussian is *exactly* exp(-x**2) -- a bare constant times the
        zeroth Hermite polynomial, P(x) = const -- for any node count M >=
        1. There's no approximation in the pulse's own representation to
        converge, and no better choice of scale to search for: it's fixed
        analytically by the pulse's known width, not fit or tuned.

        What M actually needs to resolve, then, isn't the field itself but
        everything next_order() multiplies onto it before integrating:
        the dipole overlap/eigenvector structure of the system, and in
        particular the oscillatory eigenvalue phase factor exp(i*ev2*t)
        (see UF2_open_core.py), which is not a Gaussian and does need
        enough nodes to be well resolved. M also sets how far the
        Gauss-Hermite nodes extend (roughly +/- scale*sqrt(2M)), which
        needs to comfortably contain the region where the accumulated
        density matrix is still evolving. So a convergence sweep over M
        alone is the right (and only needed) test here.

    Args:
        M_values (list of int) : Gauss-Hermite node counts to try

    Returns:
        M_values (list of int), diffs (list of float) : the same M_values
            passed in, paired with the L2_norm deviation from
            analytical_signal.npy at each one
"""
    analytical_sig_path = os.path.join('fixtures','v_3LS',
                                       'analytical_signal.npy')
    analytical_sig = np.load(analytical_sig_path)

    diffs = []
    for M in M_values:
        sig_ft = calculate_spectrum(method='hermite', M=M)
        diff = L2_norm(-np.pi*sig_ft[:,0,:],analytical_sig[:,0,:])
        diffs.append(diff)

    return list(M_values), diffs


class test_hermite_node_convergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M_values, cls.diffs = run_convergence_sweep()

    def test_converges_as_M_increases(self):
        """Error should shrink (up to small numerical wobble) as more
            Gauss-Hermite nodes are used, with no separate parameter to
            tune -- scale is fixed analytically for all M (see
            run_convergence_sweep's docstring).
"""
        for i in range(len(self.diffs)-1):
            with self.subTest(M_from=self.M_values[i],M_to=self.M_values[i+1]):
                self.assertLessEqual(self.diffs[i+1],self.diffs[i]*1.05)

    def test_converges_below_smallwood_threshold(self):
        """The same 0.007 threshold used in test_uf2.py's UF2/chebyshev/
            hermite comparisons should be reachable well before the
            largest M tried here.
"""
        self.assertLess(self.diffs[-1],0.007)

    def test_modest_M_already_within_threshold(self):
        """A relatively small number of nodes should already suffice,
            demonstrating the practical payoff of the field being exactly
            representable regardless of M: convergence is driven entirely
            by resolving the non-Gaussian parts of the calculation, not by
            resolving the pulse shape.
"""
        M_modest = 19
        self.assertIn(M_modest,self.M_values)
        idx = self.M_values.index(M_modest)
        self.assertLess(self.diffs[idx],0.007)


def main():
    M_values, diffs = run_convergence_sweep()
    print('{:>4s}  {:>12s}'.format('M','L2 diff'))
    for M,diff in zip(M_values,diffs):
        print('{:>4d}  {:>12.6f}'.format(M,diff))

    try:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.semilogy(M_values,diffs,marker='o')
        ax.set_xlabel('Number of Gauss-Hermite nodes, M')
        ax.set_ylabel('L2 deviation from analytical_signal.npy')
        ax.set_title("Hermite-method convergence (scale fixed, not swept)")
        plt.savefig('hermite_convergence.png')
        print("Saved plot to hermite_convergence.png")
    except ImportError:
        pass


if __name__ == '__main__':
    import sys
    if '--sweep' in sys.argv:
        main()
    else:
        unittest.main()
