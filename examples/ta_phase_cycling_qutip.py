"""
Non-perturbative transient-absorption (TA) signal with phase cycling in QuTiP 5.

Two-level system in the rotating frame (RWA). The same structure works for
multilevel systems if sp/sm are replaced by the raising/lowering parts of the
dipole operator, mu_plus and mu_minus = mu_plus.dag().

How the phase cycling works (in the RWA):
  Every sigma_+ interaction with pulse j carries e^{+i phi_j} and every
  sigma_- interaction carries e^{-i phi_j}. The detected polarization
  <mu_minus> is a one-quantum coherence, so the net phase signature of any
  contribution satisfies n_pu + n_pr = +1.
  The TA signal is emitted along the probe with the pump phase cancelled:
  n_pu = 0, n_pr = +1. Because of the constraint above, cycling ONLY the pump
  phase and keeping n_pu = 0 is enough.
  With N equally spaced pump phases, the n_pu = 0 component is the average
  over phases. Contributions with n_pu = +-N, +-2N, ... alias into it, so
  N = 4 rejects everything up to |n_pu| = 3.
  The n_pu = 0 component also contains the pump-free (linear) probe response,
  which is removed by subtracting a pump-off run.

Without the RWA the constraint n_pu + n_pr = 1 no longer holds, so you would
also need to cycle the probe phase and pick n_pr = +1.
"""
import numpy as np
import qutip as qt

# ---------------------------------------------------------------- system
# QuTiP convention: basis(2,0) = excited, basis(2,1) = ground
sp, sm = qt.sigmap(), qt.sigmam()
delta = 0.0            # detuning of transition from carrier (rotating frame)
mu = 1.0
gamma1, gamma_phi = 0.05, 0.1
H0 = 0.5 * delta * qt.sigmaz()
c_ops = [np.sqrt(gamma1) * sm, np.sqrt(gamma_phi / 2) * qt.sigmaz()]
rho0 = qt.ket2dm(qt.basis(2, 1))

sigma_pu, sigma_pr = 1.0, 1.0        # Gaussian pulse widths
t_start = -6 * sigma_pu              # pump centered at t = 0
t_after = 150.0                      # how long to follow the polarization after the probe
dt = 0.05

# ODE settings: max_step stops the solver from stepping over pulses, and tight
# tolerances matter because the TA signal is a small difference of large numbers.
opts = {"atol": 1e-12, "rtol": 1e-10, "max_step": 0.2 * min(sigma_pu, sigma_pr),
        "nsteps": 10**6}


def gauss(t, t0, s):
    return np.exp(-(t - t0) ** 2 / (2 * s ** 2))


def Eplus(t, pulses):
    """Positive-frequency envelope in the rotating frame (sum over pulses)."""
    return sum(A * gauss(t, t0, s) * np.exp(1j * phi)
               for (t0, phi, s, A) in pulses)


def Eminus(t, pulses):
    return np.conj(Eplus(t, pulses))


H = [H0, [-0.5 * mu * sp, Eplus], [-0.5 * mu * sm, Eminus]]


def polarization(pulses, tlist):
    """Complex polarization P(t) = mu <sigma_-> for a given pulse list."""
    res = qt.mesolve(H, rho0, tlist, c_ops, e_ops=[mu * sm],
                     args={"pulses": pulses}, options=opts)
    return np.asarray(res.expect[0])


def ta_polarization(T, A_pu, A_pr, n_phi=4, phi_pr=0.0):
    """
    Phase-cycled pump-induced polarization Delta P(t) at pump-probe delay T.
    Returns (tlist, dP, E_probe(t)).
    """
    tlist = np.arange(t_start, T + t_after, dt)
    probe = (T, phi_pr, sigma_pr, A_pr)

    # pump-off reference: the linear probe response (n_pu = 0, zeroth order in the pump)
    P_off = polarization([probe], tlist)

    # average over equally spaced pump phases -> n_pu = 0 component
    P_sum = np.zeros_like(P_off)
    for k in range(n_phi):
        phi_pu = 2 * np.pi * k / n_phi
        P_sum += polarization([(0.0, phi_pu, sigma_pu, A_pu), probe], tlist)
    dP = P_sum / n_phi - P_off

    E_pr = A_pr * gauss(tlist, T, sigma_pr) * np.exp(1j * phi_pr)
    return tlist, dP, E_pr


def spectrum(tlist, P, E):
    """
    Heterodyne-detected spectrum S(w) ~ Im[E*(w) P(w)] (w relative to the
    carrier). Up to an overall positive prefactor, positive S means extra
    absorption, so the TA signal Delta A(w) is S computed with dP.
    Check the sign convention once against a linear absorption run
    (pump off), which must come out positive.
    """
    Nt = len(tlist)
    w = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nt, d=dt))
    # FT convention f(w) = int f(t) e^{+i w t} dt (matches e^{-i w t} time dependence)
    Pw = np.fft.fftshift(np.fft.ifft(P)) * Nt * dt
    Ew = np.fft.fftshift(np.fft.ifft(E)) * Nt * dt
    return w, np.imag(np.conj(Ew) * Pw)


def order_decomposition(T, A_pr, amps, n_phi=4):
    """
    Separate pump orders at fixed (weak) probe. With n_pu = 0 the pump enters
    as |A_pu|^2, |A_pu|^4, ..., so fit
        Delta P(A) = A^2 c3 + A^4 c5 + A^6 c7 + ...
    using len(amps) amplitudes. c3 is the chi(3) TA signal to compare with
    UFSS, c5 is the fifth-order correction, etc.
    """
    dPs = []
    for A in amps:
        tlist, dP, E_pr = ta_polarization(T, A, A_pr, n_phi)
        dPs.append(dP)
    dPs = np.array(dPs)
    amps = np.asarray(amps)
    V = np.vstack([amps ** (2 * (m + 1)) for m in range(len(amps))]).T   # Vandermonde in A^2
    coeffs = np.linalg.solve(V, dPs)          # rows: c3, c5, c7, ...
    return tlist, coeffs, E_pr


if __name__ == "__main__":
    T = 10.0
    A_pr = 1e-3                       # weak probe -> linear in the probe

    # sanity check: linear absorption (pump off) must be positive
    tlist = np.arange(t_start, T + t_after, dt)
    P_lin = polarization([(T, 0.0, sigma_pr, A_pr)], tlist)
    E_lin = A_pr * gauss(tlist, T, sigma_pr)
    w, S_lin = spectrum(tlist, P_lin, E_lin)
    print("linear absorption at w=0:", S_lin[np.argmin(abs(w))], "(should be > 0)")

    # full non-perturbative TA signal for one pump strength
    tlist, dP, E_pr = ta_polarization(T, A_pu=0.3, A_pr=A_pr)
    w, dA = spectrum(tlist, dP, E_pr)
    print("TA signal at w=0:", dA[np.argmin(abs(w))], "(bleach -> negative)")

    # order-by-order decomposition from three pump amplitudes
    amps = [0.1, 0.2, 0.3]
    tlist, c, E_pr = order_decomposition(T, A_pr, amps)
    for m, cm in enumerate(c):
        w, Sm = spectrum(tlist, cm, E_pr)
        print(f"order {2*m+3} coefficient at w=0:", Sm[np.argmin(abs(w))])
