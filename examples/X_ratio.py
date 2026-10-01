import numpy as np
import numpy.polynomial.hermite as nphe
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os
import yaml
import ufss
from scipy.optimize import minimize

cm_to_eV = 1/8065.54429    # cm^-1 to eV ; easier to just spell cm
ev_to_THz = 1/4.13567E-15/1E12  # self-explanatory

f_0 = 241.8 #in THz
omega_0 = 2*np.pi*f_0
time_unit = 1/omega_0

def make_inds(n,total_num_vib):
    """Makes the indices needed to trim down the number of vibrational states used"""
    inds0 = np.arange(0,n)
    start1 = total_num_vib
    inds1 = np.arange(start1,start1+n*2)
    start2 = total_num_vib*3
    inds2 = np.arange(start2,start2+n*3)
    start3 = total_num_vib*6
    inds3 = np.arange(start3,start3+n*2)
    start4 = total_num_vib*8
    inds4 = np.arange(start4,start4+n)
    return np.hstack((inds0,inds1,inds2,inds3,inds4))

def modify_dimer(mu_cf,mu_bf,params_folder):
    yaml_file = os.path.join(params_folder,'params.yaml')
    with open(yaml_file) as yamlstream:
        params = yaml.load(yamlstream,Loader=yaml.SafeLoader)
    params['dipoles'][0][1][0] = float(mu_cf)
    params['dipoles'][1][1][0] = float(mu_bf)

    with open(yaml_file,'w') as yamlstream:
        yaml.dump(params,yamlstream)

    ufss.HLG.PolymerVibrations(os.path.join(params_folder,'params.yaml'))
    DH = ufss.HLG.DiagonalizeHamiltonian(params_folder)

    inds_dict = {'all_manifolds':make_inds(6,15)}
    DH.trim_manifolds(inds_dict)
    DH.save_eigsystem()
    DH.save_mu()

    ufss.HLG.SecularRedfieldConstructor(params_folder,conserve_memory=True)
    ufss.HLG.DiagonalizeLiouvillian(params_folder,conserve_memory=True,secular=True)

# ---------------------------------------------------------------------------
# Pump pulse definition: a sum of 5 Gaussians (in the frequency domain, each
# with its own amplitude/center/width) plus a quadratic spectral phase
# (chirp). These parameters are shared by the single-shot calculation and by
# the hermite-method convergence sweep below, so they live at module scope
# instead of being redefined inside every function that needs them.
# ---------------------------------------------------------------------------
a1, c1, sigma1 = 0.07, 1.53, 0.04
a2, c2, sigma2 = 0.21, 1.58, 0.06
a3, c3, sigma3 = 0.4, 1.717, 0.055
a4, c4, sigma4 = 0.25, 1.765, 0.03
a5, c5, sigma5 = 0.15, 1.64, 0.05

cpump = 1.65

beta2_fs2 = 36  # chirp parameter in fs^2
beta2 = beta2_fs2 / (time_unit*1000)**2

def gt(t,a,c,s):
    return a * np.exp(-t**2/2*s**2-1j*(c-cpump)*t) * s

def pump_functiont(t):
    return gt(t,a1,c1,sigma1) + gt(t,a2,c2,sigma2) + gt(t,a3,c3,sigma3) + gt(t,a4,c4,sigma4)+ gt(t,a5,c5,sigma5)

def g(w,a,c,s):
    return a * np.exp(-(w-c)**2/(2*s**2))

def pump_function(w):
    return g(w,a1,c1,sigma1) + g(w,a2,c2,sigma2) + g(w,a3,c3,sigma3) + g(w,a4,c4,sigma4)+ g(w,a5,c5,sigma5)

def build_chirped_pump_interpolant(margin):
    """Builds a cubic-spline interpolant of the chirped pump field E(t), valid
        on the window [-margin,margin]. The chirp is applied spectrally (via
        FFT / quadratic spectral phase / IFFT), which requires a uniform time
        grid -- so that's still how the chirped field is generated. But the
        hermite method needs E(t) evaluated *exactly* at the (non-uniformly
        spaced) Gauss-Hermite quadrature nodes for whatever (M,scale) is being
        tried, so a spline interpolant is built once on a fine uniform grid
        and then sampled at arbitrary node positions.

    Args:
        margin (float) : half-width (in time_units) of the uniform grid used
            to compute the chirped field. Must comfortably exceed the extent
            of the largest Gauss-Hermite node set you plan to sample from it
            (node extent grows roughly as scale*sqrt(2*M)).

    Returns:
        spline_re, spline_im (CubicSpline) : real and imaginary parts of the
            chirped pump field, callable at arbitrary t within [-margin,margin]
        chirp_t (ndarray) : the uniform time grid the field was computed on
        Et_chirped (ndarray) : the chirped pump field on that grid (kept
            mainly for diagnostic plotting)
    """
    chirp_dt = 1
    chirp_t = np.arange(-margin,margin+chirp_dt,chirp_dt)
    Et = pump_functiont(chirp_t)

    Ew = np.fft.fft(Et)
    omega = np.fft.fftfreq(len(chirp_t), d=chirp_dt) * 2 * np.pi

    omega0 = 0.0  # center in rotating frame (already offset by cpump)
    chirp_phase = np.exp(1j * beta2 / 2 * (omega - omega0)**2)

    Et_chirped = np.fft.ifft(Ew * chirp_phase)

    spline_re = CubicSpline(chirp_t,Et_chirped.real)
    spline_im = CubicSpline(chirp_t,Et_chirped.imag)
    return spline_re, spline_im, chirp_t, Et_chirped

def hermite_node_extent(M,scale):
    """Roughly how far the Gauss-Hermite nodes for a given (M,scale) extend
        from the center, in the same time units as `scale`."""
    x = nphe.hermgauss(M)[0]
    return scale * np.max(np.abs(x))

def hermite_pump_nodes(M,scale,center,spline_re,spline_im,margin):
    """Returns (t_nodes, pump_nodes): the Gauss-Hermite quadrature node
        positions for the given M/scale/center, and the chirped pump field
        sampled at exactly those positions (via the interpolant built by
        build_chirped_pump_interpolant). The hermite method's HermitePoly fit
        is only exact when the field is evaluated at these specific nodes
        (see containers.py), so t_nodes cannot be an arbitrary grid.
"""
    x = nphe.hermgauss(M)[0]
    t_nodes = center + scale*x
    if t_nodes.min() < -margin or t_nodes.max() > margin:
        raise ValueError(
            'Gauss-Hermite nodes for M={}, scale={} extend to [{:.1f},{:.1f}], '
            'outside the interpolation window [-{:.1f},{:.1f}]. Increase margin '
            'when building the chirped-pump interpolant.'.format(
                M,scale,t_nodes.min(),t_nodes.max(),margin,margin))
    pump_nodes = spline_re(t_nodes) + 1j*spline_im(t_nodes)
    return t_nodes, pump_nodes

def set_hermite_pump(re,nre,M,scale,center,spline_re,spline_im,margin,
                     *,highest_order=5,base_save_names=None,save_tag=''):
    """Configures re/nre to represent the two interacting pump pulses (and
        the impulsive probe) using the hermite method with M Gauss-Hermite
        nodes and field-representation width `scale`, and calls set_efields.
        Must be called (with a fresh M/scale) before every
        ratios_from_signals call in the convergence sweep. Also the function
        that must run *before* set_t (see setup_dimer_engines) since set_t
        sizes the detection grid partly off the pump's own time extent.

        Calls set_highest_order(highest_order) immediately after set_efields,
        every time this function runs. This matters: set_efields rebuilds
        self.pdc from scratch (via set_phase_discrimination -> set_pdc), and
        set_pdc ends with self.set_signal_pdcs([pdc]) -- i.e. it resets the
        higher-order diagram list back down to just the base (3rd-order)
        pdc. So set_highest_order has to be re-run after *every* set_efields
        call, not just once at setup, or the 5th-order diagrams silently
        vanish from self.engine.composite_rhos and S4 comes back as all
        zeros (with ratio_ESAs/ratio_GSBs then coming out as 0/0 = nan
        downstream -- this was the actual cause of the all-nan results).

    Args:
        base_save_names (tuple or None) : (re_base_name, nre_base_name) to
            append save_tag to. Since save_name is built with += elsewhere,
            passing the *base* names here (rather than letting save_tag
            accumulate onto whatever re.save_name/nre.save_name currently is)
            keeps each grid point's save name from picking up every previous
            grid point's tag as this function is called repeatedly in a
            sweep.
"""
    t_nodes, pump_nodes = hermite_pump_nodes(M,scale,center,spline_re,spline_im,margin)

    probe = np.array([1])
    probe_t = np.array([0])
    c = cpump

    for engine_obj in (re,nre):
        engine_obj.engine.method = 'hermite'
        # centers/scales for [pump,pump,probe]; the probe entry is a
        # placeholder (per ufss's own convention for impulsive/size-1
        # pulses -- see general_base_class.set_impulsive_pulses) since a
        # single-point pulse never goes through the hermite fit.
        engine_obj.engine.herm_centers = [center,center,0.0]
        engine_obj.engine.herm_scales = [scale,scale,1.0]
        engine_obj.engine.exp_cutoff = 700

    re.engine.set_efields([t_nodes,t_nodes,probe_t],[pump_nodes,pump_nodes,probe],[c,c,c],re.pdc)
    re.engine.set_polarization_sequence(['x','x','x'])
    nre.engine.set_efields([t_nodes,t_nodes,probe_t],[pump_nodes,pump_nodes,probe],[c,c,c],nre.pdc)
    nre.engine.set_polarization_sequence(['x','x','x'])

    re.set_highest_order(highest_order)
    nre.set_highest_order(highest_order)

    if save_tag and base_save_names is not None:
        re_base,nre_base = base_save_names
        re.save_name = re_base + save_tag
        nre.save_name = nre_base + save_tag

# ---------------------------------------------------------------------------
# Dimer/Hamiltonian setup (expensive: Hamiltonian diagonalization + Redfield/
# Liouvillian construction) is independent of how the pump pulse is
# represented (hermite node count M, hermite width scale). It's split out so
# a convergence sweep over M and scale doesn't have to redo it at every grid
# point.
# ---------------------------------------------------------------------------
def setup_dimer_engines(mu_cf,mu_bf,T,*,M_init,scale_init,center,
                        spline_re,spline_im,margin,params_folder='SQBC_dimer_J300'):
    """Builds the dimer Hamiltonian/Liouvillian (modify_dimer) once, and
        returns Rephasing/NonRephasing engines with the pump field, highest
        perturbative order, detection-time grid, and pulse delays (tau,T)
        already set. Also returns djdi, the pump-and-dipole-weighted e_to_f
        population ratio.

        set_t (called here) sizes the detection time/frequency grid
        (self.t/self.w) using, in part, the pump pulses' own time extent --
        see set_t_general in calculate_signals.py, which computes
        `max_efield_t = max(np.max(u) for u in self.efield_times)`. That
        means set_efields for the pump *must* run before set_t, so this
        function calls set_hermite_pump once up front with (M_init,
        scale_init) -- which should be the widest node configuration you
        intend to use anywhere in a convergence sweep -- purely to fix that
        window's extent. Each sweep point still calls set_hermite_pump again
        with its own M/scale afterward; only self.t/self.w (established
        here, once) stays fixed across the whole sweep, so every point is
        compared on the same frequency axis instead of one that shifts with
        M/scale.

    Args:
        M_init, scale_init (int, float) : hermite node count/width used only
            to seed the initial pump field before set_t runs
        center (float) : hermite center (time_units) for the pump pulses
        spline_re, spline_im (CubicSpline) : chirped pump field interpolant,
            from build_chirped_pump_interpolant
        margin (float) : half-width of the window spline_re/spline_im are
            valid on (must cover M_init/scale_init's node extent)

    Returns:
        re, nre : configured ufss.signals.Rephasing / NonRephasing engines
        tau, T : pulse delay axes, as passed to set_pulse_delays
        djdi : float
"""
    modify_dimer(mu_cf,mu_bf,params_folder)

    open_folder = os.path.join(params_folder,'open')

    re = ufss.signals.Rephasing(open_folder,conserve_memory=True)
    nre = ufss.signals.NonRephasing(open_folder,conserve_memory=True)

    # highest_order is re-established inside set_hermite_pump itself (every
    # time it's called, including here) -- see that function's docstring for
    # why a standalone set_highest_order call here would just get wiped out
    # by the next set_efields call anyway.
    set_hermite_pump(re,nre,M_init,scale_init,center,spline_re,spline_im,margin)

    dt = 4
    gamma_for_dt = 0.01
    re.set_t(gamma_for_dt,dt=dt)
    nre.set_t(gamma_for_dt,dt=dt)

    tau_max = re.engine.t[-1]
    dtau = dt*2
    tau = np.arange(0,tau_max,dtau)
    T = T/time_unit
    re.set_pulse_delays([tau,T])
    nre.set_pulse_delays([tau,T])

    re.save_name += '_custom_pulse6_new_J300_mucf{:.3f}_mubf{:.3f}_many_times'.format(mu_cf,mu_bf)
    nre.save_name += '_custom_pulse6_new_J300_mucf{:.3f}_mubf{:.3f}_many_times'.format(mu_cf,mu_bf)

    evs = re.engine.H_eigenvalues['all_manifolds']
    e_to_f_energies = evs[18:36][:,None]-evs[6:10][None,:]
    e_to_f_pump_weights = np.abs(pump_function(e_to_f_energies))**2
    e_to_f_dipole_weights = np.abs(re.engine.H_mu['up'][18:36,6:10,0])**2
    e_to_f_pump_dipole_weights = e_to_f_pump_weights * e_to_f_dipole_weights
    summed_weights = np.sum(e_to_f_pump_dipole_weights,axis=0)
    di = summed_weights[0]
    dj = summed_weights[1] + summed_weights[3]
    djdi = dj/di

    return re, nre, tau, T, djdi

def compute_S2_S4(re,nre,*,need_S4=True):
    """Runs calculate_signal_all_delays and extracts S2 (3rd-order) and,
        optionally, S4 (5th-order) real-valued signals from re/nre. Pass
        need_S4=False (with the engines' highest_order set to 3, e.g. via
        set_hermite_pump(...,highest_order=3)) to skip the much more
        expensive 5th-order composite-diagram machinery entirely -- this is
        what makes run_s2_convergence cheap.

    Returns:
        S2 (ndarray), S4 (ndarray or None)
"""
    re.calculate_signal_all_delays()
    nre.calculate_signal_all_delays()

    S2r = re.get_signal_order(3).copy()
    S2nr = nre.get_signal_order(3).copy()
    S2 = np.real(S2r + S2nr)

    S4 = None
    if need_S4:
        S4r = re.get_signal_order(5).copy()
        S4nr = nre.get_signal_order(5).copy()
        S4 = np.real(S4r + S4nr)

    return S2, S4

def extract_ratios(re,S2,S4,T):
    """Given precomputed S2 (3rd-order) and S4 (5th-order) real signals for
        the current re/nre configuration, extracts ratio_GSBs, ratio_ESAs,
        and s2_ESA_GSB_ratios. Factored out of the old ratios_from_signals so
        S2 and S4 can be computed once and reused (or, for run_s2_convergence,
        so this can be skipped entirely when only S2 was computed)."""
    wtau = re.wtau + cpump
    wt = re.engine.w + cpump

    wtau_min = 1.5
    wtau_split = 1.615
    wtau_max = 1.8
    wtau_inds = np.where((wtau>wtau_min) & (wtau < wtau_max))
    wtau_inds1 = np.where((wtau>wtau_min) & (wtau < wtau_split))
    wtau_inds2 = np.where((wtau>wtau_split) & (wtau < wtau_max))
    dwtau = wtau[1] - wtau[0]
    wt_min1 = 1.48
    wt_max1 = 1.60
    wt_min2 = 1.65
    wt_max2 = 1.75
    wt_inds1 = np.where((wt>wt_min1) & (wt < wt_max1))[0]
    wt_inds2 = np.where((wt>wt_min2) & (wt < wt_max2))[0]

    ratio_ESAs = np.zeros(T.size)
    ratio_GSBs = np.zeros(T.size)
    s2_ESA_GSB_ratios = np.zeros(T.size)
    for i in range(T.size):
        s2sim_to_plot = np.sum(S2[:,i,wt_inds2],axis=-1)
        s2sim_to_plot = np.abs(s2sim_to_plot)/np.max(np.abs(s2sim_to_plot))
        s4sim_to_plot = np.sum(S4[:,i,wt_inds2],axis=-1)
        s4sim_to_plot = np.abs(s4sim_to_plot)/np.max(np.abs(s4sim_to_plot))
        s2_sim_int_A = np.sum(s2sim_to_plot[wtau_inds1])*dwtau
        s2_sim_int_B = np.sum(s2sim_to_plot[wtau_inds2])*dwtau
        s4_sim_int_A = np.sum(s4sim_to_plot[wtau_inds1])*dwtau
        s4_sim_int_B = np.sum(s4sim_to_plot[wtau_inds2])*dwtau
        ratio2 = s2_sim_int_B/s2_sim_int_A
        ratio4 = s4_sim_int_B/s4_sim_int_A
        ratio_ESAs[i] = ratio4/ratio2

        s2sim_to_plot = np.sum(S2[:,i,wt_inds1],axis=-1)
        s2sim_to_plot = np.abs(s2sim_to_plot)/np.max(np.abs(s2sim_to_plot))
        s4sim_to_plot = np.sum(S4[:,i,wt_inds1],axis=-1)
        s4sim_to_plot = np.abs(s4sim_to_plot)/np.max(np.abs(s4sim_to_plot))
        s2_sim_int_A = np.sum(s2sim_to_plot[wtau_inds1])*dwtau
        s2_sim_int_B = np.sum(s2sim_to_plot[wtau_inds2])*dwtau
        s4_sim_int_A = np.sum(s4sim_to_plot[wtau_inds1])*dwtau
        s4_sim_int_B = np.sum(s4sim_to_plot[wtau_inds2])*dwtau
        ratio2 = s2_sim_int_B/s2_sim_int_A
        ratio4 = s4_sim_int_B/s4_sim_int_A
        ratio_GSBs[i] = ratio4/ratio2

        s2_sim = np.sum(S2[wtau_inds,i,:],axis=0)
        s2_ESA_GSB_ratios[i] = np.max(s2_sim)/np.min(s2_sim)

    return ratio_GSBs, ratio_ESAs, s2_ESA_GSB_ratios

def ratios_from_signals(re,nre,T):
    """Runs the 3rd/5th order signal calculation on the already-configured
        re/nre engines and extracts ratio_GSBs, ratio_ESAs, s2_ESA_GSB_ratios
        from the resulting spectra. This part of the calculation is identical
        regardless of how the pump pulses were represented (UF2/hermite/etc).
        Thin wrapper around compute_S2_S4 + extract_ratios."""
    S2, S4 = compute_S2_S4(re,nre,need_S4=True)
    return extract_ratios(re,S2,S4,T)

# ---------------------------------------------------------------------------
# Ground-truth reference, computed with ufss's default UF2 (DFT-based,
# uniform-grid) method -- i.e. exactly how this script represented the pump
# pulse before it was converted to the hermite method.
#
# This exists because the hermite convergence sweep's own largest-M corner
# turned out NOT to be trustworthy as a reference. Digging into
# hermpts_to_hermcoef (containers.py), the Hermite-coefficient extraction
# normalizes by gamma = 2**n * n! * sqrt(pi) (computed as
# exp(n*log(2)+gammaln(n+1)+0.5*log(pi))). That normalization overflows
# float64 once n (= M-1) gets to roughly 165-175 -- confirmed directly:
# gammaln(150) already gives log(gamma)~=704, right at np.exp's ~709.78
# overflow ceiling, and gammaln(175) is well past it. The raw physicist
# Hermite polynomial values (nphe.hermvander) evaluated at the Gauss-Hermite
# nodes themselves also reach ~1e298 by M=201 -- within a couple orders of
# magnitude of double precision's ~1.8e308 ceiling. Both effects mean the
# fit's coefficients become dominated by catastrophic floating-point
# cancellation well before outright overflow, so *larger* M can silently
# make hermite-method results *less* accurate, not more -- which is exactly
# the non-convergent, order-of-magnitude-jumping behavior you saw. That
# means the sweep can't bootstrap its own ground truth from "use the
# biggest M tried" -- an independent reference is needed instead.
# ---------------------------------------------------------------------------
def get_ratios_uf2(mu_cf,mu_bf,T,*,dt=2,params_folder='SQBC_dimer_J300',
                   highest_order=5):
    """Ground-truth calculation using ufss's default UF2 method (uniform
        pump time grid, FFT-based Heaviside convolution) -- the original,
        pre-hermite way this script represented the pump pulse.

    Keyword Args:
        dt (float) : uniform pump-field time step (time_units), matching the
            pre-hermite version of this script's ef_t = np.arange(-100,101,dt)
        highest_order (int) : set to 3 to skip the 5th-order diagrams
            entirely (cheaper) if you only need S2 as a reference, e.g. for
            run_s2_convergence

    Returns:
        dict with 'ratio_GSBs', 'ratio_ESAs', 's2_ESA_GSB_ratios' (None if
        highest_order<5), 'djdi', 'S2', 'S4' (None if highest_order<5), and
        'detection_t_shape' (re.engine.t.shape -- used by run_s2_convergence
        to confirm its own hermite sweep's detection grid actually matches
        this reference's before comparing S2 arrays elementwise)
"""
    modify_dimer(mu_cf,mu_bf,params_folder)

    open_folder = os.path.join(params_folder,'open')
    re = ufss.signals.Rephasing(open_folder,conserve_memory=True)
    nre = ufss.signals.NonRephasing(open_folder,conserve_memory=True)

    ef_t = np.arange(-100,101,dt)

    chirp_dt = 1
    chirp_t = np.arange(-500,500,chirp_dt)
    Et = pump_functiont(chirp_t)
    Ew = np.fft.fft(Et)
    omega = np.fft.fftfreq(len(chirp_t),d=chirp_dt) * 2 * np.pi
    chirp_phase = np.exp(1j*beta2/2*omega**2)
    ef_vs_t_chirped = np.fft.ifft(Ew*chirp_phase)

    stride = int(round(dt/chirp_dt))
    start_ind = np.argmin(np.abs(chirp_t-ef_t[0]))
    my_slice = slice(start_ind,ef_t.size*stride+start_ind,stride)
    pump = ef_vs_t_chirped[my_slice]

    probe = np.array([1])
    probe_t = np.array([0])
    c = cpump
    t = ef_t

    re.engine.set_efields([t,t,probe_t],[pump,pump,probe],[c,c,c],re.pdc)
    re.engine.set_polarization_sequence(['x','x','x'])
    nre.engine.set_efields([t,t,probe_t],[pump,pump,probe],[c,c,c],nre.pdc)
    nre.engine.set_polarization_sequence(['x','x','x'])

    re.set_highest_order(highest_order)
    nre.set_highest_order(highest_order)

    dt_detection = 4
    gamma_for_dt = 0.01
    re.set_t(gamma_for_dt,dt=dt_detection)
    nre.set_t(gamma_for_dt,dt=dt_detection)

    tau_max = re.engine.t[-1]
    dtau = dt_detection*2
    tau = np.arange(0,tau_max,dtau)
    T_internal = T/time_unit
    re.set_pulse_delays([tau,T_internal])
    nre.set_pulse_delays([tau,T_internal])

    re.save_name += '_UF2ref_mucf{:.3f}_mubf{:.3f}'.format(mu_cf,mu_bf)
    nre.save_name += '_UF2ref_mucf{:.3f}_mubf{:.3f}'.format(mu_cf,mu_bf)

    evs = re.engine.H_eigenvalues['all_manifolds']
    e_to_f_energies = evs[18:36][:,None]-evs[6:10][None,:]
    e_to_f_pump_weights = np.abs(pump_function(e_to_f_energies))**2
    e_to_f_dipole_weights = np.abs(re.engine.H_mu['up'][18:36,6:10,0])**2
    e_to_f_pump_dipole_weights = e_to_f_pump_weights * e_to_f_dipole_weights
    summed_weights = np.sum(e_to_f_pump_dipole_weights,axis=0)
    di = summed_weights[0]
    dj = summed_weights[1] + summed_weights[3]
    djdi = dj/di

    S2, S4 = compute_S2_S4(re,nre,need_S4=(highest_order>=5))
    ratio_GSBs = ratio_ESAs = s2_ESA_GSB_ratios = None
    if S4 is not None:
        ratio_GSBs, ratio_ESAs, s2_ESA_GSB_ratios = extract_ratios(re,S2,S4,T_internal)

    return {'ratio_GSBs':ratio_GSBs,'ratio_ESAs':ratio_ESAs,
            's2_ESA_GSB_ratios':s2_ESA_GSB_ratios,'djdi':djdi,
            'S2':S2,'S4':S4,'detection_t_shape':re.engine.t.shape}

def get_ratios(mu_cf,mu_bf,T,*,M=25,scale=20.0,center=0.0,margin=None,
              params_folder='SQBC_dimer_J300'):
    """Single-shot convenience wrapper: builds the dimer Hamiltonian, sets up
        the pump pulses with the hermite method (M nodes, given scale), and
        returns the same 4 quantities the original UF2-grid version did.
        Rebuilds the Hamiltonian from scratch on every call -- fine for a
        one-off, but see run_hermite_convergence for a sweep over (M,scale)
        that reuses a single Hamiltonian build.

    Args:
        mu_cf, mu_bf (float) : dimer transition dipole parameters
        T (ndarray) : population times (in the same units as the rest of the
            script -- gets internally converted via T/time_unit)

    Keyword Args:
        M (int) : number of Gauss-Hermite nodes used to represent each pump
            pulse
        scale (float) : hermite width parameter (time_units) for the pump
            pulses
        center (float) : hermite center (time_units) for the pump pulses
        margin (float or None) : half-width of the uniform grid used to
            compute the chirped field before resampling onto the Gauss-
            Hermite nodes. Defaults to a value comfortably larger than the
            node extent for the given M/scale.

    Returns:
        ratio_GSBs, ratio_ESAs, s2_ESA_GSB_ratios, djdi
"""
    if margin is None:
        margin = hermite_node_extent(M,scale)*1.3 + 50

    spline_re, spline_im, _, _ = build_chirped_pump_interpolant(margin)

    re, nre, tau, T, djdi = setup_dimer_engines(
        mu_cf,mu_bf,T,M_init=M,scale_init=scale,center=center,
        spline_re=spline_re,spline_im=spline_im,margin=margin,
        params_folder=params_folder)

    set_hermite_pump(re,nre,M,scale,center,spline_re,spline_im,margin,
                     base_save_names=(re.save_name,nre.save_name),
                     save_tag='_hermite_M{}_scale{:.2f}'.format(M,scale))
    ratio_GSBs, ratio_ESAs, s2_ESA_GSB_ratios = ratios_from_signals(re,nre,T)

    return ratio_GSBs, ratio_ESAs, s2_ESA_GSB_ratios, djdi

# ---------------------------------------------------------------------------
# Fast, S2-only convergence check.
#
# S2 (3rd order) needs only 2 pump interactions + 1 probe interaction, versus
# S4 (5th order)'s much larger composite-diagram combination -- so this skips
# set_highest_order(5) entirely (uses 3) and is far cheaper per grid point.
# Run this first: it maps out which (M,scale) region is numerically healthy
# (see get_ratios_uf2's module comment on hermpts_to_hermcoef's float64
# overflow around M~150-175) before paying for the expensive S4 sweep there.
# ---------------------------------------------------------------------------
def run_s2_convergence(mu_cf,mu_bf,T,*,M_values=None,scale_values=None,
                       center=0.0,params_folder='SQBC_dimer_J300',
                       reference=None,make_plot=True,
                       plot_path='hermite_s2_convergence_X_ratio.png'):
    """Sweeps the hermite pump representation over M_values x scale_values,
        computing only S2 (3rd order, highest_order=3) at each point, and
        compares each one by relative L2 norm against a ground-truth S2
        computed once with ufss's default UF2 method (get_ratios_uf2) --
        not against this sweep's own largest-M point (see get_ratios_uf2's
        module comment for why that corner can't be trusted as ground truth).

    Keyword Args:
        M_values (list of int) : defaults to [9,13,19,27,35,41,55] -- kept
            well under the ~150-175 float64 overflow ceiling in
            hermpts_to_hermcoef's gamma normalization (see get_ratios_uf2).
        reference (dict or None) : output of get_ratios_uf2 to reuse (skips
            recomputing it). If None, computed fresh with highest_order=3.

    Returns:
        dict with 'M_values', 'scale_values', 'l2_err' (shape (len(M_values),
        len(scale_values)) relative L2 error of S2 vs the UF2 reference),
        and 'reference'.
"""
    if M_values is None:
        M_values = [9,13,19,27,35,41,55]
    if scale_values is None:
        component_sigmas = np.array([sigma1,sigma2,sigma3,sigma4,sigma5])
        base = 1/component_sigmas  # ~16.7 to 33.3 time_units
        scale_values = np.round(np.linspace(base.min(),2*base.max(),5),1).tolist()

    if reference is None:
        reference = get_ratios_uf2(mu_cf,mu_bf,T,params_folder=params_folder,
                                   highest_order=3)
    S2_ref = reference['S2']

    max_M = max(M_values)
    max_scale = max(scale_values)
    margin = hermite_node_extent(max_M,max_scale)*1.3 + 50
    spline_re, spline_im, _, _ = build_chirped_pump_interpolant(margin)

    re, nre, tau, T, djdi = setup_dimer_engines(
        mu_cf,mu_bf,T,M_init=max_M,scale_init=max_scale,center=center,
        spline_re=spline_re,spline_im=spline_im,margin=margin,
        params_folder=params_folder)

    if re.engine.t.shape != reference['detection_t_shape']:
        raise ValueError(
            "Detection grid ({}) doesn't match the UF2 reference's ({}) -- "
            "can't compare S2 arrays elementwise. This happens if the "
            "widest (M,scale) here pushes set_t's max_efield_t term above "
            "the gamma-based term (~gamma_res/gamma_for_dt); keep "
            "hermite_node_extent(max(M_values),max(scale_values)) "
            "comfortably below that.".format(re.engine.t.shape,reference['detection_t_shape']))

    l2_err = np.full((len(M_values),len(scale_values)),np.nan)
    base_save_names = (re.save_name,nre.save_name)
    for i,M in enumerate(M_values):
        for j,scale in enumerate(scale_values):
            set_hermite_pump(re,nre,M,scale,center,spline_re,spline_im,margin,
                             highest_order=3,base_save_names=base_save_names,
                             save_tag='_s2conv_M{}_scale{:.2f}'.format(M,scale))
            S2,_ = compute_S2_S4(re,nre,need_S4=False)
            l2_err[i,j] = np.sqrt(np.sum(np.abs(S2-S2_ref)**2)/np.sum(np.abs(S2_ref)**2))
            print('M={:4d}  scale={:6.2f}  S2 L2 relerr vs UF2 = {:.4e}'.format(M,scale,l2_err[i,j]))

    if make_plot:
        fig,ax = plt.subplots()
        for j,scale in enumerate(scale_values):
            ax.semilogy(M_values,l2_err[:,j],marker='o',label='scale={:.1f}'.format(scale))
        ax.set_xlabel('Number of Gauss-Hermite nodes, M')
        ax.set_ylabel('S2 relative L2 error vs UF2 reference')
        ax.legend(fontsize=8)
        ax.set_title('S2-only hermite convergence (fast diagnostic)')
        fig.tight_layout()
        fig.savefig(plot_path)
        print('Saved plot to {}'.format(plot_path))

    return {'M_values':M_values,'scale_values':scale_values,'l2_err':l2_err,
            'reference':reference}

# ---------------------------------------------------------------------------
# Convergence sweep over the number of Gauss-Hermite nodes (M) and the
# hermite width (scale).
#
# Why both need to be swept: the hermite method is exact for a pulse that is
# *exactly* a single Gaussian matched to the chosen scale (see HermitePoly's
# docstring in containers.py and ufss's own test_hermite_convergence.py,
# where scale = sigma*sqrt(2) makes the field exactly the zeroth Hermite
# mode for any M). Here the pump is a sum of 5 Gaussians of different
# widths/centers, further reshaped by chirp -- there is no single scale that
# makes it an exact low-order Hermite series, so both M (how many modes are
# used to represent it) and scale (how the time axis is stretched/compressed
# before fitting those modes) affect accuracy, and neither can be fixed
# analytically the way it could be for a single bare Gaussian.
# ---------------------------------------------------------------------------
def run_hermite_convergence(mu_cf,mu_bf,T,*,M_values=None,scale_values=None,
                            center=0.0,params_folder='SQBC_dimer_J300',
                            reference=None,make_plot=True,
                            plot_path='hermite_convergence_X_ratio.png'):
    """Builds the dimer Hamiltonian once, then sweeps the hermite pump
        representation over every (M,scale) combination in M_values x
        scale_values, computing ratio_GSBs/ratio_ESAs at each point.
        Compared against a ground-truth reference computed with ufss's
        default UF2 method (get_ratios_uf2) -- NOT against this sweep's own
        largest-M point, since that corner isn't trustworthy (see
        get_ratios_uf2's module comment on hermpts_to_hermcoef's float64
        overflow around M~150-175, which also degrades accuracy well below
        that hard ceiling).

    Keyword Args:
        M_values (list of int) : Gauss-Hermite node counts to try. Defaults
            to [9,13,19,27,35,41,55] -- kept well under the ~150-175 float64
            overflow ceiling in hermpts_to_hermcoef's gamma normalization.
            Consider running run_s2_convergence first (much cheaper) to see
            where accuracy actually plateaus before spending time here.
        scale_values (list of float) : hermite width parameters (time_units)
            to try. Defaults to a handful of values spanning roughly the
            component pulses' own time-domain widths (1/sigma_i for
            sigma1..sigma5, i.e. ~17-33 time_units) up to about double that.
        reference (dict or None) : output of get_ratios_uf2 to reuse (skips
            recomputing it, e.g. if you already ran run_s2_convergence and
            have its reference handy). If None, computed fresh here.

    Returns:
        dict with keys 'M_values', 'scale_values', 'ratio_GSBs', 'ratio_ESAs'
        (each shape (len(M_values), len(scale_values), T.size)), 'djdi', and
        'reference' (the get_ratios_uf2 dict used for comparison).
"""
    if M_values is None:
        # M_values = [9,13,19,27,35,41,55]
        M_values = [9,13,19,23]
    if scale_values is None:
        # component_sigmas = np.array([sigma1,sigma2,sigma3,sigma4,sigma5])
        # base = 1/component_sigmas  # ~16.7 to 33.3 time_units
        # scale_values = np.round(np.linspace(base.min(),2*base.max(),5),1).tolist()
        scale_values = [4,6,8,10,15,20]

    if reference is None:
        reference = get_ratios_uf2(mu_cf,mu_bf,T,params_folder=params_folder,
                                   highest_order=5)
    ref_GSBs = reference['ratio_GSBs']
    ref_ESAs = reference['ratio_ESAs']

    max_M = max(M_values)
    max_scale = max(scale_values)
    margin = hermite_node_extent(max_M,max_scale)*1.3 + 50
    spline_re, spline_im, _, _ = build_chirped_pump_interpolant(margin)

    re, nre, tau, T, djdi = setup_dimer_engines(
        mu_cf,mu_bf,T,M_init=max_M,scale_init=max_scale,center=center,
        spline_re=spline_re,spline_im=spline_im,margin=margin,
        params_folder=params_folder)

    if re.engine.t.shape != reference['detection_t_shape']:
        raise ValueError(
            "Detection grid ({}) doesn't match the UF2 reference's ({}) -- "
            "results wouldn't be comparable. Keep "
            "hermite_node_extent(max(M_values),max(scale_values)) "
            "comfortably below the gamma-based detection window "
            "(~gamma_res/gamma_for_dt).".format(re.engine.t.shape,reference['detection_t_shape']))

    ratio_GSBs = np.zeros((len(M_values),len(scale_values),T.size))
    ratio_ESAs = np.zeros((len(M_values),len(scale_values),T.size))

    base_save_names = (re.save_name,nre.save_name)
    for i,M in enumerate(M_values):
        for j,scale in enumerate(scale_values):
            print('M={}, scale={:.2f}'.format(M,scale))
            set_hermite_pump(re,nre,M,scale,center,spline_re,spline_im,margin,
                             base_save_names=base_save_names,
                             save_tag='_hermite_M{}_scale{:.2f}'.format(M,scale))
            g_ratio,e_ratio,_ = ratios_from_signals(re,nre,T)
            ratio_GSBs[i,j,:] = g_ratio
            ratio_ESAs[i,j,:] = e_ratio

    print('\nUF2 reference: ratio_GSBs = {}, ratio_ESAs = {}\n'.format(ref_GSBs,ref_ESAs))
    print('{:>4s}  {:>8s}  {:>14s}  {:>14s}  {:>10s}  {:>10s}'.format(
        'M','scale','ratio_GSBs','ratio_ESAs','relerr_GSB','relerr_ESA'))
    for i,M in enumerate(M_values):
        for j,scale in enumerate(scale_values):
            relerr_GSB = np.max(np.abs(ratio_GSBs[i,j,:]-ref_GSBs)/np.abs(ref_GSBs))
            relerr_ESA = np.max(np.abs(ratio_ESAs[i,j,:]-ref_ESAs)/np.abs(ref_ESAs))
            print('{:>4d}  {:>8.2f}  {:>14.6f}  {:>14.6f}  {:>10.2e}  {:>10.2e}'.format(
                M,scale,ratio_GSBs[i,j,0],ratio_ESAs[i,j,0],relerr_GSB,relerr_ESA))

    if make_plot:
        fig,axes = plt.subplots(1,2,figsize=(11,4.5))
        for j,scale in enumerate(scale_values):
            axes[0].plot(M_values,ratio_GSBs[:,j,0],marker='o',label='scale={:.1f}'.format(scale))
            axes[1].plot(M_values,ratio_ESAs[:,j,0],marker='o',label='scale={:.1f}'.format(scale))
        axes[0].axhline(ref_GSBs[0],color='k',ls='--',lw=1,label='UF2 reference')
        axes[1].axhline(ref_ESAs[0],color='k',ls='--',lw=1,label='UF2 reference')
        axes[0].set_xlabel('Number of Gauss-Hermite nodes, M')
        axes[0].set_ylabel('ratio_GSBs')
        axes[1].set_xlabel('Number of Gauss-Hermite nodes, M')
        axes[1].set_ylabel('ratio_ESAs')
        axes[1].legend(fontsize=8)
        fig.suptitle('Hermite-method convergence: node count M and width scale')
        fig.tight_layout()
        fig.savefig(plot_path)
        print('Saved convergence plot to {}'.format(plot_path))

    return {'M_values':M_values,'scale_values':scale_values,
            'ratio_GSBs':ratio_GSBs,'ratio_ESAs':ratio_ESAs,
            'djdi':djdi,'reference':reference}


if __name__ == '__main__':
    T = np.array([1])

    mu_cf = 0.19
    mu_bf = 4.41

    # Run the cheap S2-only diagnostic first: computes one UF2 ground-truth
    # S2 (highest_order=3, so no expensive 5th-order diagrams on either side
    # of the comparison), then sweeps M/scale against it. Look at
    # hermite_s2_convergence_X_ratio.png for where the relative error
    # actually plateaus before running the full (much more expensive)
    # S4-based sweep below.
    M_values = [9,13,19,23]
    scale_values = [4,6,8,10,15,20]
    s2_results = run_s2_convergence(mu_cf,mu_bf,T,M_values=M_values,scale_values=scale_values)

    print(s2_results)

    # run_hermite_convergence needs its own reference computed with
    # highest_order=5 (s2_results['reference'] only has S2/highest_order=3,
    # so ratio_GSBs/ratio_ESAs there are None) -- it builds that itself when
    # reference=None.
    results = run_hermite_convergence(mu_cf,mu_bf,T,M_values=M_values,scale_values=scale_values)

    print('\nUF2 (ground truth) ratios: ratio_GSBs = {}, ratio_ESAs = {}'.format(
        results['reference']['ratio_GSBs'],results['reference']['ratio_ESAs']))
