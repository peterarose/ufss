import numpy as np
import numpy.polynomial.chebyshev as npch
import numpy.polynomial.hermite as nphe
from scipy.interpolate import interp1d as sinterp1d
from scipy.interpolate import PchipInterpolator
from scipy.special import gammaln
from scipy.special import erf as serf

class perturbative_container:
    """This class is used for storing wavefunctions and density matrices
        as 2D arrays, with the second index being time. They are stored in
        the interaction picture, and are typically assumed to be 0 for 
        time values before those specified, and constant for time values
        after those specified.
    """
    def __init__(self,t,f,bool_mask,pulse_number,manifold_key,pdc,t0,*,
                 interp_kind='linear',interp_left_fill=0,simultaneous=1):
        """f can be either a wavefunction or density matrix, both given as
            2D arrays, with the first index the eigen-index, and the second 
            index being the time
        Args:
            t (np.ndarrray) : 1D array of time values
            f (np.ndarrray) : 2D array representing psi or rho
            bool_mask (np.ndarray) : boolean mask of length of the eigenvalues,
                representing which eigenvectors have non-zero amplitude
            pulse_number (int) : most recent pulse interaction
            manifold_key (str) : which manifold does this psi or rho exist in
            pdc (tuple) : partial phase-discrimination condition
            t0 (float) : interaction picture time-zero value
        Keyword Args:
            interp_kind (str) : type of interpolation to use (e.g. linear, cubic, etc.)
                can now also be pchip
            interp_left_fill (float) : value to use for extrapolation to earlier times
            simultaneous (int) : number of simultaneous pulse interactions (only 
                relevant for impulsive calculations)
"""
        self.bool_mask = bool_mask
        self.pulse_number = pulse_number
        self.manifold_key = manifold_key
        self.pdc = pdc
        self.t0 = t0
        self.pdc_tuple = tuple(tuple(pdc[i,:]) for i in range(pdc.shape[0]))
        self.simultaneous = simultaneous
        
        if t.size == 1:
            if simultaneous < 1:
                raise Exception('keyword argument simultaneous must be an integer greater than 0')
            self.impulsive = True
            n, M = f.shape
            self.M = M+2
            self.n = n
            self.t = np.array([-1,0,1],dtype='float') * np.spacing(t[0]) + t[0]
            f_new = np.zeros((n,3),dtype='complex')
            f_new[:,0] = interp_left_fill
            f_new[:,1] = (1 + interp_left_fill)/2 * f[:,0]/simultaneous
            f_new[:,2] = f[:,0]*2**(simultaneous-1)/simultaneous
            self.asymptote = f_new[:,-1]
            self.f = f_new
            
            self._f = f_new

            self.f_interp = self.impulsive_fun(self.asymptote,left_fill = interp_left_fill)
            
        else:
            self.impulsive = False
            self.t = t
            self.f = f
            self._f = self.extend(f,left_fill = interp_left_fill)
            self.f_interp = self.make_interpolant(kind=interp_kind,
                                                 left_fill=interp_left_fill)
        
    def extend(self,f,*,left_fill = 0):
        """Takes the input psi or rho and creates an array that is three times
            bigger, with constant values padding before and after (the fill 
            values for before are specified, those for after are always a 
            constant extrapolation of the final value of f)

        Args:
            f (np.ndarray): 2D array representing psi or rho
        Keyword Args:
            left_fill (float) : constant extrapolation for times before
                the pulse interacted (usually 0)
        """
        n, M = f.shape
        self.M = M
        self.n = n
        new_f = np.zeros((n,3*M),dtype='complex')
        new_f[:,0:M] = left_fill
        new_f[:,M:2*M] = f
        asymptote = f[:,-1]
        self.asymptote = asymptote
        new_f[:,2*M:] = asymptote[:,np.newaxis]

        return new_f

    def make_interpolant(self,*, kind='cubic', left_fill=0):
        """Interpolates density matrix and pads using left_fill to the 
            left and f[-1] to the right if kind = 'pchip', uses Pchip
            interpolation, which is a little slower, but allows for a 
            much smoother interpolation. Use if interpolation artifacts
            are encountered
"""
        if kind == 'pchip':
            pchipr = PchipInterpolator(self.t,np.real(self.f),axis = 1,extrapolate=False)
            pchipi = PchipInterpolator(self.t,np.imag(self.f),axis = 1,extrapolate=False)
            print(kind)
            def f(t):
                tmin = self.t[0]
                tmax = self.t[-1]
                tchip_inds = np.where((t>=tmin) & (t<=tmax))[0]
                t0_inds = np.where(t<tmin)[0]
                t_asymptote_inds = np.where(t>tmax)[0]
                ans = np.zeros((self.n,t.size),dtype='complex')
                ans[:,t0_inds] = 0
                ans[:,tchip_inds] = pchipr(t[tchip_inds]) + 1j *pchipi(t[tchip_inds])
                ans[:,t_asymptote_inds] = self.asymptote[:,None]
                return ans
        else:
            left_fill = np.ones(self.n,dtype='complex')*left_fill
            right_fill = self.f[:,-1]
            f = sinterp1d(self.t,self.f,fill_value = (left_fill,right_fill),
                         assume_sorted=True,bounds_error=False,kind=kind)
        return f

    def impulsive_fun(self,asymptote,left_fill=0):
        if left_fill == 0:
            def f(t):
                zero_val = 0.5**self.simultaneous
                heavi = np.heaviside(t-self.t[1],zero_val)[np.newaxis,:]
                return asymptote[:,np.newaxis] * heavi
        else:
            def f(t):
                try:
                    return asymptote[:,np.newaxis] * np.ones(len(t))[np.newaxis,:]
                except:
                    return asymptote[:,np.newaxis]
        return f

    def __call__(self,t):
        try:
            length = len(t)
            if length == 0:
                return np.array([])
        except TypeError:
            #Assume input must be a number (not a list or an array)
            length = 1
        
        if self.impulsive:
            if length == 1:
                if np.isclose(t,self.t[1],atol=1E-15):
                    return self.f[:,1:2]
                else:
                    return self.f_interp(t)
            else:
                return self.f_interp(t)

        # the following logic is designed to speed up calculations outside of the impulsive limit
        if type(t) is np.ndarray:
            if t.size == 0:
                return np.array([])
            elif t[0] > self.t[-1]:
                if t.size <= self.M:
                    ans = self._f[:,-t.size:]#.copy()
                else:
                    ans = np.ones(t.size,dtype='complex')[np.newaxis,:] * self.asymptote[:,np.newaxis]
            elif t[-1] < self.t[0]:
                if t.size <= self.M:
                    ans = self._f[:,:t.size]#.copy()
                else:
                    ans = np.zeros((self.n,t.size),dtype='complex')
            elif t.size == self.M:
                if np.allclose(t,self.t):
                    ans = self.f#.copy()
                else:
                    ans = self.f_interp(t)
            else:
                ans = self.f_interp(t)
        else:
                ans = self.f_interp(t)
        return ans

    def __getitem__(self,inds):
        return self._f[:,inds]

class ChebPoly:
    def __init__(self,t,f,dom = (-1,1),interp_left_fill = 0,
                 interp_right_fill = 0):
        self.t = t
        self.f = f
        self.dom = dom
        self.interp_left_fill = interp_left_fill
        self.asymptote = np.ones(f.shape[0],dtype='complex') * interp_right_fill
        self.chebpts_to_chebcoef()
        
    def chebpts_to_chebcoef(self):
        """taken from numpy: https://github.com/numpy/numpy/blob/v1.24.0/numpy/polynomial/chebyshev.py#L1780-L1844
        Args:
            yvalues (np.ndarray) : evaluated at chebpts1(order), where 
                order = degree + 1
"""
        self.order = self.t.size
        self.deg = self.order - 1
        self.midpoint = (self.dom[0] + self.dom[1])/2
        self.halfwidth = (self.dom[1] - self.dom[0])/2
        x = (self.t - self.midpoint)/self.halfwidth
        m = npch.chebvander(x, self.deg)
        c = np.dot(m.T, self.f.T)
        c[0] /= self.order
        c[1:] /= 0.5*self.order

        self.cheb_coefs = c.T

        return None

    def f_interp(self,t):
        """Assumes t is sorted
"""
        a,b = self.dom
        if t[0] >= a:
            lowest_ind = 0
        else:
            lowest_ind = np.argmin(np.abs(t-a))
            if t[lowest_ind] < a:
                lowest_ind += 1
        if t[-1] <= b:
            highest_ind = t.size
        else:
            highest_ind = np.argmin(np.abs(t-b))
            if t[highest_ind] < b:
                highest_ind += 1

        ans = np.zeros((self.f.shape[0],t.size),dtype='complex')

        x = (t[lowest_ind:highest_ind] - self.midpoint)/self.halfwidth
        chv = npch.chebvander(x,self.deg)
        ans_interp = chv.dot(self.cheb_coefs.T)
        ans[:,lowest_ind:highest_ind] = ans_interp.T

        if lowest_ind > 0:
            ans[:,:lowest_ind] = self.interp_left_fill

        if highest_ind < t.size:
            high_f = np.ones((t.size - highest_ind),dtype='complex')
            high_f = high_f[np.newaxis,:] * self.asymptote[:,np.newaxis]
            ans[:,highest_ind:] = high_f
            
        return ans

    def integrate(self):
        # scl=self.halfwidth converts the antiderivative from the
        # dimensionless x = (t-midpoint)/halfwidth variable back to
        # physical t (dt = halfwidth*dx); chebint's default scl=1 silently
        # omits this and returns the integral over x instead of over t,
        # off by a factor of halfwidth (confirmed by direct testing against
        # scipy.integrate.quad -- this was a real, pre-existing bug).
        self.cheb_coefs = npch.chebint(self.cheb_coefs,axis=1,
                                       lbnd=-1,scl=self.halfwidth)
        self.order = self.cheb_coefs.shape[1]
        self.deg = self.order - 1

        return None

    def __call__(self,t):
        return self.f_interp(t)

_HERMITE_EVAL_EPS = 1E-17  # relative size below which Hermite-function tails are dropped

def _scaled_hermite(x,deg,log_prefactor):
    """Returns h_n(x) * exp(log_prefactor(x)) for n = 0..deg, as an array of
        shape (deg+1, x.size), where h_n = H_n/sqrt(2**n n! sqrt(pi)) are the
        orthonormalized physicists' Hermite polynomials (so that
        h_n(x)*exp(-x**2/2) are the Hermite functions psi_n(x), all bounded
        by pi**(-1/4)).

        Uses the stable three-term recurrence
            h_0 = pi**(-1/4),  h_1 = sqrt(2) x h_0,
            h_{n+1} = sqrt(2/(n+1)) x h_n - sqrt(n/(n+1)) h_{n-1},
        carrying a per-point log scale so that neither the polynomial growth
        of h_n at large |x| nor exp(log_prefactor) ever overflows or
        underflows on its own -- only their (bounded) product is formed.
        This is what lets degrees in the hundreds be evaluated far from the
        origin without the inf*0 = NaN that hermvander(...)*exp(-x**2) gives.

    Args:
        x (np.ndarray) : 1D array of dimensionless points
        deg (int) : highest degree needed (>= 0)
        log_prefactor (np.ndarray) : same shape as x
"""
    x = np.asarray(x,dtype='float')
    out = np.zeros((deg+1,x.size))
    if x.size == 0:
        return out
    logscale = np.array(log_prefactor,dtype='float',copy=True)
    p_prev = np.zeros(x.size)
    p = np.full(x.size,np.pi**(-0.25))
    big = 1E150
    for n in range(deg+1):
        with np.errstate(over='ignore',under='ignore'):
            out[n] = p * np.exp(logscale)
        if n == deg:
            break
        p_next = np.sqrt(2/(n+1)) * x * p - np.sqrt(n/(n+1)) * p_prev
        p_prev, p = p, p_next
        mag = np.abs(p)
        rescale = mag > big
        if np.any(rescale):
            s = mag[rescale]
            p[rescale] /= s
            p_prev[rescale] /= s
            logscale[rescale] += np.log(s)
    return out

_hermite_projection_cache = {}

def _hermite_projection_matrix(order):
    """For Gauss-Hermite nodes x_k (k = 0..order-1) and weights w_k, returns
        (x, V) with V[n,k] = w_k exp(x_k**2/2) h_n(x_k) = w_k exp(x_k**2)
        psi_n(x_k).

        If f_k = exp(-x_k**2) P(x_k) with P a polynomial of degree < order,
        then b_n = sum_k V[n,k] * (f_k exp(x_k**2/2)) are exactly the
        coefficients of P in the orthonormal basis h_n (Gauss-Hermite
        quadrature is exact for the products involved). Every entry of V is
        bounded (|psi_n| <= pi**(-1/4) and w_k exp(x_k**2) is O(1)), so this
        is well conditioned for any order. Cached per order because
        next_order() calls it for every interaction.
"""
    try:
        return _hermite_projection_cache[order]
    except KeyError:
        pass
    with np.errstate(all='ignore'):
        x = nphe.hermgauss(order)[0]
    # w_k exp(x_k**2) = 1/(order * psi_{order-1}(x_k)**2), computed from the
    # stable recurrence rather than from hermgauss's weights, which
    # overflow/underflow for order >~ 500
    psi = _scaled_hermite(x,order-1,-x**2/2)
    lam = 1/(order * psi[-1]**2)
    V = psi * lam[np.newaxis,:]
    _hermite_projection_cache[order] = (x,V)
    return x, V

class HermitePoly:
    """Infinite-domain analogue of ChebPoly for pulse-localized source terms.

        f(t) is represented, with x = (t - center)/scale, as

            f = exp(-x**2) * sum_{n=0}^{order-1} b_n h_n(x),

        where h_n = H_n/sqrt(2**n n! sqrt(pi)) are orthonormalized
        physicists' Hermite polynomials. f must be sampled at the
        Gauss-Hermite nodes t_k = center + scale*x_k of the same order; the
        coefficients are then recovered exactly (for f of this form) by
        Gauss-Hermite quadrature, see _hermite_projection_matrix. With scale
        = sigma*sqrt(2), a Gaussian pulse exp(-t**2/(2 sigma**2)) is exactly
        exp(-x**2), so the only thing the series has to resolve is whatever
        smooth factor multiplies the pulse (earlier-order density matrix,
        dipole overlaps, eigenvalue phases).

        integrate() gives the causal antiderivative F(t) = int_{-inf}^t f
        in closed form. Using d/dx[exp(-x**2) H_{n-1}] = -exp(-x**2) H_n,

            F = scale * [ b_0 pi**(1/4) Phi(x)
                          - exp(-x**2) sum_{m=0}^{order-2} q_m h_m(x) ],
            q_m = b_{m+1}/sqrt(2(m+1)),     Phi(x) = (1 + erf(x))/2,

        which is exactly 0 at -inf and exactly the total integral
        (self.asymptote = scale*b_0*pi**(1/4)) at +inf. After integrate(),
        self.ramp_amplitude and self.decay_coefs hold the two pieces
        (with the scale factor folded in), which is what
        hermite_perturbative_container stores -- F itself is never
        resampled or interpolated.

        Everything is evaluated through _scaled_hermite, which stays finite
        for any degree and any |x|; and since |exp(-x**2) h_m| =
        exp(-x**2/2)|psi_m| <= exp(-x**2/2) pi**(-1/4), the decaying part is
        only evaluated where that bound exceeds _HERMITE_EVAL_EPS relative
        to the result (typically |x| < ~9 regardless of order), and is set
        to exactly 0 elsewhere.
"""
    def __init__(self,t,f,center = 0.0,scale = 1.0):
        self.t = np.asarray(t,dtype='float')
        f = np.asarray(f)
        if f.ndim == 1:
            f = f[np.newaxis,:]
        self.f = f
        self.center = center
        self.scale = scale
        self.order = self.t.size
        self.integrated = False
        self.hermpts_to_hermcoef()
        self.asymptote = self.scale * self.coefs[:,0] * np.pi**0.25

    def hermpts_to_hermcoef(self):
        """Recovers the coefficients b_n (see class docstring) from f
            sampled at the scaled Gauss-Hermite nodes. f*exp(x**2/2) is
            formed in log space so that it neither overflows (large |x|)
            nor loses f's tiny values to underflow first.
"""
        x_std, V = _hermite_projection_matrix(self.order)
        x = (self.t - self.center)/self.scale
        if not np.allclose(x,x_std,rtol=1E-10,atol=1E-10):
            raise ValueError('HermitePoly requires t to be the Gauss-Hermite nodes center + scale*hermgauss(t.size)[0]')
        tiny = 1E-300
        abs_f = np.abs(self.f)
        phase = self.f / (abs_f + tiny)
        with np.errstate(divide='ignore',over='ignore',under='ignore'):
            g = phase * np.exp(np.log(abs_f + tiny) + x_std[np.newaxis,:]**2/2)
        self.coefs = g.dot(V.T)                     # (n_species, order)
        return None

    def integrate(self):
        """Switches to the closed-form causal antiderivative (class docstring).
"""
        b = self.coefs
        order = b.shape[1]
        m = np.arange(1,order)
        self.ramp_amplitude = self.scale * b[:,0] * np.pi**0.25
        self.decay_coefs = self.scale * b[:,1:] / np.sqrt(2*m)[np.newaxis,:]
        self.asymptote = self.ramp_amplitude.copy()
        self.integrated = True
        return None

    def f_interp(self,t):
        t = np.atleast_1d(np.asarray(t,dtype='float'))
        if self.integrated:
            return _eval_hermite_ramp(t,self.center,self.scale,
                                      self.ramp_amplitude,self.decay_coefs)
        return _eval_hermite_decay(t,self.center,self.scale,self.coefs)

    def __call__(self,t):
        return self.f_interp(t)

def _hermite_window(x,coefs,reference):
    """Boolean mask of the points x where exp(-x**2) sum_m coefs_m h_m(x)
        can exceed _HERMITE_EVAL_EPS*reference (using |exp(-x**2) h_m| <=
        exp(-x**2/2) pi**(-1/4)); the sum is exactly negligible elsewhere.
"""
    csum = np.sum(np.max(np.abs(coefs),axis=0)) if coefs.size else 0.0
    if csum == 0:
        return np.zeros(x.size,dtype='bool')
    ref = max(reference,csum)
    arg = np.log(np.pi**(-0.25)*csum/(_HERMITE_EVAL_EPS*ref))
    x_cut = np.sqrt(2*max(arg,0.0))
    return np.abs(x) <= x_cut

def _eval_hermite_decay(t,center,scale,coefs):
    """exp(-x**2) sum_n coefs[:,n] h_n(x), x = (t-center)/scale."""
    x = (t - center)/scale
    ans = np.zeros((coefs.shape[0],t.size),dtype='complex')
    win = _hermite_window(x,coefs,0.0)
    if np.any(win):
        xw = x[win]
        H = _scaled_hermite(xw,coefs.shape[1]-1,-xw**2)
        ans[:,win] = coefs.dot(H)
    return ans

def _eval_hermite_ramp(t,center,scale,ramp_amplitude,decay_coefs):
    """ramp_amplitude*Phi(x) - exp(-x**2) sum_m decay_coefs[:,m] h_m(x)."""
    x = (t - center)/scale
    n = ramp_amplitude.size
    reference = np.max(np.abs(ramp_amplitude)) if n else 0.0
    win = _hermite_window(x,decay_coefs,reference)
    # Phi saturates to exactly 0/1 (to double precision) for |x| > ~6
    phi = np.where(x > 0,1.0,0.0)
    near = np.abs(x) < 7
    phi[near] = 0.5*(1.0 + serf(x[near]))
    ans = ramp_amplitude[:,np.newaxis] * phi[np.newaxis,:]
    ans = ans.astype('complex')
    if np.any(win) and decay_coefs.shape[1] > 0:
        xw = x[win]
        H = _scaled_hermite(xw,decay_coefs.shape[1]-1,-xw**2)
        ans[:,win] -= decay_coefs.dot(H)
    return ans

class _HermiteRampTerm:
    """One closed-form contribution ramp_amplitude*Phi(x) - exp(-x**2)*Q(x)
        to a hermite_perturbative_container, all sharing one center/scale.
        Rows are the container's species (its bool_mask order).
"""
    def __init__(self,center,scale,ramp_amplitude,decay_coefs):
        self.center = float(center)
        self.scale = float(scale)
        self.ramp_amplitude = np.asarray(ramp_amplitude,dtype='complex')
        self.decay_coefs = np.asarray(decay_coefs,dtype='complex')

    def same_grid(self,other):
        return (np.isclose(self.center,other.center,rtol=0,atol=1E-12*max(1,abs(self.center)))
                and np.isclose(self.scale,other.scale,rtol=1E-12,atol=0))

    def __call__(self,t):
        return _eval_hermite_ramp(t,self.center,self.scale,
                                  self.ramp_amplitude,self.decay_coefs)

    def embedded(self,rows,n_total,factor):
        """Copy scaled by the per-row constant `factor`, scattered into rows
            `rows` of a container with n_total species.
"""
        amp = np.zeros(n_total,dtype='complex')
        amp[rows] = self.ramp_amplitude * factor
        dc = np.zeros((n_total,self.decay_coefs.shape[1]),dtype='complex')
        dc[rows,:] = self.decay_coefs * factor[:,np.newaxis]
        return _HermiteRampTerm(self.center,self.scale,amp,dc)

    def add_inplace(self,other):
        """Adds a term on the same center/scale, padding the lower degree."""
        d1 = self.decay_coefs.shape[1]
        d2 = other.decay_coefs.shape[1]
        if d2 > d1:
            pad = np.zeros((self.decay_coefs.shape[0],d2-d1),dtype='complex')
            self.decay_coefs = np.hstack((self.decay_coefs,pad))
        self.decay_coefs[:,:d2] += other.decay_coefs
        self.ramp_amplitude = self.ramp_amplitude + other.ramp_amplitude

class hermite_perturbative_container:
    """Stores a wavefunction or density matrix (in the interaction picture)
        for method='hermite' *exactly*, as the closed-form expressions that
        HermitePoly.integrate() produces, rather than as samples.

        After one pulse interaction the stored quantity is
            F(t) = A*Phi(x) - exp(-x**2) Q(x),   x = (t - center)/scale,
        which is 0 before the pulse, A after it, and exact in between (an
        earlier version stored F on the Gauss-Hermite nodes and used a cubic
        spline between them, which limited accuracy to ~1E-6 and O(M**-2)
        convergence during the pulse).

        A sum of such objects (add_rhos/add_psis) is kept as a list of
        terms, each with its own center/scale, so density matrices whose
        most recent interactions come from pulses with *different* arrival
        times are still represented exactly; terms that share a center and
        scale are merged by adding coefficients, so the list only grows with
        the number of distinct pulse arrival times. Constant per-species
        factors (e.g. the interaction-picture shift exp(e*(t0 - t0_a)) in
        add_rhos) are folded into the coefficients.

        Three construction modes:
            - from_hermite_poly(hp,...) / combine(...): the exact modes above
            - t.size == 1: impulsive pulse, identical to the previous
              behavior (step function, 'simultaneous' bookkeeping)
            - t.size > 1 with interp_left_fill != 0 and f constant in time:
              a time-independent state such as rho0/psi0
        Summing an impulsive container with a finite one is not supported
        (combine raises a ValueError).
"""
    def __init__(self,t,f,bool_mask,pulse_number,manifold_key,pdc,t0,*,
                 interp_kind=None,interp_left_fill=0,simultaneous=1,
                 center = 0.0, scale = 1.0, terms = None, constant = None):
        """f can be either a wavefunction or density matrix, both given as
            2D arrays, with the first index the eigen-index, and the second
            index being the time. center/scale/interp_kind are accepted for
            backwards compatibility and are not used.
"""
        self.bool_mask = bool_mask
        self.pulse_number = pulse_number
        self.manifold_key = manifold_key
        self.pdc = pdc
        self.t0 = t0
        self.pdc_tuple = tuple(tuple(pdc[i,:]) for i in range(pdc.shape[0]))
        self.simultaneous = simultaneous
        self.interp_left_fill = interp_left_fill
        self.center = center
        self.scale = scale

        if terms is not None or constant is not None:
            self.impulsive = False
            self.terms = list(terms) if terms is not None else []
            if constant is None:
                n = self.terms[0].ramp_amplitude.size
                constant = np.zeros(n,dtype='complex')
            self.constant = np.asarray(constant,dtype='complex')
            self.n = self.constant.size
            self._set_asymptote()
            return None

        t = np.atleast_1d(t)
        f = np.asarray(f)
        if t.size == 1:
            if simultaneous < 1:
                raise Exception('keyword argument simultaneous must be an integer greater than 0')
            self.impulsive = True
            self.order = 1
            n, M = f.shape
            self.M = M+2
            self.n = n
            self.t = np.array([-1,0,1],dtype='float') * np.spacing(t[0]) + t[0]
            f_new = np.zeros((n,3),dtype='complex')
            f_new[:,0] = interp_left_fill
            f_new[:,1] = (1 + interp_left_fill)/2 * f[:,0]/simultaneous
            f_new[:,2] = f[:,0]*2**(simultaneous-1)/simultaneous
            self.asymptote = f_new[:,-1]
            self.f = f_new
            self._f = f_new
            self.f_interp = self.impulsive_fun(self.asymptote,left_fill = interp_left_fill)
            return None

        if interp_left_fill != 0 and np.allclose(f,f[:,:1]):
            self.impulsive = False
            self.terms = []
            self.constant = np.array(f[:,0],dtype='complex')
            self.n = self.constant.size
            self._set_asymptote()
            return None

        raise ValueError('hermite_perturbative_container no longer interpolates '
                         'sampled data; build it with '
                         'hermite_perturbative_container.from_hermite_poly or '
                         '.combine')

    def _set_asymptote(self):
        asym = self.constant.copy()
        for term in self.terms:
            asym = asym + term.ramp_amplitude
        self.asymptote = asym

    @classmethod
    def from_hermite_poly(cls,hp,bool_mask,pulse_number,manifold_key,pdc,t0,*,
                          simultaneous=1):
        """Container holding exactly the causal antiderivative represented
            by an integrated HermitePoly.
"""
        if not hp.integrated:
            hp.integrate()
        term = _HermiteRampTerm(hp.center,hp.scale,hp.ramp_amplitude,
                                hp.decay_coefs)
        return cls(None,None,bool_mask,pulse_number,manifold_key,pdc,t0,
                   simultaneous=simultaneous,terms=[term],
                   constant=np.zeros(hp.ramp_amplitude.size,dtype='complex'))

    @classmethod
    def combine(cls,a,b,factor_a,factor_b,bool_mask,pulse_number,manifold_key,
                pdc,t0):
        """Exact sum factor_a*a + factor_b*b on the union mask bool_mask.

        Args:
            a, b : hermite_perturbative_container (non-impulsive)
            factor_a, factor_b (np.ndarray) : constant per-species factors,
                ordered like a's (b's) species
            bool_mask (np.ndarray) : logical_or of a.bool_mask, b.bool_mask
"""
        for c in (a,b):
            if getattr(c,'impulsive',False) or not hasattr(c,'terms'):
                raise ValueError('Cannot add an impulsive (or non-hermite) '
                                 'container to a finite-pulse hermite '
                                 'container; mixing impulsive and finite '
                                 'pulses within one sum is not supported by '
                                 "method='hermite'")
        union_inds = np.where(bool_mask)[0]
        n_total = union_inds.size
        position = -np.ones(bool_mask.size,dtype='int')
        position[union_inds] = np.arange(n_total)
        terms = []
        constant = np.zeros(n_total,dtype='complex')
        for c, factor in ((a,factor_a),(b,factor_b)):
            factor = np.broadcast_to(np.asarray(factor,dtype='complex'),(c.n,))
            rows = position[np.where(c.bool_mask)[0]]
            constant[rows] += c.constant * factor
            for term in c.terms:
                new_term = term.embedded(rows,n_total,factor)
                for existing in terms:
                    if existing.same_grid(new_term):
                        existing.add_inplace(new_term)
                        break
                else:
                    terms.append(new_term)
        return cls(None,None,bool_mask,pulse_number,manifold_key,pdc,t0,
                   terms=terms,constant=constant)

    def impulsive_fun(self,asymptote,left_fill=0):
        if left_fill == 0:
            def f(t):
                zero_val = 0.5**self.simultaneous
                heavi = np.heaviside(t-self.t[1],zero_val)[np.newaxis,:]
                return asymptote[:,np.newaxis] * heavi
        else:
            def f(t):
                try:
                    return asymptote[:,np.newaxis] * np.ones(len(t))[np.newaxis,:]
                except:
                    return asymptote[:,np.newaxis]
        return f

    def __call__(self,t):
        try:
            length = len(t)
            if length == 0:
                return np.array([])
        except TypeError:
            #Assume input must be a number (not a list or an array)
            length = 1

        if self.impulsive:
            if length == 1:
                if np.isclose(t,self.t[1],atol=1E-15):
                    return self.f[:,1:2]
                else:
                    return self.f_interp(t)
            else:
                return self.f_interp(t)

        t = np.atleast_1d(np.asarray(t,dtype='float'))
        ans = np.zeros((self.n,t.size),dtype='complex')
        ans += self.constant[:,np.newaxis]
        for term in self.terms:
            ans += term(t)
        return ans

class cheb_perturbative_container(ChebPoly):
    def __init__(self,t,f,bool_mask,pulse_number,manifold_key,pdc,t0,*,
                 interp_kind='chebyshev',interp_left_fill=0,simultaneous=1,
                 dom = (-1,1)):
        """f can be either a wavefunction or density matrix, both given as
            2D arrays, with the first index the eigen-index, and the second 
            index being the time. Argument interp_kind is ignored
"""
        self.bool_mask = bool_mask
        self.pulse_number = pulse_number
        self.manifold_key = manifold_key
        self.pdc = pdc
        self.t0 = t0
        self.pdc_tuple = tuple(tuple(pdc[i,:]) for i in range(pdc.shape[0]))
        self.simultaneous = simultaneous
        self.dom = dom
        self.interp_left_fill = interp_left_fill
        
        if t.size == 1:
            if simultaneous < 1:
                raise Exception('keyword argument simultaneous must be an integer greater than 0')
            self.impulsive = True
            n, M = f.shape
            self.M = M+2
            self.n = n
            self.t = np.array([-1,0,1],dtype='float') * np.spacing(t[0]) + t[0]
            f_new = np.zeros((n,3),dtype='complex')
            f_new[:,0] = interp_left_fill
            f_new[:,1] = (1 + interp_left_fill)/2 * f[:,0]/simultaneous
            f_new[:,2] = f[:,0]*2**(simultaneous-1)/simultaneous
            self.asymptote = f_new[:,-1]
            self.f = f_new
            
            self._f = f_new

            self.f_interp = self.impulsive_fun(self.asymptote,left_fill = interp_left_fill)
            
        else:
            self.impulsive = False
            self.t = t
            self.f = f
            self._f = self.extend(f,left_fill = interp_left_fill)
            self.chebpts_to_chebcoef()

    def extend(self,f,*,left_fill = 0):
        n, M = f.shape
        self.M = M
        self.n = n
        new_f = np.zeros((n,3*M),dtype='complex')
        new_f[:,0:M] = left_fill
        new_f[:,M:2*M] = f
        asymptote = f[:,-1]
        self.asymptote = asymptote
        new_f[:,2*M:] = asymptote[:,np.newaxis]

        return new_f

    def impulsive_fun(self,asymptote,left_fill=0):
        if left_fill == 0:
            def f(t):
                zero_val = 0.5**self.simultaneous
                heavi = np.heaviside(t-self.t[1],zero_val)[np.newaxis,:]
                return asymptote[:,np.newaxis] * heavi
        else:
            def f(t):
                try:
                    return asymptote[:,np.newaxis] * np.ones(len(t))[np.newaxis,:]
                except:
                    return asymptote[:,np.newaxis]
        return f

    def __call__(self,t):
        try:
            length = len(t)
            if length == 0:
                return np.array([])
        except TypeError:
            #Assume input must be a number (not a list or an array)
            length = 1
        
        if self.impulsive:
            if length == 1:
                if np.isclose(t,self.t[1],atol=1E-15):
                    return self.f[:,1:2]
                else:
                    return self.f_interp(t)
            else:
                return self.f_interp(t)

        # the following logic is designed to speed up calculations outside of the impulsive limit
        if type(t) is np.ndarray:
            if t.size == 0:
                return np.array([])
            elif t[0] > self.t[-1]:
                if t.size <= self.M:
                    ans = self._f[:,-t.size:]#.copy()
                else:
                    ans = np.ones(t.size,dtype='complex')[np.newaxis,:] * self.asymptote[:,np.newaxis]
            elif t[-1] < self.t[0]:
                if t.size <= self.M:
                    ans = self._f[:,:t.size]#.copy()
                else:
                    ans = np.zeros((self.n,t.size),dtype='complex')
            elif t.size == self.M:
                if np.allclose(t,self.t):
                    ans = self.f#.copy()
                else:
                    ans = self.f_interp(t)
            else:
                ans = self.f_interp(t)
        else:
                ans = self.f_interp(t)
        return ans

class RK_perturbative_container:
    def __init__(self,t,f,pulse_number,manifold_key,pdc,*,
                 interp_kind='linear',interp_left_fill=0,simultaneous=1):
        self.pulse_number = pulse_number
        self.manifold_key = manifold_key
        self.pdc = pdc
        self.pdc_tuple = tuple(tuple(pdc[i,:]) for i in range(pdc.shape[0]))
        self.simultaneous = simultaneous

        self.n, self.M = f.shape
        if t.size == 1:
            if simultaneous < 1:
                raise Exception('keyword argument simultaneous must be an integer greater than 0')
            self.impulsive = True
            self.M = self.M+2
            self.t = np.array([-1,0,1],dtype='float') * np.spacing(t[0]) + t[0]
            f_new = np.zeros((self.n,3),dtype='complex')
            f_new[:,0] = interp_left_fill
            f_new[:,1] = (1 + interp_left_fill)/2 * f[:,0]/simultaneous
            f_new[:,2] = f[:,0]*2**(simultaneous-1)/simultaneous
            self.f = f_new
            
            self.interp = self.make_interpolant(kind='zero',left_fill=interp_left_fill)
            
        else:
            self.impulsive = False
            self.t = t
            self.f = f

            self.interp = self.make_interpolant(kind=interp_kind,left_fill=interp_left_fill)

        self.t_checkpoint = self.t
        self.f_checkpoint = self.f

    def make_interpolant(self,*, kind='cubic', left_fill=0):
        """Interpolates density matrix and pads using 0 to the left
            and f[-1] to the right
"""
        if kind == 'pchip':
            pchipr = PchipInterpolator(self.t,np.real(self.f),axis = 1,extrapolate=False)
            pchipi = PchipInterpolator(self.t,np.imag(self.f),axis = 1,extrapolate=False)
            print(kind)
            def f(t):
                tmin = self.t[0]
                tmax = self.t[-1]
                tchip_inds = np.where((t>=tmin) & (t<=tmax))[0]
                t0_inds = np.where(t<tmin)[0]
                t_asymptote_inds = np.where(t>tmax)[0]
                ans = np.zeros((self.n,t.size),dtype='complex')
                ans[:,t0_inds] = 0
                ans[:,tchip_inds] = pchipr(t[tchip_inds]) + 1j *pchipi(t[tchip_inds])
                ans[:,t_asymptote_inds] = self.asymptote[:,None]
                return ans
        else:
            left_fill = np.ones(self.n)*left_fill
            right_fill = np.ones(self.n)*np.nan
            fill_value = (left_fill,right_fill)
            f = sinterp1d(self.t,self.f, kind=kind,
                         fill_value = fill_value,
                         assume_sorted=True,bounds_error=False)
        return f

    def one_time_step(self,f0,t0,tf,*,find_best_starting_time = True):
        if find_best_starting_time and tf < self.t_checkpoint[-1]:
            diff1 = tf - t0

            diff2 = tf - self.t[-1]

            closest_t_checkpoint_ind = np.argmin(np.abs(self.t_checkpoint - tf))
            closest_t_checkpoint = self.t_checkpoint[closest_t_checkpoint_ind]
            diff3 = tf - closest_t_checkpoint

            f0s = [f0,self.f[:,-1],self.f_checkpoint[:,closest_t_checkpoint_ind]]
            
            neighbor_ind = closest_t_checkpoint_ind - 1
            if neighbor_ind >= 0:
                neighbor = self.t_checkpoint[closest_t_checkpoint_ind-1]
                diff4 = tf - neighbor
                f0s.append(self.f_checkpoint[:,neighbor_ind])
            else:
                neighbor = np.nan
                diff4 = np.inf
                

            t0s = np.array([t0,self.t[-1],closest_t_checkpoint,neighbor])
            diffs = np.array([diff1,diff2,diff3,diff4])
            
            for i in range(diffs.size):
                if diffs[i] < 0:
                    diffs[i] = np.inf
            
            if np.allclose(diffs,np.inf):
                raise ValueError('Method extend is only valid for times after the pulse has ended')
            
            t0 = t0s[np.argmin(diffs)]
            f0 = f0s[np.argmin(diffs)]
            
        elif find_best_starting_time and tf > self.t_checkpoint[-1]:
            if self.t_checkpoint[-1] > t0:
                t0 = self.t_checkpoint[-1]
                f0 = self.f_checkpoint[:,-1]
            else:
                pass
            
        else:
            pass

        return self.one_time_step_function(f0,t0,tf,manifold_key=self.manifold_key)

    def extend(self,t):
        ans = np.zeros((self.n,t.size),dtype='complex')
        
        if t[0] >= self.t_checkpoint[0]:

            t_intersect, t_inds, t_checkpoint_inds = np.intersect1d(t,self.t_checkpoint,return_indices=True)

            ans[:,t_inds] = self.f_checkpoint[:,t_checkpoint_inds]

            if t_inds.size == t.size:
                return ans
            else:
                all_t_inds = np.arange(t.size)
                other_t_inds = np.setdiff1d(all_t_inds,t_inds)
                t0 = self.t_checkpoint[-1]
                f0 = self.f_checkpoint[:,-1]
                if t[other_t_inds[0]] >= t0:
                    find_best_starting_time = False
                else:
                    find_best_starting_time = True
                for t_ind in other_t_inds:
                    tf = t[t_ind]
                    ans[:,t_ind] = self.one_time_step(f0,t0,tf,find_best_starting_time = find_best_starting_time)
                    t0 = tf
                    f0 = ans[:,t_ind]
            
        elif t[0] >= self.t[-1]:
            t0 = self.t[-1]
            f0 = self.f[:,-1]
            for i in range(len(t)):
                ans[:,i] = self.one_time_step(f0,t0,t[i],find_best_starting_time = True)
                t0 = t[i]
                f0 = ans[:,i]
        else:
            raise ValueError('Method extend is only valid for times after the pulse has ended')

        self.f_checkpoint = ans
        self.t_checkpoint = t
        return ans

    def __call__(self,t):
        """Assumes t is sorted """
        try:
            length = len(t)
            if length == 0:
                return np.array([])
        except TypeError:
            #Assume input must be a number (not a list or an array)
            length = 1
        if type(t) is np.ndarray:
            pass
        elif type(t) is list:
            t = np.array(t)
        else:
            t = np.array([t])
        extend_inds = np.where(t>self.t[-1])
        interp_inds = np.where(t<=self.t[-1])
        ta = t[interp_inds]
        tb = t[extend_inds]
        if ta.size > 0:
            ans_a_flag = True
            if ta.size == self.M and np.allclose(ta,self.t):
                ans_a = self.f
            else:
                ans_a = self.interp(ta)
        else:
            ans_a_flag = False
        if tb.size > 0:
            ans_b = self.extend(tb)
            ans_b_flag = True
        else:
            ans_b_flag = False
            
        if ans_a_flag and ans_b_flag:
            ans = np.hstack((ans_a,ans_b))
        elif ans_a_flag:
            ans = ans_a
        elif ans_b_flag:
            ans = ans_b
        else:
            ans = None
        return ans

    def __getitem__(self,inds):
        return self.f[:,inds]
