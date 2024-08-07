import numpy as np
import numpy.polynomial.chebyshev as npch
from scipy.interpolate import interp1d as sinterp1d

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
            left and f[-1] to the right
"""
        left_fill = np.ones(self.n,dtype='complex')*left_fill
        right_fill = self.f[:,-1]
        return sinterp1d(self.t,self.f,fill_value = (left_fill,right_fill),
                         assume_sorted=True,bounds_error=False,kind=kind)

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
        self.cheb_coefs = npch.chebint(self.cheb_coefs,axis=1,
                                       lbnd=-1)*self.halfwidth
        self.order = self.cheb_coefs.shape[1]
        self.deg = self.order - 1

        return None

    def __call__(self,t):
        return self.f_interp(t)

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
        left_fill = np.ones(self.n)*left_fill
        right_fill = np.ones(self.n)*np.nan
        fill_value = (left_fill,right_fill)
        return sinterp1d(self.t,self.f, kind=kind,
                         fill_value = fill_value,
                         assume_sorted=True,bounds_error=False)

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
