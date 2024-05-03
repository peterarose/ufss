import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def midpoints_to_edges(x_arr):
    if x_arr[0] < x_arr[-1]:
        flip = False
    else:
        flip = True
        x_arr = x_arr[::-1]
    new_x_arr = np.zeros(x_arr.size+1)
    dx = x_arr[1] - x_arr[0]
    new_x_arr[:-1] = x_arr - dx/2
    new_x_arr[-1] = new_x_arr[-2] + dx
    if flip:
        new_x_arr = new_x_arr[::-1]
    return new_x_arr

def log_midpoints_to_edges(x_arr):
    if x_arr[0] < x_arr[-1]:
        flip = False
    else:
        flip = True
        x_arr = x_arr[::-1]
    xa = x_arr[0]
    xb = x_arr[-1]
    n = x_arr.size
    exp_a = np.log10(xa)
    exp_b = np.log10(xb)
    exp_dt = np.log10(x_arr[1]) - np.log10(x_arr[0])
    new_x_arr = np.logspace(exp_a-exp_dt/2,exp_b+exp_dt/2,num=n+1)
    if flip:
        new_x_arr = new_x_arr[::-1]
    return new_x_arr



class Convergence():
    def __init__(self,spec_func,convergence_params,ref_params):
        """
        Args:
            spec_func (function) : must take as arguments Delta and M, and
                return a spectroscopy object that has already performed its
                calculation
            convergence_params (list) : must be a list of numpy arrays, lists,
                or tuples (if there is only one convergence parameter, the 
                format must be [[a,b,c,...]])
            ref_param (list) : must be a list of reference parameters to be
                fed to spec_func to obtain a reference signal for convergence
                comparison
"""
        self.spec_func = spec_func
        self.convergence_params = convergence_params
        self.ref_params = ref_params

    def L2_norm(self,a,b):
        return np.sqrt(np.sum(np.abs(b-a)**2)/np.sum(np.abs(b)**2))

    def run(self):
        shape = tuple([len(params) for params in self.convergence_params])
        L2norms = []
        times = []
        Ms = []
        self.ref_obj = self.spec_func(*self.ref_params)
        ref_sig = self.ref_obj.signal
        
        param_combinations = itertools.product(*self.convergence_params)
        for p_comb in param_combinations:
            print(p_comb)
            spec_obj = self.spec_func(*p_comb)
            sig = spec_obj.signal
            calc_time = spec_obj.calculation_time
            L2norms.append(self.L2_norm(sig,ref_sig))
            times.append(calc_time)
            Ms.append(spec_obj.efields[0].size)
            print(calc_time)

        self.Ms = np.array(Ms).reshape(shape)
        self.L2norms = np.array(L2norms).reshape(shape)
        self.times = np.array(times).reshape(shape)  

class efieldConvergence(Convergence):
    def __init__(self,spec_func,dts,Deltas):
        """
        Args:
            spec_func (function) : must take as arguments dt and Delta, and
                return an ndarray, which should be a spectroscopic signal
"""
        max_Delta = Deltas[-1]
        min_dt = dts[0]
        super().__init__(spec_func,[dts[1:],Deltas[:-1]],[min_dt,max_Delta])

    def plot(self):
        fig, ax = plt.subplots()

        dts, Deltas = self.convergence_params

        X,Y = np.meshgrid(log_midpoints_to_edges(dts),midpoints_to_edges(Deltas),indexing='ij')
        Xcont,Ycont = np.meshgrid(dts,Deltas,indexing='ij')
        handle = ax.pcolormesh(X,Y,self.L2norms,norm=LogNorm(vmin=self.L2norms.min(), vmax=self.L2norms.max()),shading='flat')
        cb = fig.colorbar(handle)
        cb.set_label('Relative Error',fontsize=16)
        chandle = ax.contour(Xcont,Ycont,self.L2norms,levels=np.logspace(-7,-2,num=6),colors='w')
        min_M_coords = self.find_minimum_M()
        ax.plot(*min_M_coords,'r*')
        ax.set_xscale('log')
        ax.set_xlabel('d$t$ ($\sigma$)',fontsize=16)
        ax.set_ylabel('$\Delta$ ($\sigma$)',fontsize=16)
        ax.set_xlim([dts[0],dts[-1]])
        ax.set_ylim([Deltas[0],Deltas[-1]])
        fmt = {}
        lvl_strs = [r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$']
        for l,s in zip(chandle.levels,lvl_strs):
            fmt[l] = s

        ax.clabel(chandle,chandle.levels,fmt=fmt,fontsize=8)

        fig.tight_layout()
        #fig.subplots_adjust(wspace=0.25)
        # fig.savefig(os.path.join(folder,'Smallwood2017_UF2Comparison.pdf'))

    def find_minimum_M(self,*,signal_threshold = 1E-2):
        inds = np.where(self.L2norms > signal_threshold)
        dts, Deltas = self.convergence_params
        Ms = np.zeros((dts.size,Deltas.size))
        for i in range(dts.size):
            for j in range(Deltas.size):
                m = round(Deltas[j]/2/dts[i])
                Ms[i,j] = 2*m+1

        Ms[inds] = np.inf
        min_ind = np.argmin(Ms)
        dt_ind,Delta_ind = np.unravel_index(min_ind,Ms.shape)
        return dts[dt_ind],Deltas[Delta_ind]
