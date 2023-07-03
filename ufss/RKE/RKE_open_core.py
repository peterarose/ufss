#Standard python libraries
import os
import warnings
import copy
import time
import itertools
import functools

#Dependencies - numpy, scipy, matplotlib, pyfftw
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq
from scipy.interpolate import interp1d as sinterp1d
import scipy
from scipy.sparse import save_npz, load_npz, eye, csr_matrix
from scipy.sparse.linalg import eigs

from ufss import DiagramGenerator
from scipy.integrate import RK45

class RK_rho_container:
    def __init__(self,t,rho,pulse_number,manifold_key,*,interp_kind='linear',
                 optical_gap = 0):
        self.pulse_number = pulse_number
        self.n, self.M = rho.shape
        self.manifold_key = manifold_key
        self.optical_gap = optical_gap
        if t.size == 1:
            self.M = self.M+2
            self.t = np.array([-1,0,1],dtype='float') * np.spacing(t[0]) + t[0]
            rho_new = np.zeros((self.n,3),dtype='complex')
            rho_new[:,0] = 0
            rho_new[:,1] = 0.5 * rho[:,0]
            rho_new[:,2] = rho[:,0]
            self.rho = rho_new
            
            self.interp = self.make_interpolant(kind='zero')
            
        else:
            self.t = t
            self.rho = rho

            self.interp = self.make_interpolant(kind=interp_kind)

        self.t_checkpoint = t
        self.rho_checkpoint = rho

    def make_interpolant(self,*, kind='cubic'):
        """Interpolates density matrix
"""
        return sinterp1d(self.t,self.rho,fill_value = (0,np.nan),bounds_error = False,
                         assume_sorted=True,kind=kind)

    def one_time_step(self,rho0,t0,tf,*,find_best_starting_time = True):
        if find_best_starting_time and tf < self.t_checkpoint[-1]:
            diff1 = tf - t0

            diff2 = tf - self.t[-1]

            closest_t_checkpoint_ind = np.argmin(np.abs(self.t_checkpoint - tf))
            closest_t_checkpoint = self.t_checkpoint[closest_t_checkpoint_ind]
            diff3 = tf - closest_t_checkpoint

            rho0s = [rho0,self.rho[:,-1],self.rho_checkpoint[:,closest_t_checkpoint_ind]]
            
            neighbor_ind = closest_t_checkpoint_ind - 1
            if neighbor_ind >= 0:
                neighbor = self.t_checkpoint[closest_t_checkpoint_ind-1]
                diff4 = tf - neighbor
                rho0s.append(self.rho_checkpoint[:,neighbor_ind])
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
            rho0 = rho0s[np.argmin(diffs)]
            
        elif find_best_starting_time and tf > self.t_checkpoint[-1]:
            if self.t_checkpoint[-1] > t0:
                t0 = self.t_checkpoint[-1]
                rho0 = self.rho_checkpoint[:,-1]
            else:
                pass
            
        else:
            pass
        # RWA_gap = self.manifold.dot(np.array([1,-1])) * self.optical_gap
        return self.one_time_step_function(rho0,t0,tf,manifold_key=self.manifold_key)#,RWA_gap = RWA_gap)

    def extend(self,t):
        ans = np.zeros((self.n,t.size),dtype='complex')
        
        if t[0] >= self.t_checkpoint[0]:

            t_intersect, t_inds, t_checkpoint_inds = np.intersect1d(t,self.t_checkpoint,return_indices=True)

            ans[:,t_inds] = self.rho_checkpoint[:,t_checkpoint_inds]

            if t_inds.size == t.size:
                return ans
            else:
                all_t_inds = np.arange(t.size)
                other_t_inds = np.setdiff1d(all_t_inds,t_inds)
                t0 = self.t_checkpoint[-1]
                rho0 = self.rho_checkpoint[:,-1]
                if t[other_t_inds[0]] >= t0:
                    find_best_starting_time = False
                else:
                    find_best_starting_time = True
                for t_ind in other_t_inds:
                    tf = t[t_ind]
                    ans[:,t_ind] = self.one_time_step(rho0,t0,tf,find_best_starting_time = find_best_starting_time)
                    t0 = tf
                    rho0 = ans[:,t_ind]
            
        elif t[0] >= self.t[-1]:
            t0 = self.t[-1]
            rho0 = self.rho[:,-1]
            for i in range(len(t)):
                ans[:,i] = self.one_time_step(rho0,t0,t[i],find_best_starting_time = True)
                t0 = t[i]
                rho0 = ans[:,i]
        else:
            raise ValueError('Method extend is only valid for times after the pulse has ended')

        self.rho_checkpoint = ans
        self.t_checkpoint = t
        return ans

    def __call__(self,t):
        """Assumes t is sorted """
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
                ans_a = self.rho
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
        return self.rho[:,inds]

class RKE_DensityMatrices(DiagramGenerator):
    """This class is designed to calculate perturbative wavepackets in the
        light-matter interaction given the eigenvalues of the unperturbed 
        hamiltonian and the material dipole operator evaluated in the
        eigenbasis of the unperturbed hamiltonian.

    Args:
        file_path (string): path to folder containing eigenvalues and the
            dipole operator for the system Hamiltonian
        detection_type (string): options are 'polarization' (default) or 'fluorescence'

"""
    def __init__(self,file_path,*,detection_type = 'polarization',
                 conserve_memory=False):
        self.slicing_time = 0
        self.interpolation_time = 0
        self.expectation_time = 0
        self.RK45_step_time = 0
        self.dipole_dot_rho_time = 0
        self.dipole_time = 0
        self.automation_time = 0
        self.diagram_to_signal_time = 0
        
        self.base_path = file_path

        self.undersample_factor = 1

        self.gamma_res = 6.91

        self.sparsity_threshold = 0.1

        self.conserve_memory = conserve_memory

        self.load_L()

        self.set_rho_shapes()

        if not self.conserve_memory:
            self.load_mu()

        try:
            self.load_H_mu()
            # more efficient if H_mu is available
            self.dipole_down = self.dipole_down_H_mu

        except:
            # generally less efficient - mostly here for backwards compatibility
            self.dipole_down = self.dipole_down_L_mu

        self.optical_gap = 0

        self.atol = 1E-6
        self.rtol = 1E-5

        if detection_type == 'polarization':
            self.rho_to_signal = self.polarization_detection_rho_to_signal
            self.return_complex_signal = False
            
        elif detection_type == 'complex_polarization':
            self.rho_to_signal = self.polarization_detection_rho_to_signal
            self.return_complex_signal = True
            detection_type = 'polarization'
            
        elif detection_type == 'integrated_polarization':
            raise Exception('detection_type: Integrated polarization is not implemented for Open RKE')
            self.rho_to_signal = self.integrated_polarization_detection_rho_to_signal
            self.return_complex_signal = False
            
        elif detection_type == 'fluorescence':
            self.rho_to_signal = self.fluorescence_detection_rho_to_signal

        DiagramGenerator.__init__(self,detection_type=detection_type)
        self.KB_dict = {'Bu':self.bra_up,'Ku':self.ket_up,'Kd':self.ket_down,'Bd':self.bra_down}

        # Code will not actually function until the following three empty lists are set by the user
        self.efields = [] #initialize empty list of electric field shapes
        self.efield_times = [] #initialize empty list of times assoicated with each electric field shape
        self.dts = [] #initialize empty list of time spacings associated with each electric field shape
        self.polarization_sequence = [] #initialize empty polarization sequence
        self.pulse_times = [] #initialize empty list of pulse arrival times
        self.centers = [] #initialize empty list of pulse center frequencies
        self.efield_wavevectors = []
        self.rhos = dict()
        
        # Initialize unperturbed wavefunction
        self.set_rho0_auto()

    def set_pulse_delays(self,all_delays):
        """Must be a list of numpy arrays, where each array is a
            list of delay times between pulses
"""
        self.all_pulse_delays = all_delays
        num_delays = len(self.all_pulse_delays)
        num_pulses = len(self.efields)
        
        if num_delays == num_pulses - 1:
            pass
        elif num_delays == num_pulses - 2 and self.detection_type == 'polarization':
            # If there is a local oscillator, it arrives simultaneously with the last pulse
            self.all_pulse_delays.append(np.array([0]))
        elif num_delays <= num_pulses -2:
            raise Exception('There are not enough delay times')
        elif num_delays >= num_pulses:
            raise Exception('There are too many delay times')

    def calculate_diagrams_all_delays(self,diagrams):
        t0 = time.time()
        num_delays = len(self.all_pulse_delays)
        num_pulses = len(self.efields)

        all_delay_combinations = list(itertools.product(*self.all_pulse_delays))
        
        signal_shape = [delays.size for delays in self.all_pulse_delays]
        if self.detection_type == 'polarization':
            signal = np.zeros((len(all_delay_combinations),self.w.size),dtype='complex')
            if len(signal_shape) == self.pdc.shape[0]:
                # get rid of the "delay" between the last pulse and the local oscillator
                signal_shape[-1] = self.w.size
            elif len(signal_shape) == self.pdc.shape[0] - 1:
                # append the shape of the polariation-detection axis
                signal_shape.append(self.w.size)
            else:
                raise Exception('Cannot automatically determine final signal shape')
        else:
            signal = np.zeros((len(all_delay_combinations)),dtype='complex')

        counter = 0
        for delays in all_delay_combinations:
            arrival_times = [0]
            for delay in delays:
                arrival_times.append(arrival_times[-1]+delay)

            if self.detection_type == 'polarization':
                signal[counter,:] = self.calculate_diagrams(diagrams,arrival_times)
            else:
                signal[counter] = self.calculate_diagrams(diagrams,arrival_times)
            counter += 1

        self.signal = signal.reshape(signal_shape)
        self.calculation_time = time.time() - t0
        
        return self.signal

    def save_timing(self):
        save_dict = {'RKE_calculation_time':self.calculation_time}
        np.savez(os.path.join(self.base_path,'RKE_calculation_time.npz'),**save_dict)
        
    def calculate_signal_all_delays(self):
        t0 = time.time()
        num_delays = len(self.all_pulse_delays)
        num_pulses = len(self.efields)

        all_delay_combinations = list(itertools.product(*self.all_pulse_delays))
        
        signal_shape = [delays.size for delays in self.all_pulse_delays]
        if self.detection_type == 'polarization':
            signal = np.zeros((len(all_delay_combinations),self.w.size),dtype='complex')
            if len(signal_shape) == self.pdc.shape[0]:
                # get rid of the "delay" between the last pulse and the local oscillator
                signal_shape[-1] = self.w.size
            elif len(signal_shape) == self.pdc.shape[0] - 1:
                # append the shape of the polariation-detection axis
                signal_shape.append(self.w.size)
            else:
                raise Exception('Cannot automatically determine final signal shape')
        else:
            signal = np.zeros((len(all_delay_combinations)),dtype='complex')

        counter = 0
        for delays in all_delay_combinations:
            arrival_times = [0]
            for delay in delays:
                arrival_times.append(arrival_times[-1]+delay)

            if self.detection_type == 'polarization':
                signal[counter,:] = self.calculate_signal(arrival_times)
            else:
                signal[counter] = self.calculate_signal(arrival_times)
            counter += 1

        self.signal = signal.reshape(signal_shape)
        self.calculation_time = time.time() - t0
        return self.signal

    def set_t(self,optical_dephasing_rate,*,dt='auto'):
        """Sets the time grid upon which all frequency-detected signals will
be calculated on
"""
        max_pos_t = int(self.gamma_res/optical_dephasing_rate)
        if dt == 'auto':
            dt = self.dts[-1] # signal detection bandwidth determined by local oscillator
        self.t = np.arange(-max_pos_t,max_pos_t+dt/2,dt)
        # if self.t.size % 2:
        #     self.t = self.t[:-1]
        self.w = fftshift(fftfreq(self.t.size,d=dt)*2*np.pi)

    def execute_diagram(self,instructions):
        r = self.rho0
        name = ''
        for i in range(len(instructions)):
            key, num = instructions[i]
            name += key+str(num)
            # Try to re-use previous calculations, if they exist
            try:
                new_r = self.rhos[name]
            except KeyError:
                new_r = self.KB_dict[key](r,pulse_number=num)
                self.rhos[name] = new_r
            r = new_r
        sig = self.rho_to_signal(r)
        return sig

    def remove_rhos_by_pulse_number(self,pulse_number):
        num = str(pulse_number)
        keys = self.rhos.keys()
        keys_to_remove = []
        for key in keys:
            flag = key.find(num)
            if flag >= 0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.rhos.pop(key)

    def calculate_signal(self,arrival_times):
        t0 = time.time()
        try:
            old_pulse_times = self.pulse_times
            for i in range(len(old_pulse_times)):
                if old_pulse_times[i] != arrival_times[i]:
                    self.remove_rhos_by_pulse_number(i)
        except AttributeError:
            pass
        
        self.pulse_times = arrival_times
        if self.detection_type == 'polarization':
            times = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times)-1)]
        elif self.detection_type == 'integrated_polarization':
            times = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times)-1)]
        elif self.detection_type == 'fluorescence':
            times = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times))]
        
        efield_permutations = self.relevant_permutations(times)
        
        diagram_instructions = []
        for perm in efield_permutations:
            diagram_instructions += self.instructions_from_permutation(perm)
        self.current_instructions = diagram_instructions

        t1 = time.time()
        try:
            instructions = diagram_instructions[0]
            signal = self.execute_diagram(instructions)
            for instructions in diagram_instructions[1:]:
                signal += self.execute_diagram(instructions)
        except IndexError:
            signal = 0

        t2 = time.time()
        self.automation_time += t1-t0
        self.diagram_to_signal_time += t2-t1
        return signal

    def calculate_diagrams(self,diagram_instructions,arrival_times):
        try:
            old_pulse_times = self.pulse_times
            for i in range(len(old_pulse_times)):
                if old_pulse_times[i] != arrival_times[i]:
                    self.remove_rhos_by_pulse_number(i)
        except AttributeError:
            pass
        
        self.pulse_times = arrival_times
            
        self.current_instructions = diagram_instructions
        instructions = diagram_instructions[0]
        signal = self.execute_diagram(instructions)
        for instructions in diagram_instructions[1:]:
            signal += self.execute_diagram(instructions)
        return signal

    def polarization_detection_rho_to_signal(self,rho):    
        p_of_t = self.dipole_expectation(rho,pulse_number=-1,ket_flag=True)
        return self.polarization_to_signal(p_of_t,local_oscillator_number=-1)

    def integrated_polarization_detection_rho_to_signal(self,rho):    
        p = self.integrated_dipole_expectation(rho,ket_flag=True)
        return self.integrated_polarization_to_signal(p,local_oscillator_number=-1)

    # def fluorescence_detection_rho_to_signal(self,rho):
    #     L_size = self.eigenvalues[0].size
    #     H_size = int(np.sqrt(L_size))

    #     # reshape rho into a normal density matrix representation
    #     rho = rho.reshape((H_size,H_size))

    #     fluorescence_yield = np.array([0,1,1,self.f_yield])

    #     signal = np.dot(np.diagonal(rho),fluorescence_yield)
        
    #     return signal

    def set_efields(self,times_list,efields_list,centers_list,phase_discrimination,*,reset_rhos = True,
                    plot_fields = False):
        self.efield_times = times_list
        self.efields = efields_list
        self.centers = centers_list
        self.set_phase_discrimination(phase_discrimination)
        self.dts = []
        self.efield_frequencies = []
        if reset_rhos:
            self.rhos = dict()
        for t in times_list:
            if t.size == 1:
                dt = 1
                w = np.array([0])
            else:
                dt = t[1] - t[0]
                w = fftshift(fftfreq(t.size,d=dt))*2*np.pi
            self.dts.append(dt)
            self.efield_frequencies.append(w)

        self.dt = self.dts[0]

        if self.detection_type == 'polarization':
            try:
                self.local_oscillator = self.efields[-1].copy()
            except:
                self.local_oscillator = copy.deepcopy(self.efields[-1])

        for field in self.efields:
            if len(field) == 1:
                # M = 1 is the impulsive limit
                pass
            else:
                self.check_efield_resolution(field,plot_fields = plot_fields)

    def check_efield_resolution(self,efield,*,plot_fields = False):
        efield_tail = np.max(np.abs([efield[0],efield[-1]]))


        if efield_tail > np.max(np.abs(efield))/100:
            warnings.warn('Consider using larger num_conv_points, pump does not decay to less than 1% of maximum value in time domain')
            
        efield_fft = fftshift(fft(ifftshift(efield)))*self.dt
        efield_fft_tail = np.max(np.abs([efield_fft[0],efield_fft[-1]]))
        
        if efield_fft_tail > np.max(np.abs(efield_fft))/100:
            warnings.warn('''Consider using smaller value of dt, pump does not decay to less than 1% of maximum value in frequency domain''')

        if plot_fields:
            fig, axes = plt.subplots(1,2)
            l1,l2, = axes[0].plot(self.efield_t,np.real(efield),self.efield_t,np.imag(efield))
            plt.legend([l1,l2],['Real','Imag'])
            axes[1].plot(self.efield_w,np.real(efield_fft),self.efield_w,np.imag(efield_fft))

            axes[0].set_ylabel('Electric field Amp')
            axes[0].set_xlabel('Time ($\omega_0^{-1})$')
            axes[1].set_xlabel('Frequency ($\omega_0$)')

            fig.suptitle('Check that efield is well-resolved in time and frequency')
            plt.show()

    def set_local_oscillator_phase(self,phase):
        self.efields[-1] = np.exp(1j*phase) * self.local_oscillator

    def get_closest_index_and_value(self,value,array):
        """Given an array and a desired value, finds the closest actual value
stored in that array, and returns that value, along with its corresponding 
array index
"""
        index = np.argmin(np.abs(array - value))
        value = array[index]
        return index, value

    def load_L(self):
        """Load in known eigenvalues. Must be stored as a numpy archive file,
with keys: GSM, SEM, and optionally DEM.  The eigenvalues for each manifold
must be 1d arrays, and are assumed to be ordered by increasing energy. The
energy difference between the lowest energy ground state and the lowest 
energy singly-excited state should be set to 0
"""
        L_save_name = os.path.join(self.base_path,'L.npz')
        try:
            with np.load(L_save_name,allow_pickle=True) as L_archive:
                self.L = dict()
                for key in L_archive.keys():
                    L = L_archive[key]
                    if L.dtype == np.dtype('O'):
                        self.L[key] = L[()]
                    else:
                        if self.check_sparsity(L):
                            self.L[key] = csr_matrix(L)
                        else:
                            self.L[key] = L
        except:
            self.L = {'all_manifolds':load_npz(L_save_name)}
        self.manifolds = list(self.L.keys())

    def check_sparsity(self,mat):
        csr_mat = csr_matrix(mat)
        sparsity = csr_mat.nnz / (csr_mat.shape[0]*csr_mat.shape[1])
        if sparsity < self.sparsity_threshold:
            return True
        else:
            return False
        
    def dL(self,t,rho):
        try:
            L = self.L['all_manifolds']
        except KeyError:
            L = self.L[rho.manifold_key]
        return L.dot(rho)

    def get_dL_manual(self,manifold_key):
        try:
            L = self.L['all_manifolds']
        except KeyError:
            L = self.L[manifold_key]

        def L_fun(t,rho):
            return L.dot(rho)

        return L_fun

    def one_time_step_function(self,rho0,t0,tf,*,manifold_key = None):
        num_steps = 0
        if manifold_key == None:
            rk45 = RK45(self.dL,t0,rho0,tf,atol=self.atol,rtol=self.rtol)
        else:
            dL = self.get_dL_manual(manifold_key)
            rk45 = RK45(dL,t0,rho0,tf,atol=self.atol,rtol=self.rtol)
        while rk45.t < tf:
            rk45.step()
            num_steps += 1
        rho_final = rk45.y
        return rho_final

    def get_bottom_eigenvector(self):
        try:
            L = self.L['all_manifolds']
        except KeyError:
            L = self.L['00']
        if L.shape == (1,1):
            e = L[0,0]
            ev = np.array([[1]])
        else:
            e, ev = eigs(L,k=1,which='SM',maxiter=10000)
        if e.size == 1 and np.allclose(e,0):
            pass
        else:
            raise Exception('Smallest magnitude eigenvalue of L is {}. L must have a single stationary state for this code to work'.format(e))
        v = ev[:,0]
        H_size = int(np.sqrt(v.size))
        rho = v.reshape((H_size,H_size))
        trace = rho.trace()
        v = v/trace # Need to start with a trace 1 object
        return v

    def set_rho0_auto(self):
        try:
            rho0 = np.load(os.path.join(self.base_path,'rho0.npy'))
        except FileNotFoundError:
            rho0 = self.get_bottom_eigenvector()
        t = np.array([-np.inf,0,np.inf])
        rho0 = rho0[:,np.newaxis] * np.ones((rho0.size,t.size))
        pulse_number = None
        manifold_key = '00'
        self.rho0 = RK_rho_container(t,rho0,pulse_number,manifold_key,
                                     interp_kind = 'zero',optical_gap = self.optical_gap)

    def set_rho_shapes(self):
        self.rho_shapes = dict()
        if 'all_manifolds' in self.manifolds:
            L_size = self.L['all_manifolds'].size
            H_size = int(np.sqrt(L_size))
            self.rho_shapes['all_manifolds'] = (H_size,H_size)
        else:
            H_sizes = dict()
            for key in self.manifolds:
                ket_key, bra_key = key
                if ket_key == bra_key:
                    L_size = self.L[key].shape[0]
                    H_size = int(np.sqrt(L_size))
                    H_sizes[ket_key] = H_size
            for key in self.manifolds:
                ket_key, bra_key = key
                ket_size = H_sizes[ket_key]
                bra_size = H_sizes[bra_key]
                self.rho_shapes[key] = (ket_size,bra_size)

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
            be stored as a .npz file, and must contain at least one array, each with three 
            indices: (new manifold index, old manifold eigenfunction, 
            cartesian coordinate)."""
        try:
            file_name = os.path.join(self.base_path,'mu_site_basis.npz')
            with np.load(file_name) as mu_archive:
                self.mu = {key:mu_archive[key] for key in mu_archive.keys()}
        except FileNotFoundError:
            try:
                file_name = os.path.join(self.base_path,'mu_original_L_basis.npz')
                with np.load(file_name) as mu_archive:
                    self.mu = {key:mu_archive[key] for key in mu_archive.keys()}
            except FileNotFoundError:
                file_name = os.path.join(self.base_path,'mu.npz')
                with np.load(file_name) as mu_archive:
                    self.mu = {key:mu_archive[key] for key in mu_archive.keys()}
        sparse_flags = []
        for key in self.mu.keys():
            mu_2D = np.sum(np.abs(self.mu[key])**2,axis=-1)
            sparse_flags.append(self.check_sparsity(mu_2D))
        sparse_flags = np.array(sparse_flags)
        if np.allclose(sparse_flags,True):
            self.sparse_mu_flag = True
        else:
            self.sparse_mu_flag = False

        for key in self.mu.keys():
            mu_x = self.mu[key][...,0]
            mu_y = self.mu[key][...,1]
            mu_z = self.mu[key][...,2]

            if self.sparse_mu_flag:
                self.mu[key] = [csr_matrix(mu_x),csr_matrix(mu_y),csr_matrix(mu_z)]
            else:
                self.mu[key] = [mu_x,mu_y,mu_z]

        print('RKE_sparse_mu_flag',self.sparse_mu_flag)
        
    ### Setting the electric field to be used

    def set_polarization_sequence(self,polarization_list,*,reset_rhos=True):
        """Sets the sequences used for either parallel or crossed pump and probe
        
        Args:
            polarization_list (list): list of four strings, can be 'x','y' or 'z'
        Returns:
            None: sets the attribute polarization sequence
"""

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        pol_options = {'x':x,'y':y,'z':z}

        self.polarization_sequence = [pol_options[pol] for pol in polarization_list]

        if reset_rhos:
            self.rhos = dict()


    ### Tools for recursively calculating perturbed density maatrices using TDPT

    def dipole_matrix(self,pulse_number,key,ket_flag=True,up_flag=True):
        """Calculates the dipole matrix given the electric field polarization vector,
            if ket_flag = False then uses the bra-interaction"""
        t0 = time.time()
        pol = self.polarization_sequence[pulse_number]

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        try:
            mu = self.mu[key]
        except KeyError:
            if ket_flag:
                key = 'ket'
            else:
                key = 'bra'
            if up_flag:
                key += '_up'
            else:
                key += '_down'
            mu = self.mu[key]
            
        if np.all(pol == x):
            overlap_matrix = mu[0]#.copy()
        elif np.all(pol == y):
            overlap_matrix = mu[1]#.copy()
        elif np.all(pol == z):
            overlap_matrix = mu[2]#.copy()
        else:
            overlap_matrix = mu[0]*pol[0] + mu[1]*pol[1] + mu[2]*pol[2]

        # if self.sparse_mu_flag:
        #     to_return = csr_matrix(overlap_matrix)
        # else:
        #     to_return = overlap_matrix

        t1 = time.time()
        self.dipole_time += t1-t0

        return overlap_matrix

    def manifold_key_to_array(self,key):
        """Key must be a string of exactly 2 integers, the first describing
            the ket manifold, the second the bra manifold.  If the density 
            matrix is represented in the full space, rather than being divided
            into manifolds, the first integer reperesents the total number of
            excitations to the ket side, and the second integers represents 
            the sum of all excitations to the bra side."""
        if len(key) != 2:
            raise Exception('manifold key must be a string of exactly two intgers')
        return np.array([int(char) for char in key],dtype=int)
    
    def manifold_array_to_key(self,manifold):
        """Inverse of self.manifold_key_to_array"""
        if manifold.size != 2 or manifold.dtype != int:
            raise Exception('manifold array must contain exactly 2 integer') 
        return str(manifold[0]) + str(manifold[1])

    def next_order(self,rho_in,*,ket_flag=True,up_flag=True,pulse_number = 0):
        """This function connects psi_p to psi+pj^(*) using the Euler Method.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)
        
        Return:
            rho_dict (rho_container): next-order density matrix
"""     
        pulse_time = self.pulse_times[pulse_number]
        t = self.efield_times[pulse_number] + pulse_time
        old_manifold_key = rho_in.manifold_key
        if up_flag:
            change = 1
        else:
            change = -1
        if ket_flag:
            manifold_change = np.array([change,0],dtype=int)
        else:
            manifold_change = np.array([0,change],dtype=int)
        old_manifold = self.manifold_key_to_array(old_manifold_key)
        new_manifold = old_manifold + manifold_change
        new_manifold_key = self.manifold_array_to_key(new_manifold)
        mu_key = old_manifold_key + '_to_' + new_manifold_key
        
        if ket_flag == up_flag:
            # Rotating term excites the ket and de-excites the bra
            conjugate_flag = False
        else:
            # Counter-rotating term
            conjugate_flag = True

        if conjugate_flag:
            center = -self.centers[pulse_number]
        else:
            center = self.centers[pulse_number]

        M = t.size
        old_rho = rho_in(t)

        if self.conserve_memory:
            # move back to the basis the Liouvillian was written in
            if 'all_manifolds' in self.manifolds:
                ket_size,bra_size = self.rho_shapes['all_manifolds']
            else:
                ket_size,bra_size = self.rho_shapes[old_manifold_key]
                
            old_rho = old_rho.reshape(ket_size,bra_size,M)

            if ket_flag:
                old_ket_key = old_manifold_key[0]
                new_ket_key = new_manifold_key[0]
                if up_flag:
                    H_mu_key = old_ket_key + '_to_' + new_ket_key
                else:
                    H_mu_key = new_ket_key + '_to_' + old_ket_key

                mu_up_flag = up_flag
            else:
                old_bra_key = old_manifold_key[1]
                new_bra_key = new_manifold_key[1]
                if up_flag:
                    H_mu_key = old_bra_key + '_to_' + new_bra_key
                else:
                    H_mu_key = new_bra_key + '_to_' + old_bra_key
                mu_up_flag = not up_flag

            overlap_matrix = self.get_H_mu(pulse_number,H_mu_key,up_flag=mu_up_flag)
            
            ta = time.time()
            if ket_flag:
                mu_old_rho = np.einsum('ij,jkl',overlap_matrix,old_rho)
            else:
                mu_old_rho = np.einsum('ijl,jk',old_rho,overlap_matrix)
            tb = time.time()
            
            rho_vec_size = mu_old_rho.shape[0]*mu_old_rho.shape[1]
            mu_old_rho = mu_old_rho.reshape(rho_vec_size,M)

        else:
            overlap_matrix = self.dipole_matrix(pulse_number,mu_key,ket_flag=ket_flag,up_flag=up_flag)
        
        
            ta = time.time()
            mu_old_rho = overlap_matrix.dot(old_rho)
            tb = time.time()
            
        self.dipole_dot_rho_time += tb - ta
        next_rho = np.zeros(mu_old_rho.shape,dtype='complex')

        if M == 1:
            next_rho[:,0] = self.efields[pulse_number] * mu_old_rho
        else:
            if conjugate_flag:
                efield = self.efields[pulse_number]*np.exp(-1j*center*t)
            else:
                efield = np.conjugate(self.efields[pulse_number])*np.exp(-1j*center*t)


            ############
            # This 1j vs -1j needs to be derived!!! #####
            ############
            if ket_flag:
                efield = 1j * efield
            else:
                efield = -1j * efield
            ###########
            ###########
            
            dt = self.dts[pulse_number]
            next_rho[:,0] = efield[0] * mu_old_rho[:,0] * dt
            for i in range(1,t.size):
                rho0 = next_rho[:,i-1]
                t0 = t[i-1]
                ta = time.time()
                next_rho[:,i] = self.one_time_step_function(rho0,t0,t[i],manifold_key=new_manifold_key)

                tb = time.time()
                self.RK45_step_time += tb - ta
        
                next_rho[:,i] += efield[i] * mu_old_rho[:,i] * dt

        # # i/hbar Straight from perturbation theory
        # if ket_flag:
        #     rho *= 1j
        # else:
        #     rho *= -1j

        rho_out = RK_rho_container(t,next_rho,pulse_number,new_manifold_key,
                                   optical_gap = self.optical_gap)
        rho_out.one_time_step_function = self.one_time_step_function
    
        return rho_out
            
    def ket_up(self,rho_in,*,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)

        Returns:
            (rho_container): output from method next_order
"""
        return self.next_order(rho_in,ket_flag=True,up_flag=True,
                               pulse_number = pulse_number)

    def ket_down(self,rho_in,*,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)

        Returns:
            (rho_container): output from method next_order
"""
        return self.next_order(rho_in,ket_flag=True,up_flag=False,
                               pulse_number = pulse_number)

    def bra_up(self,rho_in,*,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)

        Returns:
            (rho_container): output from method next_order
"""
        return self.next_order(rho_in,ket_flag=False,up_flag=True,
                               pulse_number = pulse_number)

    def bra_down(self,rho_in,*,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)

        Returns:
            (rho_container): output from method next_order
"""
        return self.next_order(rho_in,ket_flag=False,up_flag=False,
                               pulse_number = pulse_number)

    ### Tools for taking the expectation value of the dipole operator with perturbed density matrices

    def dipole_down_H_mu(self,rho,manifold_key,*,new_manifold_mask = None,pulse_number = -1,
                    ket_flag=True):
        """This method is similar to the method down, but does not involve 
            the electric field shape or convolutions. It is the action of the 
            dipole operator on the ket-side without TDPT effects.  It also includes
            the dot product of the final electric field polarization vector."""
        
        if not ket_flag:
            raise Exception('Not implemented for bra-side')
        old_manifold_key = manifold_key
        old_ket_key = old_manifold_key[0]
        new_ket_key = str(int(old_ket_key)-1)
        mu_key = new_ket_key + '_to_' + old_ket_key

        if ket_flag:
            center = - self.centers[pulse_number]
            conjugate_flag = True
        else:
            center = self.centers[pulse_number]
            conjugate_flag = False

        t_size = rho.shape[-1]
        
        if 'all_manifolds' in self.L.keys():
            L_size = rho.size
            H_size = int(np.sqrt(L_size))
            rho = rho.reshape(H_size,H_size,t_size)
        else:
            ket_manifold_key = old_manifold_key[0] + old_manifold_key[0]
            ket_L_manifold_size = self.L[ket_manifold_key].shape[0]
            ket_H_size = int(np.sqrt(ket_L_manifold_size))

            bra_manifold_key = old_manifold_key[1] + old_manifold_key[1]
            bra_L_manifold_size = self.L[bra_manifold_key].shape[0]
            bra_H_size = int(np.sqrt(bra_L_manifold_size))

            rho = rho.reshape(ket_H_size,bra_H_size,t_size)
        
        overlap_matrix = self.get_H_mu(pulse_number,mu_key,ket_flag=ket_flag,up_flag=False)

        t0 = time.time()
        polarization_field = np.einsum('ij,jik',overlap_matrix,rho)
                
        t1 = time.time()

        return polarization_field

    def dipole_down_L_mu(self,rho,manifold_key,*,new_manifold_mask = None,
                         pulse_number = -1,ket_flag=True):
        """This method is similar to the method down, but does not involve 
            the electric field shape or convolutions. It is the action of the 
            dipole operator on the ket-side without TDPT effects.  It also includes
            the dot product of the final electric field polarization vector."""
        old_manifold_key = manifold_key
        change = -1
        if ket_flag:
            manifold_change = np.array([change,0])
        else:
            manifold_change = np.array([0,change])

        old_manifold = self.manifold_key_to_array(old_manifold_key)
        new_manifold = old_manifold + manifold_change
        new_manifold_key = self.manifold_array_to_key(new_manifold)
        mu_key = old_manifold_key + '_to_' + new_manifold_key
        
        if ket_flag:
            center = - self.centers[pulse_number]
            conjugate_flag = True
        else:
            center = self.centers[pulse_number]
            conjugate_flag = False

        rho_in = rho
        
        overlap_matrix = self.dipole_matrix(pulse_number,mu_key,ket_flag=True,up_flag=False)

        t0 = time.time()
        rho = overlap_matrix.dot(rho_in)
                
        t1 = time.time()

        L_size = rho.shape[0]
        H_size = int(np.sqrt(L_size))

        # reshape rho into a normal density matrix representation
        rho = rho.reshape((H_size,H_size,rho.shape[-1]))

        polarization_field = np.einsum('iij',rho)

        return polarization_field

    def load_H_mu(self):
        parent_dir = os.path.split(self.base_path)[0]
        file_name = os.path.join(parent_dir,'closed','mu.npz')

        with np.load(file_name) as mu_archive:
            self.H_mu = {key:mu_archive[key] for key in mu_archive.keys()}

    def get_H_mu(self,pulse_number,key,ket_flag=True,up_flag=True):
        """Calculates the dipole matrix given the electric field polarization vector,
            if ket_flag = False then uses the bra-interaction"""
        t0 = time.time()
        pol = self.polarization_sequence[pulse_number]

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        try:
            mu = self.H_mu[key]
        except KeyError:
            try:
                key = 'up'
                mu = self.H_mu[key]
            except KeyError:
                key = 'ket_up'
                mu = self.H_mu[key]
            
        if np.all(pol == x):
            overlap_matrix = mu[:,:,0].copy()
        elif np.all(pol == y):
            overlap_matrix = mu[:,:,1].copy()
        elif np.all(pol == z):
            overlap_matrix = mu[:,:,2].copy()
        else:
            overlap_matrix = np.tensordot(mu,pol,axes=(-1,0))

        if not up_flag:
            overlap_matrix = overlap_matrix.T

        t1 = time.time()
        self.dipole_time += t1-t0

        return overlap_matrix

    def set_undersample_factor(self,frequency_resolution):
        """dt is set by the pulse. However, the system dynamics may not require such a 
            small dt.  Therefore, this allows the user to set a requested frequency
            resolution for any spectrally resolved signals."""
        # f = pi/dt
        dt = np.pi/frequency_resolution
        u = int(np.floor(dt/self.dt))
        self.undersample_factor = max(u,1)
        
    def dipole_expectation(self,rho_in,*,pulse_number = -1,ket_flag=True):
        """Computes the expectation value of the dipole operator"""
        t0 = time.time()

        pulse_number = -1
        
        pulse_time = self.pulse_times[pulse_number]
        
        efield_t = self.efield_times[pulse_number] + pulse_time

        if ket_flag:
            center = - self.centers[pulse_number]
        else:
            center = self.centers[pulse_number]
        
        # The signal is zero before the final pulse arrives, and persists
        # until it decays. Therefore we avoid taking the sum at times
        # where the signal is zero.
        t = self.t + pulse_time

        pulse_start_ind = np.argmin(np.abs(t-efield_t[0]))
        if efield_t[0] < t[pulse_start_ind]:
            pulse_start_ind -= 1
        pulse_end_ind = np.argmin(np.abs(t-efield_t[-1]))
        if efield_t[-1] > t[pulse_end_ind]:
            pulse_end_ind += 1
        
        t_slice = slice(pulse_start_ind,None,None)
        t1_slice = slice(pulse_start_ind,pulse_end_ind,None)
        u = self.undersample_factor
        t2_slice = slice(pulse_end_ind,None,u)

        t = self.t[t_slice] + pulse_time
        t1 = self.t[t1_slice] + pulse_time
        t2 = self.t[t2_slice] + pulse_time

        rho1 = rho_in(t1)
        rho2 = rho_in(t2)

        rho1 *= np.exp(-1j * center * t1[np.newaxis,:])

        rho2 *= np.exp(-1j * center * t2[np.newaxis,:])

        # _u is an abbreviation for undersampled
        t_u = np.hstack((t1,t2))
        rho_u = np.hstack((rho1,rho2))
        
        tb = time.time()
        self.slicing_time += tb-t0

        t0 = time.time()
        exp_val_u = self.dipole_down(rho_u,rho_in.manifold_key,pulse_number = pulse_number,
                                     ket_flag = ket_flag)
        
        tb = time.time()
        self.expectation_time += tb-t0

        t0 = time.time()

        # Interpolate expectation value back onto the full t-grid
        if u != 1:
            # Often must extrapolate the final point
            exp_val_interp = scipy.interpolate.interp1d(t_u,exp_val_u,kind='cubic',fill_value='extrapolate')
            exp_val = exp_val_interp(t)
        else:
            exp_val = exp_val_u
        # print(exp_val.size/exp_val_u.size)
        tb = time.time()
        self.interpolation_time += tb-t0
        
        # Initialize return array with zeros
        ret_val = np.zeros(self.t.size,dtype='complex')
        
        # set non-zero values using t_slice
        ret_val[pulse_start_ind:] = exp_val
        return ret_val

    def integrated_dipole_expectation(self,rho_in,*,ket_flag=True):
        """Computes the expectation value of the dipole operator"""
        
        pulse_number = -1
        
        pulse_time = self.pulse_times[pulse_number]
        t = pulse_time + self.efield_times[pulse_number]

        if ket_flag:
            center = - self.centers[pulse_number]
        else:
            center = self.centers[pulse_number]

        rho = rho_in(t)

        rho_nonzero = rho_in.bool_mask
        try:
            ev = self.eigenvalues['all_manifolds'][rho_nonzero]
        except KeyError:
            ev = self.eigenvalues[rho_in.manifold_key][rho_nonzero]

        rho = rho * np.exp((ev[:,np.newaxis] - 1j*center)*t)

        rho_dict = {'bool_mask':rho_nonzero,'rho':rho,'manifold_key':rho_in.manifold_key}

        t0 = time.time()
        exp_val = self.dipole_down(rho_dict,pulse_number = pulse_number,
                                     ket_flag = ket_flag)
        
        tb = time.time()
        self.expectation_time += tb-t0

        return exp_val

    def get_local_oscillator(self):
        local_oscillator_number = -1
        efield_t = self.efield_times[local_oscillator_number]
        efield = self.efields[local_oscillator_number]

        if efield_t.size == 1:
            # Impulsive limit: delta in time is flat in frequency
            efield_ft = np.ones(self.w.size)*efield
            return efield_ft
        
        e_dt = efield_t[1] - efield_t[0]
        dt = self.t[1] - self.t[0]
            
        if (np.isclose(e_dt,dt) and efield_t[-1] <= self.t[-1]):
            full_efield = np.zeros(self.t.size,dtype='complex')

            # the local oscillator sets the "zero" on the clock
            pulse_time_ind = np.argmin(np.abs(self.t))

            pulse_start_ind = pulse_time_ind - efield_t.size//2
            pulse_end_ind = pulse_time_ind + efield_t.size//2 + efield_t.size%2

            t_slice = slice(pulse_start_ind, pulse_end_ind,None)
            
            full_efield[t_slice] = efield
            efield_ft = fftshift(ifft(ifftshift(full_efield)))*full_efield.size * dt
        else:
            efield_ft = fftshift(ifft(ifftshift(efield))) * efield.size * e_dt
            efield_w = fftshift(fftfreq(efield_t.size,d=e_dt)) * 2 * np.pi
            fill_value = (efield_ft[0],efield_ft[-1])
            f = sinterp1d(efield_w,efield_ft,fill_value = fill_value,
                          bounds_error=False,kind='quadratic')
            efield_ft = f(self.w)

        return efield_ft
    
    def polarization_to_signal(self,P_of_t_in,*,
                                local_oscillator_number = -1,undersample_factor = 1):
        """This function generates a frequency-resolved signal from a polarization field
           local_oscillator_number - usually the local oscillator will be the last pulse 
                                     in the list self.efields"""
        undersample_slice = slice(None,None,undersample_factor)
        P_of_t = P_of_t_in[undersample_slice].copy()
        t = self.t[undersample_slice]
        dt = t[1] - t[0]
        pulse_time = self.pulse_times[local_oscillator_number]
        efield_t = self.efield_times[local_oscillator_number]
        
        pulse_time_ind = np.argmin(np.abs(self.t))
        
        efield = self.get_local_oscillator()

        halfway = self.w.size//2
        pm = self.w.size//(2*undersample_factor)
        efield_min_ind = halfway - pm
        efield_max_ind = halfway + pm + self.w.size%2
        efield = efield[efield_min_ind:efield_max_ind]

        P_of_w = fftshift(ifft(ifftshift(P_of_t)))*len(P_of_t)*dt#/np.sqrt(2*np.pi)

        signal = P_of_w * np.conjugate(efield)
        if not self.return_complex_signal:
            return np.imag(signal)
        else:
            return -1j*signal

    def integrated_polarization_to_signal(self,P,*,
                                local_oscillator_number = -1):
        """This function generates a frequency-resolved signal from a polarization field
           local_oscillator_number - usually the local oscillator will be the last pulse 
                                     in the list self.efields"""
        efield_t = self.efield_times[local_oscillator_number]

        efield = self.efields[local_oscillator_number]

        signal = np.trapz(P * np.conjugate(efield),x=efield_t)
        if not self.return_complex_signal:
            return np.imag(signal)
        else:
            return -1j*signal

    def save(self,file_name,pulse_delay_names,*,use_base_path=True):
        if use_base_path:
            file_name = os.path.join(self.base_path,file_name)
        save_dict = {}
        for name,delays in zip(pulse_delay_names,self.all_pulse_delays):
            save_dict[name] = delays
        if self.detection_type == 'polarization':
            save_dict['wt'] = self.w
        save_dict['signal'] = self.signal
        save_dict['signal_calculation_time'] = self.calculation_time
        np.savez(file_name,**save_dict)
