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

from ufss import DiagramGenerator, HeavisideConvolve

def set_identical_efields(obj):
    """This should contain more"""
    obj.efields = []

class rho_container:
    def __init__(self,t,rho,bool_mask,pulse_number,manifold_key,*,interp_kind='linear',
                 interp_left_fill=0):
        self.bool_mask = bool_mask
        self.pulse_number = pulse_number
        self.manifold_key = manifold_key
        
        
        if t.size == 1:
            self.impulsive = True
            n, M = rho.shape
            self.M = M+2
            self.n = n
            self.t = np.array([-1,0,1],dtype='float') * np.spacing(t[0]) + t[0]
            rho_new = np.zeros((n,3),dtype='complex')
            rho_new[:,0] = interp_left_fill
            rho_new[:,1] = (1 + interp_left_fill)/2 * rho[:,0]
            rho_new[:,2] = rho[:,0]
            self.asymptote = rho[:,-1]
            self.rho = rho_new
            
            self._rho = rho_new

            self.rho_fun = self.impulsive_rhofun(self.asymptote,left_fill = interp_left_fill)
            
        else:
            self.impulsive = False
            self.t = t
            self.rho = rho
            self._rho = self.extend(rho,left_fill = interp_left_fill)
            self.rho_fun = self.make_interpolant(kind=interp_kind,
                                                 left_fill=interp_left_fill)
        
    def extend(self,rho,*,left_fill = 0):
        n, M = rho.shape
        self.M = M
        self.n = n
        new_rho = np.zeros((n,3*M),dtype='complex')
        new_rho[:,0:M] = left_fill
        new_rho[:,M:2*M] = rho
        asymptote = rho[:,-1]
        self.asymptote = asymptote
        new_rho[:,2*M:] = asymptote[:,np.newaxis]

        return new_rho

    def make_interpolant(self,*, kind='cubic', left_fill=0):
        """Interpolates density matrix and pads using 0 to the left
            and rho[-1] to the right
"""
        left_fill = np.ones(self.n,dtype='complex')*left_fill
        right_fill = self.rho[:,-1]
        return sinterp1d(self.t,self.rho,fill_value = (left_fill,right_fill),
                         assume_sorted=True,bounds_error=False,kind=kind)

    def impulsive_rhofun(self,asymptote,left_fill=0):
        if left_fill == 0:
            def f(t):
                return asymptote[:,np.newaxis] * np.heaviside(t-self.t[1],0.5)[np.newaxis,:]
        else:
            def f(t):
                try:
                    return asymptote[:,np.newaxis] * np.ones(len(t))[np.newaxis,:]
                except:
                    return asymptote[:,np.newaxis]
        return f

    def __call__(self,t):
        if self.impulsive:
            # The 
            return self.rho_fun(t)

        # the following logic is designed to speed up calculations outside of the impulsive limit
        if type(t) is np.ndarray:
            if t[0] > self.t[-1]:
                if t.size <= self.M:
                    ans = self._rho[:,-t.size:].copy()
                else:
                    ans = np.ones(t.size,dtype='complex')[np.newaxis,:] * self.asymptote[:,np.newaxis]
            elif t[-1] < self.t[0]:
                if t.size <= self.M:
                    ans = self._rho[:,:t.size].copy()
                else:
                    ans = np.zeros((self.n,t.size),dtype='complex')
            elif t.size == self.M:
                if np.allclose(t,self.t):
                    ans = self.rho.copy()
                else:
                    ans = self.rho_fun(t)
            else:
                ans = self.rho_fun(t)
        else:
                ans = self.rho_fun(t)
        return ans

    def __getitem__(self,inds):
        return self._rho[:,inds].copy()

class DensityMatrices(DiagramGenerator):
    """This class is designed to calculate perturbative wavepackets in the
        light-matter interaction given the eigenvalues of the unperturbed 
        hamiltonian and the material dipole operator evaluated in the
        eigenbasis of the unperturbed hamiltonian.

    Args:
        file_path (string): path to folder containing eigenvalues and the
            dipole operator for the system Hamiltonian
        num_conv_points (int): number of desired points for linear 
            convolution. Also number of points used to resolve all optical
            pulse shapes
        dt (float): time spacing used to resolve the shape of all optical
            pulses
        initial_state (int): index of initial state for psi^0

"""
    def __init__(self,file_path,*,
                 initial_state=0, total_num_time_points = None,
                 detection_type = 'polarization'):
        self.slicing_time = 0
        self.interpolation_time = 0
        self.expectation_time = 0
        self.next_order_expectation_time = 0
        self.convolution_time = 0
        self.extend_time = 0
        self.mask_time = 0
        self.dipole_time = 0
        self.automation_time = 0
        self.diagram_to_signal_time = 0
        self.diagram_generation_counter = 0
        self.number_of_diagrams_calculated = 0
        
        self.base_path = file_path

        self.undersample_factor = 1

        self.gamma_res = 6.91

        self.load_eigensystem()

        self.load_mu()

        if detection_type == 'polarization':
            self.rho_to_signal = self.polarization_detection_rho_to_signal
            self.return_complex_signal = False
        elif detection_type == 'complex_polarization':
            self.rho_to_signal = self.polarization_detection_rho_to_signal
            self.return_complex_signal = True
            detection_type = 'polarization'
        elif detection_type == 'integrated_polarization':
            self.rho_to_signal = self.integrated_polarization_detection_rho_to_signal
        elif detection_type == 'fluorescence':
            self.rho_to_signal = self.fluorescence_detection_rho_to_signal

        DiagramGenerator.__init__(self,detection_type=detection_type)
        self.KB_dict = {'Bu':self.bra_up,'Ku':self.ket_up,'Kd':self.ket_down,'Bd':self.bra_down}

        # Code will not actually function until the following empty lists are set by the user
        self.efields = [] #initialize empty list of electric field shapes
        self.efield_times = [] #initialize empty list of times assoicated with each electric field shape
        self.dts = [] #initialize empty list of time spacings associated with each electric field shape
        self.polarization_sequence = [] #initialize empty polarization sequence
        self.pulse_times = [] #initialize empty list of pulse arrival times
        self.centers = [] #initialize empty list of pulse center frequencies
        self.efield_wavevectors = []
        self.rhos = dict()

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
            if len(signal_shape) == 1:
                pass
            else:
                signal_shape[-1] = self.w.size
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
        
    def calculate_signal_all_delays(self):
        t0 = time.time()
        num_delays = len(self.all_pulse_delays)
        num_pulses = len(self.efields)

        all_delay_combinations = list(itertools.product(*self.all_pulse_delays))
        
        signal_shape = [delays.size for delays in self.all_pulse_delays]
        if self.detection_type == 'polarization':
            signal = np.zeros((len(all_delay_combinations),self.w.size),dtype='complex')
            if len(signal_shape) == 1:
                pass
            else:
                signal_shape[-1] = self.w.size
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
        max_efield_t = max([np.max(u) for u in self.efield_times]) * 1.05
        max_pos_t = max(max_pos_t,max_efield_t)
        if dt == 'auto':
            dt = self.dts[-1] # signal detection bandwidth determined by local oscillator
        n = int(max_pos_t/dt)
        self.t = np.arange(-n,n+1,1)*dt
        self.w = fftshift(fftfreq(self.t.size,d=dt)*2*np.pi)

    def execute_diagram(self,instructions):
        self.number_of_diagrams_calculated += 1
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

    def set_identical_gaussians(self,sigma_t,c,phase_discrimination):
        """
"""
        L = len(phase_discrimination) # number of pulses
        # Delta = 10 and M = 41 hard-coded in
        efield_t = np.linspace(-5,5,num=41)
        times = [efield_t] * L
        self.set_polarization_sequence(['x'] * L)
        centers = [c] * L
        ef = np.exp(-efield_t**2/(2*sigma_t**2))
        efields = [ef for i in L]

        self.set_efields(times,efields,centers,phase_discrimination,
                         reset_rhos = True,plot_fields = False)

    def set_current_diagram_instructions(self,times):
        self.diagram_generation_counter += 1
        t0a = time.time()
        efield_permutations = self.relevant_permutations(times)
        diagram_instructions = []
        for perm in efield_permutations:
            diagram_instructions += self.instructions_from_permutation(perm)
        self.current_instructions = diagram_instructions
        t0b = time.time()
        self.diagram_generation_time = t0b - t0a

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

        new = np.array(arrival_times)
        new_pulse_sequence = np.argsort(new)
        new_pulse_overlap_array = np.zeros((len(times),len(times)),dtype='bool')
        for i in range(len(times)):
            ti = times[i]
            for j in range(i+1,len(times)):
                tj = times[j]
                if ti[0] >= tj[0] and ti[0] <= tj[-1]:
                    new_pulse_overlap_array[i,j] = True
                elif ti[-1] >= tj[0] and ti[-1] <= tj[-1]:
                    new_pulse_overlap_array[i,j] = True
        try:
            logic_statement = (np.allclose(new_pulse_overlap_array,self.pulse_overlap_array)
                and np.allclose(new_pulse_sequence,self.pulse_sequence))
            # logic_statement = False
            if logic_statement:
                pass
            else:
                self.set_current_diagram_instructions(times)
        except:
            self.set_current_diagram_instructions(times)

        if len(self.current_instructions) == 0:
            print(arrival_times)

        t1 = time.time()
        try:
            instructions = self.current_instructions[0]
            signal = self.execute_diagram(instructions)
            for instructions in self.current_instructions[1:]:
                signal += self.execute_diagram(instructions)
        except IndexError:
            print('error')
            signal = 0

        t2 = time.time()
        self.automation_time += t1-t0
        self.diagram_to_signal_time += t2-t1

        self.pulse_sequence = new_pulse_sequence
        self.pulse_overlap_array = new_pulse_overlap_array
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
        if self.detection_type == 'polarization':
            times = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times)-1)]
        elif self.detection_type == 'integrated_polarization':
            times = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times)-1)]
        elif self.detection_type == 'fluorescence':
            times = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times))]
            
        self.current_instructions = diagram_instructions
        instructions = diagram_instructions[0]
        signal = self.execute_diagram(instructions)
        for instructions in diagram_instructions[1:]:
            signal += self.execute_diagram(instructions)
        return signal

    def polarization_detection_rho_to_signal(self,rho):
        p_of_t = self.dipole_expectation(rho,ket_flag=True)
        return self.polarization_to_signal(p_of_t,local_oscillator_number=-1)

    def integrated_polarization_detection_rho_to_signal(self,rho):    
        p = self.integrated_dipole_expectation(rho,ket_flag=True)
        return self.integrated_polarization_to_signal(p,local_oscillator_number=-1)

    def fluorescence_detection_rho_to_signal(self,rho):
        n_nonzero = rho['bool_mask']
        L_size = self.eigenvalues[0].size
        H_size = int(np.sqrt(L_size))

        # move back to the basis the Liouvillian was written in
        rho = self.eigenvectors['right'][:,n_nonzero].dot(rho['rho'][:,-1])

        # reshape rho into a normal density matrix representation
        rho = rho.reshape((H_size,H_size))

        fluorescence_yield = np.array([0,1,1,self.f_yield])

        signal = np.dot(np.diagonal(rho),fluorescence_yield)
        
        return signal

    def interpolate_rho(self,t,rho,kind='cubic',left_fill=0):
        """Interpolates density matrix at given inputs, and pads using 0 to the left
            and rho[-1] to the right
"""
        left_fill = np.ones(rho.shape[0],dtype='complex')*left_fill
        right_fill = rho[:,-1]
        return sinterp1d(t,rho,fill_value = (left_fill,right_fill),assume_sorted=True,bounds_error=False,kind=kind)

    def set_efields(self,times_list,efields_list,centers_list,phase_discrimination,*,reset_rhos = True,
                    plot_fields = False):
        self.efield_times = times_list
        self.efields = efields_list
        self.centers = centers_list
        self.set_phase_discrimination(phase_discrimination)
        self.dts = []
        self.efield_frequencies = []
        self.heaviside_convolve_list = []
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
            self.heaviside_convolve_list.append(HeavisideConvolve(t.size))

        # Initialize unperturbed wavefunction
        self.set_rho0()

        self.dt = self.dts[0]

        if self.detection_type == 'polarization' or 'integrated_polarization':
            try:
                self.local_oscillator = self.efields[-1].copy()
            except:
                self.local_oscillator = copy.deepcopy(self.efields[-1])

        # for i in range(len(self.efields)):
        #     self.check_efield_resolution(i,plot_fields = plot_fields)
                

    def check_efield_resolution(self,pulse_number,*,plot_fields = False):
        efield = self.efields[pulse_number]
        if len(efield) == 1:
            return None
        dt = self.dts[pulse_number]
        efield_t = self.efield_times[pulse_number]
        efield_w = fftshift(fftfreq(efield_t.size,d=dt))*2*np.pi
        efield_tail = np.max(np.abs([efield[0],efield[-1]]))


        if efield_tail > np.max(np.abs(efield))/100:
            warnings.warn('Consider using larger time interval, pulse does not decay to less than 1% of maximum value in time domain')
            
        efield_fft = fftshift(fft(ifftshift(efield)))*dt
        efield_fft_tail = np.max(np.abs([efield_fft[0],efield_fft[-1]]))
        
        if efield_fft_tail > np.max(np.abs(efield_fft))/100:
            warnings.warn('''Consider using smaller value of dt, pulse does not decay to less than 1% of maximum value in frequency domain''')

        if plot_fields:
            fig, axes = plt.subplots(1,2)
            l1,l2, = axes[0].plot(efield_t,np.real(efield),efield_t,np.imag(efield))
            plt.legend([l1,l2],['Real','Imag'])
            axes[1].plot(efield_w,np.real(efield_fft),efield_w,np.imag(efield_fft))

            axes[0].set_ylabel('Electric field Amp')
            axes[0].set_xlabel('Time ($\omega_0^{-1})$')
            axes[1].set_xlabel('Frequency ($\omega_0$)')

            fig.suptitle('Check that efield is well-resolved in time and frequency')
            plt.show()

    def set_local_oscillator_phase(self,phase):
        self.efields[-1] = np.exp(1j*phase) * self.local_oscillator

    def get_rho_site_basis(self,t,key,*,reshape=True):
        rho_obj = self.rhos[key]
        manifold_key = rho_obj.manifold_key
        mask = rho_obj.bool_mask
        
        try:
            size_L = self.eigenvalues['all_manifolds'].size
            e = self.eigenvalues['all_manifolds'][mask]
            ev = self.eigenvectors['all_manifolds'][:,mask]
        except KeyError:
            size_L = self.eigenvalues[manifold_key].size
            e = self.eigenvalues[manifold_key][mask]
            ev = self.eigenvectors[manifold_key][:,mask]
        
        size_H = int(np.sqrt(size_L))
        rho = rho_obj(t)*np.exp(e[:,np.newaxis]*t[np.newaxis,:])
        new_rho = ev.dot(rho)
        if reshape:
            new_rho = new_rho.reshape(size_H,size_H,rho.shape[-1])
        return new_rho

    def get_rho_site_basis_by_order(self,t,order):
        keys = self.rhos.keys()
        order_keys = []
        for key in keys:
            if len(key) == 3*order:
                order_keys.append(key)
        rho_total = self.get_rho_site_basis(t,order_keys.pop(0))
        for key in order_keys:
            rho_total += self.get_rho_site_basis(t,key)

        return rho_total

    def set_rho0(self):
        """Creates the unperturbed wavefunction. This code does not 
            support initial states that are coherent super-positions of 
            eigenstates. To perform thermal averaging, recalculate spectra 
            for each initial state that contributes to the thermal ensemble.
"""
        try:
            ev = self.eigenvalues['all_manifolds']
        except KeyError:
            ev = self.eigenvalues['00']

        t = self.efield_times[0] # can be anything of the correct length
        
        initial_state = np.where(ev==0)[0]
        rho0 = np.ones((1,t.size),dtype=complex)
        bool_mask = np.zeros(ev.size,dtype='bool')
        bool_mask[initial_state] = True

        

        self.rho0 = rho_container(t,rho0,bool_mask,None,'00',interp_kind='zero',
                                  interp_left_fill=1)

    def set_rho0_manual(self,manifold_key,bool_mask,weights):
        """
"""
        ev = self.eigenvalues[manifold_key][bool_mask]

        t = self.efield_times[0] # can be anything of the correct length
        
        initial_state = np.where(ev==0)[0]
        rho0 = np.ones((1,t.size),dtype=complex)*weights[:,np.newaxis]

        

        self.rho0 = rho_container(t,rho0,bool_mask,None,manifold_key,interp_kind='zero',
                                  interp_left_fill=1)

    def get_closest_index_and_value(self,value,array):
        """Given an array and a desired value, finds the closest actual value
stored in that array, and returns that value, along with its corresponding 
array index
"""
        index = np.argmin(np.abs(array - value))
        value = array[index]
        return index, value

#     def load_eigenvalues(self):
#         """Load in known eigenvalues. Must be stored as a numpy archive file,
# with keys: GSM, SEM, and optionally DEM.  The eigenvalues for each manifold
# must be 1d arrays, and are assumed to be ordered by increasing energy. The
# energy difference between the lowest energy ground state and the lowest 
# energy singly-excited state should be set to 0
# """
#         eigval_save_name = os.path.join(self.base_path,'eigenvalues.npz')
#         with np.load(eigval_save_name) as eigval_archive:
#             self.manifolds = eigval_archive.keys()
#             self.eigenvalues = [eigval_archive[key] for key in self.manifolds]
        
#         ### store original eigenvalues for recentering purposes
#         self.original_eigenvalues = copy.deepcopy(self.eigenvalues)

#     def load_eigenvectors(self):
#         """Only need the right eigenvectors"""
#         eigvec_save_name = os.path.join(self.base_path,'right_eigenvectors.npz')
#         with np.load(eigvec_save_name) as eigvec_archive:
#             self.right_eigenvectors = {'right':eigvec_archive['all_manifolds']}

    def load_eigensystem(self):
        """Load in known eigenvalues. Must be stored as a numpy archive file,
with keys: GSM, SEM, and optionally DEM.  The eigenvalues for each manifold
must be 1d arrays, and are assumed to be ordered by increasing energy. The
energy difference between the lowest energy ground state and the lowest 
energy singly-excited state should be set to 0
"""
        eigval_save_name = os.path.join(self.base_path,'eigenvalues.npz')
        eigvec_save_name = os.path.join(self.base_path,'right_eigenvectors.npz')
        with np.load(eigval_save_name) as eigval_archive:
            self.manifolds = eigval_archive.keys()
            self.eigenvalues = {key:eigval_archive[key] for key in self.manifolds}
        with np.load(eigvec_save_name) as eigvec_archive:
            self.eigenvectors = {key:eigvec_archive[key] for key in self.manifolds}
        
        ### store original eigenvalues for recentering purposes
        self.original_eigenvalues = copy.deepcopy(self.eigenvalues)

    # def load_mu_by_manifold(self):
    #     file_name = os.path.join(self.base_path,'mu_by_manifold.npz')
    #     self.mu_by_manifold = dict()
    #     self.boolean_mu_by_manifold = dict()
    #     with np.load(file_name) as mu_archive:
    #         for key in mu_archive.keys():
    #             self.mu_by_manifold[key] = mu_archive[key]
    #             self.boolean_mu_by_manifold[key] = np.ones(mu_archive[key].shape[:2],dtype='bool')

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
            be stored as a .npz file, and must contain at least one array, each with three 
            indices: (new manifold eigenfunction, old manifold eigenfunction, 
            cartesian coordinate)."""
        file_name = os.path.join(self.base_path,'mu.npz')
        file_name_pruned = os.path.join(self.base_path,'mu_pruned.npz')
        file_name_bool = os.path.join(self.base_path,'mu_boolean.npz')
        try:
            file_name = file_name_pruned
            mu_boolean_archive = np.load(file_name_bool)
            # self.mu_boolean = {'ket':mu_boolean_archive['ket'],'bra':mu_boolean_archive['bra']}
            with np.load(file_name_bool) as mu_boolean_archive:
                self.mu_boolean = {key:mu_boolean_archive[key] for key in mu_boolean_archive.keys()}
            pruned = True
        except FileNotFoundError:
            pruned = False
        # self.mu = {'ket':mu_archive['ket'],'bra':mu_archive['bra']}
        with np.load(file_name) as mu_archive:
            self.mu = {key:mu_archive[key] for key in mu_archive.keys()}
        if pruned == False:
            self.mu_boolean = dict()
            for key in self.mu.keys():
                self.mu_boolean[key] = np.ones(self.mu[key].shape[:2],dtype='bool')
        
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


    ### Tools for recursively calculating perturbed wavepackets using TDPT

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
            boolean_matrix = self.mu_boolean[key]
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
            boolean_matrix = self.mu_boolean[key]
            
        if np.all(pol == x):
            overlap_matrix = mu[:,:,0].copy()
        elif np.all(pol == y):
            overlap_matrix = mu[:,:,1].copy()
        elif np.all(pol == z):
            overlap_matrix = mu[:,:,2].copy()
        else:
            overlap_matrix = np.tensordot(mu,pol,axes=(-1,0))

        t1 = time.time()
        self.dipole_time += t1-t0

        return boolean_matrix, overlap_matrix

#     def electric_field_mask(self,pulse_number,conjugate_flag=False):
#         """This method determines which molecular transitions will be 
# supported by the electric field.  We assume that the electric field has
# 0 amplitude outside the minimum and maximum frequency immplied by the 
# choice of dt and num_conv_points.  Otherwise we will inadvertently 
# alias transitions onto nonzero electric field amplitudes.
# """
#         efield_t = self.efield_times[pulse_number]
#         efield_w = self.efield_frequencies[pulse_number]
#         if conjugate_flag:
#             center = -self.centers[pulse_number]
#         else:
#             center = self.centers[pulse_number]
        
#         eig = self.eigenvalues[0]
#         # imag part corresponds to the energetic transitions
#         diff = np.imag(eig[:,np.newaxis] - eig[np.newaxis,:])
        
#         if efield_t.size == 1:
#             mask = np.ones(diff.shape,dtype='bool')
#             # try:
#             #     inds_allowed = np.where((diff + center > -self.RWA_width/2) & (diff + center < self.RWA_width/2))
#             #     mask = np.zeros(diff.shape,dtype='bool')
#             #     mask[inds_allowed] = 1
#             # except AttributeError:
#             #     raise Exception('When working with impulsive pulses you must set object attribute RWA_width')
#         else:
#             # The only transitions allowed by the electric field shape are
#             inds_allowed = np.where((diff + center > efield_w[0]) & (diff + center < efield_w[-1]))
#             mask = np.zeros(diff.shape,dtype='bool')
#             mask[inds_allowed] = 1

#         return mask

    def electric_field_mask(self,pulse_number,key,conjugate_flag=False):
        """This method determines which molecular transitions will be 
supported by the electric field.  We assume that the electric field has
0 amplitude outside the minimum and maximum frequency immplied by the 
choice of dt and num_conv_points.  Otherwise we will inadvertently 
alias transitions onto nonzero electric field amplitudes.
"""
        starting_key, ending_key = key.split('_to_')
        efield_t = self.efield_times[pulse_number]
        efield_w = self.efield_frequencies[pulse_number]
        if conjugate_flag:
            center = -self.centers[pulse_number]
        else:
            center = self.centers[pulse_number]
        try:
            eig_starting = self.eigenvalues['all_manifolds']
            eig_ending = self.eigenvalues['all_manifolds']
        except KeyError:
            eig_starting = self.eigenvalues[starting_key]
            eig_ending = self.eigenvalues[ending_key]
        # imag part corresponds to the energetic transitions
        diff = np.imag(eig_ending[:,np.newaxis] - eig_starting[np.newaxis,:])
        
        if efield_t.size == 1:
            mask = np.ones(diff.shape,dtype='bool')
        else:
            # The only transitions allowed by the electric field shape are
            inds_allowed = np.where((diff + center > efield_w[0]) & (diff + center < efield_w[-1]))
            mask = np.zeros(diff.shape,dtype='bool')
            mask[inds_allowed] = 1

        return mask

    def mask_dipole_matrix(self,boolean_matrix,overlap_matrix,
                           starting_manifold_mask,*,next_manifold_mask = None):
        """Takes as input the boolean_matrix and the overlap matrix that it 
            corresponds to. Also requires the starting manifold mask, which specifies
            which states have non-zero amplitude, given the signal tolerance requested.
            Trims off unnecessary starting elements, and ending elements. If 
            next_manifold_mask is None, then the masking is done automatically
            based upon which overlap elements are nonzero. If next_manifold_mask is
            a 1D numpy boolean array, it is used as the mask for next manifold."""
        t0 = time.time()
        if np.all(starting_manifold_mask == True):
            pass
        else:
            boolean_matrix = boolean_matrix[:,starting_manifold_mask]
            overlap_matrix = overlap_matrix[:,starting_manifold_mask]

        #Determine the nonzero elements of the new psi, in the
        #eigenenergy basis, n_nonzero
        if type(next_manifold_mask) is np.ndarray:
            n_nonzero = next_manifold_mask
        else:
            n_nonzero = np.any(boolean_matrix,axis=1)
        if np.all(n_nonzero == True):
            pass
        else:
            overlap_matrix = overlap_matrix[n_nonzero,:]

        t1 = time.time()
        self.mask_time += t1-t0

        return overlap_matrix, n_nonzero

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
    
    def next_order(self,rho_in,*,ket_flag=True,up_flag=True,
                   new_manifold_mask = None,pulse_number = 0):
        """This function connects rho_p to rho_pj^(*) using a DFT convolution algorithm.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold
        
        Return:
            rho_dict (rho_container): next-order density matrix
"""
        pulse_time = self.pulse_times[pulse_number]
        t = self.efield_times[pulse_number] + pulse_time
        dt = self.dts[pulse_number]
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
        
        m_nonzero = rho_in.bool_mask
        try:
            ev1 = self.eigenvalues['all_manifolds']
            ev2 = self.eigenvalues['all_manifolds']
        except KeyError:
            ev1 = self.eigenvalues[old_manifold_key]
            ev2 = self.eigenvalues[new_manifold_key]

        exp_factor1 = np.exp( (ev1[m_nonzero,np.newaxis] - 1j*center)*t[np.newaxis,:])
        
        rho = rho_in(t) * exp_factor1
        
        boolean_matrix, overlap_matrix = self.dipole_matrix(pulse_number,mu_key,ket_flag=ket_flag,up_flag=up_flag)

        e_mask = self.electric_field_mask(pulse_number,mu_key,conjugate_flag=conjugate_flag)
        boolean_matrix = boolean_matrix * e_mask
        overlap_matrix = overlap_matrix * e_mask

        overlap_matrix, n_nonzero = self.mask_dipole_matrix(boolean_matrix,overlap_matrix,m_nonzero,
                                                                next_manifold_mask = new_manifold_mask)
        
        t0 = time.time()
        rho = overlap_matrix.dot(rho)
        
        t1 = time.time()
        self.next_order_expectation_time += t1-t0

        exp_factor2 = np.exp(-ev2[n_nonzero,np.newaxis]*t[np.newaxis,:])

        rho *= exp_factor2

        t0 = time.time()

        M = self.efield_times[pulse_number].size

        fft_convolve_fun = self.heaviside_convolve_list[pulse_number].fft_convolve2
        # fft_convolve_fun = self.fft_convolve2

        if M == 1:
            rho *= self.efields[pulse_number]
        else:
            if conjugate_flag:
                efield = np.conjugate(self.efields[pulse_number])
            else:
                efield = self.efields[pulse_number]
            rho = fft_convolve_fun(rho * efield[np.newaxis,:],d=dt)

        t1 = time.time()
        self.convolution_time += t1-t0

        # i/hbar Straight from perturbation theory
        if ket_flag:
            rho *= 1j
        else:
            rho *= -1j

        rho_out = rho_container(t,rho,n_nonzero,pulse_number,new_manifold_key)
    
        return rho_out
            
    def ket_up(self,rho_in,*,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) 
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (rho_container): output from method next_order
"""

        return self.next_order(rho_in,ket_flag=True,up_flag=True,
                               new_manifold_mask = new_manifold_mask,
                               pulse_number = pulse_number)

    def ket_down(self,rho_in,*,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) 
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (rho_container): output from method next_order
"""

        return self.next_order(rho_in,ket_flag=True,up_flag=False,
                               new_manifold_mask = new_manifold_mask,
                               pulse_number = pulse_number)

    def bra_up(self,rho_in,*,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (rho_container): output from method next_order
"""

        return self.next_order(rho_in,ket_flag=False,up_flag=True,
                               new_manifold_mask = new_manifold_mask,
                               pulse_number = pulse_number)

    def bra_down(self,rho_in,*,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (rho_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (rho_container): output from method next_order
"""

        return self.next_order(rho_in,ket_flag=False,up_flag=False,
                               new_manifold_mask = new_manifold_mask,
                               pulse_number = pulse_number)

    ### Tools for taking the expectation value of the dipole operator with perturbed wavepackets
    
    def dipole_down(self,rho_dict,*,new_manifold_mask = None,pulse_number = -1,
                    ket_flag=True):
        """This method is similar to the method down, but does not involve 
            the electric field shape or convolutions. It is the action of the 
            dipole operator on the ket-side without TDPT effects.  It also includes
            the dot product of the final electric field polarization vector."""
        old_manifold_key = rho_dict['manifold_key']
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
            manifold_change = np.array([change,0],dtype=int)
        else:
            center = self.centers[pulse_number]
            conjugate_flag = False
            manifold_change = np.array([0,change],dtype=int)

        rho_in = rho_dict['rho']
        
        m_nonzero = rho_dict['bool_mask']
        
        boolean_matrix, overlap_matrix = self.dipole_matrix(pulse_number,mu_key,ket_flag=True,up_flag=False)

        e_mask = self.electric_field_mask(pulse_number,mu_key,conjugate_flag=conjugate_flag)
        boolean_matrix = boolean_matrix * e_mask
        overlap_matrix = overlap_matrix * e_mask

        overlap_matrix, n_nonzero = self.mask_dipole_matrix(boolean_matrix,overlap_matrix,m_nonzero,
                                                                next_manifold_mask = new_manifold_mask)
        t0 = time.time()
        rho = overlap_matrix.dot(rho_in)
                
        t1 = time.time()

        try:
            L_size = self.eigenvalues['all_manifolds'].size
        except KeyError:
            L_size = self.eigenvalues[new_manifold_key].size
        H_size = int(np.sqrt(L_size))

        # move back to the basis the Liouvillian was written in
        try:
            rho = self.eigenvectors['all_manifolds'][:,n_nonzero].dot(rho)
        except KeyError:
            rho = self.eigenvectors[new_manifold_key][:,n_nonzero].dot(rho)

        # reshape rho into a normal density matrix representation
        rho = rho.reshape((H_size,H_size,rho.shape[-1]))

        polarization_field = np.einsum('iij',rho)

        return polarization_field

    def set_undersample_factor(self,frequency_resolution):
        """dt is set by the pulse. However, the system dynamics may not require such a 
            small dt.  Therefore, this allows the user to set a requested frequency
            resolution for any spectrally resolved signals.  This assumes that the 
            undersampling is defined with respect to the last pulse, which is assumed
            to be the local oscillator"""
        # f = pi/dt
        dt = np.pi/frequency_resolution
        u = int(np.floor(dt/self.dts[-1]))
        self.undersample_factor = max(u,1)
        
    def dipole_expectation(self,rho_in,*,ket_flag=True):
        """Computes the expectation value of the dipole operator"""
        t0 = time.time()

        pulse_number = -1
        
        pulse_time = self.pulse_times[pulse_number]
        dt = self.dts[pulse_number]
        efield_t = self.efield_times[pulse_number]

        if ket_flag:
            center = - self.centers[pulse_number]
        else:
            center = self.centers[pulse_number]

        M = efield_t.size

        #pulse_time is at the center of self.t
        center_index = self.t.size//2
        pulse_start_ind = center_index - (M//2)
        pulse_end_ind = center_index + (M//2 + M%2)
        
        # The signal is zero before the final pulse arrives, and persists
        # until it decays. Therefore we avoid taking the sum at times
        # where the signal is zero.
        t = self.t[pulse_start_ind:] + pulse_time

        t1_slice = slice(pulse_start_ind,pulse_end_ind,None)
        u = self.undersample_factor
        t2_slice = slice(pulse_end_ind,None,u)
        
        t1 = self.t[t1_slice] + pulse_time
        t2 = self.t[t2_slice] + pulse_time

        rho1 = rho_in(t1)
        rho2 = rho_in(t2)

        rho_nonzero = rho_in.bool_mask
        try:
            ev = self.eigenvalues['all_manifolds'][rho_nonzero]
        except KeyError:
            ev = self.eigenvalues[rho_in.manifold_key][rho_nonzero]

        rho1 *= np.exp((ev[:,np.newaxis] - 1j*center)* t1[np.newaxis,:])

        rho2 *= np.exp((ev[:,np.newaxis] - 1j*center) * t2[np.newaxis,:])

        # _u is an abbreviation for undersampled
        t_u = np.hstack((t1,t2))
        rho_u = np.hstack((rho1,rho2))
        
        tb = time.time()
        self.slicing_time += tb-t0

        rho_dict = {'bool_mask':rho_nonzero,'rho':rho_u,'manifold_key':rho_in.manifold_key}

        t0 = time.time()
        exp_val_u = self.dipole_down(rho_dict,pulse_number = pulse_number,
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
        # the density matrix object knows which pulse caused the most recenter interaction
        pulse_number = rho_in.pulse_number
        
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

        # the local oscillator sets the "zero" on the clock
        pulse_time_ind = np.argmin(np.abs(self.t))
        efield = np.zeros(self.t.size,dtype='complex')

        if efield_t.size == 1:
            # Impulsive limit: delta in time is flat in frequency
            efield = np.ones(self.w.size)*self.efields[local_oscillator_number]
        else:
            pulse_start_ind = pulse_time_ind - efield_t.size//2
            pulse_end_ind = pulse_time_ind + efield_t.size//2 + efield_t.size%2

            t_slice = slice(pulse_start_ind, pulse_end_ind,None)
            
            efield[t_slice] = self.efields[local_oscillator_number]
            efield = fftshift(ifft(ifftshift(efield)))*efield.size*dt#/np.sqrt(2*np.pi)

        halfway = self.w.size//2
        pm = self.w.size//(2*undersample_factor)
        efield_min_ind = halfway - pm
        efield_max_ind = halfway + pm + self.w.size%2
        efield = efield[efield_min_ind:efield_max_ind]

        P_of_w = fftshift(ifft(ifftshift(P_of_t)))*P_of_t.size*dt#/np.sqrt(2*np.pi)

        signal = P_of_w * np.conjugate(efield)
        if not self.return_complex_signal:
            return np.imag(signal)
        else:
            return 1j*signal

    def integrated_polarization_to_signal(self,P,*,
                                local_oscillator_number = -1):
        """This function generates a frequency-resolved signal from a polarization field
           local_oscillator_number - usually the local oscillator will be the last pulse 
                                     in the list self.efields"""
        efield_t = self.efield_times[local_oscillator_number]

        efield = self.efields[local_oscillator_number]

        signal = np.trapz(P * np.conjugate(efield),x=efield_t)
        return np.imag(signal)

    def add_gaussian_linewidth(self,sigma):
        self.old_signal = self.signal.copy()

        sig_tau_t = fftshift(fft(ifftshift(self.old_signal,axes=(-1)),axis=-1),axes=(-1))
        sig_tau_t = sig_tau_t * (np.exp(-self.t**2/(2*sigma**2))[np.newaxis,np.newaxis,:]
                                 *np.exp(-self.t21_array**2/(2*sigma**2))[:,np.newaxis,np.newaxis])
        sig_tau_w = fftshift(ifft(ifftshift(sig_tau_t,axes=(-1)),axis=-1),axes=(-1))
        self.signal = sig_tau_w

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

    def load(self,file_name,pulse_delay_names,*,use_base_path=True):
        if use_base_path:
            file_name = os.path.join(self.base_path,file_name)
        arch = np.load(file_name)
        self.all_pulse_delays = []
        for name in pulse_delay_names:
            self.all_pulse_delays.append(arch[name])
        if self.detection_type == 'polarization':
            self.w = arch['wt']
        self.signal = arch['signal']
        self.calculation_time = arch['signal_calculation_time']
