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

from ufss import DiagramGenerator
from ufss.UF2.heaviside_convolve import HeavisideConvolve

def set_identical_efields(obj):
    """This should contain more"""
    obj.efields = []

class psi_container:
    def __init__(self,t,psi,bool_mask,pulse_number,manifold_key,*,interp_kind='linear',
                 interp_left_fill=0):
        self.bool_mask = bool_mask
        self.pulse_number = pulse_number
        self.manifold_key = manifold_key
        
        
        if t.size == 1:
            n, M = psi.shape
            self.M = M+2
            self.n = n
            self.t = np.array([-1,0,1],dtype='float') * np.spacing(t[0]) + t[0]
            psi_new = np.zeros((n,3),dtype='complex')
            psi_new[:,0] = interp_left_fill
            psi_new[:,1] = (1 + interp_left_fill)/2 * psi[:,0]
            psi_new[:,2] = psi[:,0]
            self.asymptote = psi[:,-1]
            self.psi = psi_new
            
            self._psi = psi_new

            self.psi_fun = self.impulsive_psifun(self.asymptote,left_fill = interp_left_fill)
            
        else:
            self.t = t
            self.psi = psi
            self._psi = self.extend(psi,left_fill = interp_left_fill)
            self.psi_fun = self.make_interpolant(kind=interp_kind,
                                                 left_fill=interp_left_fill)
        
    def extend(self,psi,*,left_fill = 0):
        n, M = psi.shape
        self.M = M
        self.n = n
        new_psi = np.zeros((n,3*M),dtype='complex')
        new_psi[:,0:M] = left_fill
        new_psi[:,M:2*M] = psi
        asymptote = psi[:,-1]
        self.asymptote = asymptote
        new_psi[:,2*M:] = asymptote[:,np.newaxis]

        return new_psi

    def make_interpolant(self,*, kind='cubic', left_fill=0):
        """Interpolates density matrix and pads using 0 to the left
            and psi[-1] to the right
"""
        left_fill = np.ones(self.n,dtype='complex')*left_fill
        right_fill = self.psi[:,-1]
        return sinterp1d(self.t,self.psi,fill_value = (left_fill,right_fill),
                         assume_sorted=True,bounds_error=False,kind=kind)

    def impulsive_psifun(self,asymptote,left_fill=0):
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
        if type(t) is np.ndarray:
            if t[0] > self.t[-1]:
                if t.size <= self.M:
                    ans = self._psi[:,-t.size:].copy()
                else:
                    ans = np.ones(t.size,dtype='complex')[np.newaxis,:] * self.asymptote[:,np.newaxis]
            elif t[-1] < self.t[0]:
                if t.size <= self.M:
                    ans = self._psi[:,:t.size].copy()
                else:
                    ans = np.zeros((self.n,t.size),dtype='complex')
            elif t.size == self.M:
                if np.allclose(t,self.t):
                    ans = self.psi.copy()
                else:
                    ans = self.psi_fun(t)
            else:
                ans = self.psi_fun(t)
        else:
                ans = self.psi_fun(t)
        return ans

    def __getitem__(self,inds):
        return self._psi[:,inds].copy()

class Wavepackets(DiagramGenerator):
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
    def __init__(self,file_path,*, num_conv_points=41,
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
        
        self.base_path = file_path

        self.undersample_factor = 1

        self.gamma_res = 6.91

        self.initial_state = initial_state

        self.load_eigensystem()

        self.load_mu()

        if detection_type == 'polarization':
            self.psi_to_signal = self.polarization_detection_signal
            self.return_complex_signal = False
            
        elif detection_type == 'complex_polarization':
            self.psi_to_signal = self.polarization_detection_signal
            self.return_complex_signal = True
            detection_type = 'polarization'
            
        elif detection_type == 'integrated_polarization':
            self.psi_to_signal = self.integrated_polarization_detection_signal
            self.return_complex_signal = False
            
        elif detection_type == 'fluorescence':
            self.psi_to_signal = self.fluorescence_detection_signal
            self.f_yield = f_yield #quantum yield of doubly excited manifold relative to singly excited manifold

        DiagramGenerator.__init__(self,detection_type=detection_type)
        self.K_dict = {'u':self.up,'d':self.down}

        # Code will not actually function until the following empty lists are set by the user
        self.efields = [] #initialize empty list of electric field shapes
        self.efield_times = [] #initialize empty list of times assoicated with each electric field shape
        self.dts = [] #initialize empty list of time spacings associated with each electric field shape
        self.polarization_sequence = [] #initialize empty polarization sequence
        self.pulse_times = [] #initialize empty list of pulse arrival times
        self.centers = [] #initialize empty list of pulse center frequencies
        self.efield_wavevectors = []
        self.heaviside_convolve_list = []
        self.psis = dict()

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
        max_efield_t = max([np.max(u) for u in self.efield_times]) * 1.05
        max_pos_t = max(max_pos_t,max_efield_t)
        if dt == 'auto':
            dt = self.dts[-1] # signal detection bandwidth determined by local oscillator
        n = int(max_pos_t/dt)
        self.t = np.arange(-n,n+1,1)*dt
        self.w = fftshift(fftfreq(self.t.size,d=dt)*2*np.pi)

    def execute_diagram(self,instructions):
        num_instructions = len(instructions['ket']) + len(instructions['bra'])
        
        ket = self.psi0
        bra = self.psi0
        ketname = ''
        braname = ''
        ket_instructions = instructions['ket']
        bra_instructions = instructions['bra']
        for i in range(len(ket_instructions)):
            key, num = ket_instructions[i]
            ketname += key+str(num)
            # Try to re-use previous calculations, if they exist
            try:
                new_ket = self.psis[ketname]
            except KeyError:
                new_ket = self.K_dict[key](ket,pulse_number=num)
                self.psis[ketname] = new_ket
            ket = new_ket
                

        for i in range(len(bra_instructions)):
            key, num = bra_instructions[i]
            braname += key+str(num)
            # Try to re-use previous calculations, if they exist
            try:
                new_bra = self.psis[braname]
            except KeyError:
                new_bra = self.K_dict[key](bra,pulse_number=num)
                self.psis[braname] = new_bra
            bra = new_bra
                
        sig = self.psi_to_signal(bra,ket)

        return sig

    def remove_psis_by_pulse_number(self,pulse_number):
        num = str(pulse_number)
        keys = self.psis.keys()
        keys_to_remove = []
        for key in keys:
            flag = key.find(num)
            if flag >= 0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.psis.pop(key)

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
                         reset_psis = True,plot_fields = False)

    def set_current_diagram_instructions(self,arrival_times):
        self.current_instructions = self.get_wavefunction_diagrams(arrival_times)

    def calculate_signal(self,arrival_times):
        t0 = time.time()
        try:
            old_pulse_times = self.pulse_times
            for i in range(len(old_pulse_times)):
                if old_pulse_times[i] != arrival_times[i]:
                    self.remove_psis_by_pulse_number(i)
        except AttributeError:
            pass
        
        self.pulse_times = arrival_times
        self.set_current_diagram_instructions(arrival_times)
        diagram_instructions = self.current_instructions
        if len(diagram_instructions) == 0:
            print(arrival_times)

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
                    self.remove_psis_by_pulse_number(i)
        except AttributeError:
            pass
        
        self.pulse_times = arrival_times
            
        self.current_instructions = diagram_instructions
        instructions = diagram_instructions[0]
        signal = self.execute_diagram(instructions)
        for instructions in diagram_instructions[1:]:
            signal += self.execute_diagram(instructions)
        return signal

    def polarization_detection_signal(self,bra_dict,ket_dict):
        p_of_t = self.dipole_expectation(bra_dict,ket_dict,pulse_number = -1)
        return self.polarization_to_signal(p_of_t,local_oscillator_number=-1)

    def integrated_polarization_detection_signal(self,bra_dict,ket_dict):
        p = self.integrated_dipole_expectation(bra_dict,ket_dict,pulse_number=-1)
        return self.integrated_polarization_to_signal(p,local_oscillator_number=-1)

    def fluorescence_detection_signal(self,bra_dict,ket_dict,*,time_index = -1):
        """Calculate inner product given an input bra and ket dictionary
            at a time given by the time index argument.  Default is -1, since
            2DFS is concerned with the amplitude of arriving in the given manifold
            after the pulse has finished interacting with the system."""
        bra = np.conjugate(bra_dict['psi'][:,-1])
        ket = ket_dict['psi'][:,-1]
        return np.dot(bra,ket)

    def reset(self):
        self.psis = dict()
        
    def set_efields(self,times_list,efields_list,centers_list,phase_discrimination,*,reset = True,
                    plot_fields = False):
        self.efield_times = times_list
        self.efields = efields_list
        self.centers = centers_list
        self.set_phase_discrimination(phase_discrimination)
        self.dts = []
        self.efield_frequencies = []
        if reset:
            self.reset()
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

        self.dt = self.dts[0]

        # Initialize unperturbed wavefunction
        self.set_psi0(self.initial_state)

        if self.detection_type == 'polarization' or 'integrated_polarization':
            try:
                self.local_oscillator = self.efields[-1].copy()
            except:
                self.local_oscillator = copy.deepcopy(self.efields[-1])

        # for field in self.efields:
        #     if len(field) == 1:
        #         # M = 1 is the impulsive limit
        #         pass
        #     else:
        #         self.check_efield_resolution(field,plot_fields = plot_fields)

    def check_efield_resolution(self,efield,*,plot_fields = False):
        efield_tail = np.max(np.abs([efield[0],efield[-1]]))


        if efield_tail > np.max(np.abs(efield))/100:
            warnings.warn('Consider using larger time interval, pulse does not decay to less than 1% of maximum value in time domain')
            
        efield_fft = fftshift(fft(ifftshift(efield)))*self.dt
        efield_fft_tail = np.max(np.abs([efield_fft[0],efield_fft[-1]]))
        
        if efield_fft_tail > np.max(np.abs(efield_fft))/100:
            warnings.warn('''Consider using smaller value of dt, pulse does not decay to less than 1% of maximum value in frequency domain''')

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

    def get_psi_eigen_basis(self,t,key):
        psi_obj = self.psis[key]
        mask = psi_obj.bool_mask
        all_e = self.eigenvalues[psi_obj.manifold_key]
        e = all_e[mask]
        psi = psi_obj(t)*np.exp(-1j*e[:,np.newaxis]*t[np.newaxis,:])
        full_size = all_e.size
        total_psi = np.zeros(full_size,t.size)
        total_psi[mask,:] = psi
        return total_psi

    def get_psi_site_basis(self,t,key):
        psi_obj = self.psis[key]
        mask = psi_obj.bool_mask
        manifold_num = self.manifold_key_to_number(psi_obj.manifold_key)
        e = self.eigenvalues[manifold_num][mask]
        psi = psi_obj(t)*np.exp(-1j*e[:,np.newaxis]*t[np.newaxis,:])
        ev = self.eigenvectors[manifold_num][:,mask]
        new_psi = ev.dot(psi)
        full_size = 0
        manifold_sizes = []
        for i in range(len(self.eigenvalues)):
            manifold_sizes.append(self.eigenvectors[manifold_num].shape[0])
            full_size += manifold_sizes[-1]
        total_psi = np.zeros(full_size,t.size)
        start = 0
        for i in range(len(self.eigenvalues)):
            if i == manifold_num:
                end = start + manifold_sizes[i]
                total_psi[start:end,:] = new_psi
            else:
                start += manifold_sizes[i]
        return total_psi

    def get_psi_site_basis_by_order(self,t,order):
        keys = self.psis.keys()
        order_keys = []
        for key in keys:
            if len(key) == 2*order:
                order_keys.append(key)
        psi_total = self.get_psi_site_basis(t,order_keys.pop(0))
        for key in order_keys:
            psi_total += self.get_psi_site_basis(t,key)

        return psi_total

    def set_psi0(self,initial_state):
        """Creates the unperturbed wavefunction. This code does not 
            support initial states that are coherent super-positions of 
            eigenstates. To perform thermal averaging, recalculate spectra 
            for each initial state that contributes to the thermal ensemble.
        Args:
            initial_state (int): index for initial eigenstate in GSM
"""
        # initial state must be interpreted given the fact that masking may have been done
        try:
            trimmed_indices = np.where(self.trimming_masks[0])[0]
            initial_state = np.where(trimmed_indices == initial_state)[0]
        except AttributeError:
            pass

        t = self.efield_times[0] # can be anything of the correct length

        key = self.ordered_manifolds[0]
        psi0 = np.ones((1,t.size),dtype=complex)
        bool_mask = np.zeros(self.eigenvalues[key].size,dtype='bool')
        bool_mask[initial_state] = True
 
        self.psi0 = psi_container(t,psi0,bool_mask,None,key,interp_kind='zero',
                                  interp_left_fill=1)

    def get_closest_index_and_value(self,value,array):
        """Given an array and a desired value, finds the closest actual value
stored in that array, and returns that value, along with its corresponding 
array index
"""
        index = np.argmin(np.abs(array - value))
        value = array[index]
        return index, value

    def load_eigensystem(self):
        """Load in known eigenvalues. Must be stored as a numpy archive file,
with keys: GSM, SEM, and optionally DEM.  The eigenvalues for each manifold
must be 1d arrays, and are assumed to be ordered by increasing energy. The
energy difference between the lowest energy ground state and the lowest 
energy singly-excited state should be set to 0
"""
        eigval_save_name = os.path.join(self.base_path,'eigenvalues.npz')
        eigvec_save_name = os.path.join(self.base_path,'eigenvectors.npz')
        with np.load(eigval_save_name) as eigval_archive:
            self.manifolds = list(eigval_archive.keys())
            self.eigenvalues = {key:eigval_archive[key] for key in self.manifolds}
        with np.load(eigvec_save_name) as eigvec_archive:
            self.eigenvectors = {key:eigvec_archive[key] for key in self.manifolds}

        if '0' in self.manifolds:
            self.ordered_manifolds = [str(i) for i in range(len(self.manifolds))]
        else:
            self.ordered_manifolds = ['GSM','SEM','DEM','TEM','QEM']
        
        ### store original eigenvalues for recentering purposes
        self.original_eigenvalues = copy.deepcopy(self.eigenvalues)

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
            be stored as a .npz file, and must contain at least one array, each with three 
            indices: (upper manifold eigenfunction, lower manifold eigenfunction, 
            cartesian coordinate).  So far this code supports up to three manifolds, and
            therefore up to two dipole operators (connecting between manifolds)"""
        file_name = os.path.join(self.base_path,'mu.npz')
        file_name_pruned = os.path.join(self.base_path,'mu_pruned.npz')
        file_name_bool = os.path.join(self.base_path,'mu_boolean.npz')

        try:
            mu_boolean_archive = np.load(file_name_bool)
            # self.mu_boolean = {'ket':mu_boolean_archive['ket'],'bra':mu_boolean_archive['bra']}
            with np.load(file_name_bool) as mu_boolean_archive:
                self.mu_boolean = {key:mu_boolean_archive[key] for key in mu_boolean_archive.keys()}
            pruned = True
            file_name = file_name_pruned
        except FileNotFoundError:
            pruned = False

        with np.load(file_name) as mu_archive:
            self.mu = {key:mu_archive[key] for key in mu_archive.keys()}
        if pruned == False:
            self.mu_boolean = dict()
            for key in self.mu.keys():
                self.mu_boolean[key] = np.ones(self.mu[key].shape[:2],dtype='bool')
        
    ### Setting the electric field to be used

    def set_polarization_sequence(self,polarization_list,*,reset_psis=True):
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
        if reset_psis:
            self.psis = dict()


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
            try:
                key = 'up'
                mu = self.mu[key]
                boolean_matrix = self.mu_boolean[key]
            except KeyError:
                key = 'ket_up'
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

        if not up_flag:
            overlap_matrix = overlap_matrix.T
            boolean_matrix = boolean_matrix.T

        t1 = time.time()
        self.dipole_time += t1-t0

        return boolean_matrix, overlap_matrix

    def electric_field_mask(self,pulse_number,key,conjugate_flag=False):
        """This method determines which molecular transitions will be 
supported by the electric field.  We assume that the electric field has
0 amplitude outside the minimum and maximum frequency immplied by the 
choice of dt and num_conv_points.  Otherwise we will inadvertently 
alias transitions onto nonzero electric field amplitudes.
"""
        if conjugate_flag:
            ending_key, starting_key = key.split('_to_')
        else:
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
        
        diff = eig_ending[:,np.newaxis] - eig_starting[np.newaxis,:]
        
        if efield_t.size == 1:
            mask = np.ones(diff.shape,dtype='bool')
        else:
            # The only transitions allowed by the electric field shape are
            inds_allowed = np.where((diff - center > efield_w[0]) & (diff - center < efield_w[-1]))
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

    def manifold_key_to_number(self,key):
        num = self.ordered_manifolds.index(key)
        return num

    def manifold_number_to_key(self,num):
        key = self.ordered_manifolds[num]
        return key
    
    def next_order(self,psi_in,*,up_flag=True,
                   new_manifold_mask = None,pulse_number = 0):
        """This function connects psi_p to psi_pj^(*) using a DFT convolution algorithm.

        Args:
            psi_in (psi_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold
        
        Return:
            psi_dict (psi_container): next-order wavefunction
"""
        pulse_time = self.pulse_times[pulse_number]
        t = self.efield_times[pulse_number] + pulse_time
        old_manifold_key = psi_in.manifold_key
        if up_flag:
            change = 1
            conjugate_flag = False
        else:
            change = -1
            conjugate_flag = True
        old_manifold = self.manifold_key_to_number(old_manifold_key)
        new_manifold = old_manifold + change
        new_manifold_key = self.manifold_number_to_key(new_manifold)
        if up_flag:
            mu_key = old_manifold_key + '_to_' + new_manifold_key
        else:
            mu_key = new_manifold_key + '_to_' + old_manifold_key

        if conjugate_flag:
            center = - self.centers[pulse_number]
        else:
            center = self.centers[pulse_number]
        
        m_nonzero = psi_in.bool_mask
        try:
            ev1 = self.eigenvalues['all_manifolds']
            ev2 = self.eigenvalues['all_manifolds']
        except KeyError:
            ev1 = self.eigenvalues[old_manifold_key]
            ev2 = self.eigenvalues[new_manifold_key]
            
        exp_factor_starting = np.exp( -1j*(ev1[m_nonzero,np.newaxis])*t[np.newaxis,:])
            
        
        psi = psi_in(t) * exp_factor_starting
        
        boolean_matrix, overlap_matrix = self.dipole_matrix(pulse_number,mu_key,up_flag=up_flag)

        e_mask = self.electric_field_mask(pulse_number,mu_key,conjugate_flag=conjugate_flag)
        boolean_matrix = boolean_matrix * e_mask
        overlap_matrix = overlap_matrix * e_mask

        overlap_matrix, n_nonzero = self.mask_dipole_matrix(boolean_matrix,overlap_matrix,m_nonzero,
                                                                next_manifold_mask = new_manifold_mask)
        
        t0 = time.time()
        psi = overlap_matrix.dot(psi)
        
        t1 = time.time()
        self.next_order_expectation_time += t1-t0

        exp_factor_ending = np.exp(1j*ev2[n_nonzero,np.newaxis]*t[np.newaxis,:])

        psi = psi*exp_factor_ending

        psi = psi * np.exp(-1j*center*t)

        t0 = time.time()

        M = self.efield_times[pulse_number].size

        fft_convolve_fun = self.heaviside_convolve_list[pulse_number].fft_convolve2

        if M == 1:
            psi = psi * self.efields[pulse_number]
        else:
            if conjugate_flag:
                efield = np.conjugate(self.efields[pulse_number])
            else:
                efield = self.efields[pulse_number]
            psi = fft_convolve_fun(psi * efield[np.newaxis,:],d=self.dt)

        t1 = time.time()
        self.convolution_time += t1-t0

        # i/hbar Straight from perturbation theory
        psi *= 1j


        psi_out = psi_container(t,psi,n_nonzero,pulse_number,new_manifold_key)
    
        return psi_out
            
    def up(self,psi_in,*,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            psi_in (psi_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (psi_container): output from method next_order
"""

        return self.next_order(psi_in,up_flag=True,
                               new_manifold_mask = new_manifold_mask,
                               pulse_number = pulse_number)

    def down(self,psi_in,*,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            psi_in (psi_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (psi_container): output from method next_order
"""

        return self.next_order(psi_in,up_flag=False,
                               new_manifold_mask = new_manifold_mask,
                               pulse_number = pulse_number)

    ### Tools for taking the expectation value of the dipole operator with perturbed wavepackets
    
    def dipole_down(self,psi_dict,*,new_manifold_mask = None,pulse_number = -1):
        """This method is similar to the method down, but does not involve 
            the electric field shape or convolutions. It is the action of the 
            dipole operator on the ket-side without TDPT effects.  It also includes
            the dot product of the final electric field polarization vector."""
        old_manifold_key = psi_dict['manifold_key']
        
        change = -1

        old_manifold = self.manifold_key_to_number(old_manifold_key)
        new_manifold = old_manifold + change
        new_manifold_key = self.manifold_number_to_key(new_manifold)
        mu_key = new_manifold_key + '_to_' + old_manifold_key
        conjugate_flag = True

        psi_in = psi_dict['psi']
        
        m_nonzero = psi_dict['bool_mask']
        
        boolean_matrix, overlap_matrix = self.dipole_matrix(pulse_number,mu_key,up_flag=False)

        e_mask = self.electric_field_mask(pulse_number,mu_key,conjugate_flag=conjugate_flag)
        boolean_matrix = boolean_matrix * e_mask
        overlap_matrix = overlap_matrix * e_mask

        overlap_matrix, n_nonzero = self.mask_dipole_matrix(boolean_matrix,overlap_matrix,m_nonzero,
                                                                next_manifold_mask = new_manifold_mask)
        t0 = time.time()
        psi = overlap_matrix.dot(psi_in)
                
        t1 = time.time()

        new_dict = {'bool_mask':n_nonzero,'manifold_key':new_manifold_key,'psi':psi}

        return new_dict

    def set_undersample_factor(self,frequency_resolution):
        """dt is set by the pulse. However, the system dynamics may not require such a 
            small dt.  Therefore, this allows the user to set a requested frequency
            resolution for any spectrally resolved signals."""
        # f = pi/dt
        dt = np.pi/frequency_resolution
        u = int(np.floor(dt/self.dt))
        self.undersample_factor = max(u,1)

    def dipole_expectation(self,bra_in,ket_in,*,pulse_number = -1):
        """Computes the expectation value of the two wavefunctions with respect 
            to the dipole operator.  Both wavefunctions are taken to be kets, and the one 
            named 'bra' is converted to a bra by taking the complex conjugate."""
        t0 = time.time()
        pulse_time = self.pulse_times[pulse_number]
        dt = self.dts[pulse_number]
        efield_t = self.efield_times[pulse_number]

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

        manifold1_key = bra_in.manifold_key
        manifold2_key = ket_in.manifold_key

        manifold1_num = self.manifold_key_to_number(manifold1_key)
        manifold2_num = self.manifold_key_to_number(manifold2_key)

        bra_nonzero = bra_in.bool_mask
        ket_nonzero = ket_in.bool_mask

        bra_ev = self.eigenvalues[manifold1_key][bra_nonzero]
        ket_ev = self.eigenvalues[manifold2_key][ket_nonzero]

        e_center = -self.centers[pulse_number]

        bra_in1 = bra_in(t1) * np.exp(-1j*bra_ev[:,np.newaxis]*t1[np.newaxis,:])
        bra_in2 = bra_in(t2) * np.exp(-1j*bra_ev[:,np.newaxis]*t2[np.newaxis,:])
        ket_in1 = ket_in(t1) * np.exp(-1j*(ket_ev[:,np.newaxis]+e_center)*t1[np.newaxis,:])
        ket_in2 = ket_in(t2) * np.exp(-1j*(ket_ev[:,np.newaxis]+e_center)*t2[np.newaxis,:])

        # _u is an abbreviation for undersampled
        t_u = np.hstack((t1,t2))
        bra_u = np.hstack((bra_in1,bra_in2))
        ket_u = np.hstack((ket_in1,ket_in2))
        
        bra_dict = {'bool_mask':bra_nonzero,'manifold_key':manifold1_key,'psi':bra_u}
        ket_dict = {'bool_mask':ket_nonzero,'manifold_key':manifold2_key,'psi':ket_u}

        if np.abs(manifold1_num - manifold2_num) != 1:
            warnings.warn('Dipole only connects manifolds 0 to 1 or 1 to 2')
            return None
        t0 = time.time()
        if manifold1_num > manifold2_num:
            bra_new_mask = ket_dict['bool_mask']
            bra_dict = self.dipole_down(bra_dict,new_manifold_mask = bra_new_mask,
                                        pulse_number = pulse_number)
        else:
            ket_new_mask = bra_dict['bool_mask']
            ket_dict = self.dipole_down(ket_dict,new_manifold_mask = ket_new_mask,
                                        pulse_number = pulse_number)
        
        bra_u = bra_dict['psi']
        ket_u = ket_dict['psi']

        exp_val_u = np.sum(np.conjugate(bra_u) * ket_u,axis=0)
        t1 = time.time()
        self.expectation_time += t1-t0

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

    def integrated_dipole_expectation(self,bra_in,ket_in,*,pulse_number = -1):
        """Computes the expectation value of the two wavefunctions with respect 
            to the dipole operator.  Both wavefunctions are taken to be kets, and the one 
            named 'bra' is converted to a bra by taking the complex conjugate."""
        t0 = time.time()
        pulse_time = self.pulse_times[pulse_number]

        efield_t = self.efield_times[pulse_number]
        
        # The signal is zero before the final pulse arrives, and persists
        # until it decays. Therefore we avoid taking the sum at times
        # where the signal is zero.
        t = efield_t + pulse_time

        manifold1_key = bra_in.manifold_key
        manifold2_key = ket_in.manifold_key

        manifold1_num = self.manifold_key_to_number(manifold1_key)
        manifold2_num = self.manifold_key_to_number(manifold2_key)

        bra_nonzero = bra_in.bool_mask
        ket_nonzero = ket_in.bool_mask

        bra_ev = self.eigenvalues[manifold1_key][bra_nonzero]
        ket_ev = self.eigenvalues[manifold2_key][ket_nonzero]

        e_center = -self.centers[pulse_number]

        bra_in = bra_in(t) * np.exp(-1j*bra_ev[:,np.newaxis]*t[np.newaxis,:])
        ket_in = ket_in(t) * np.exp(-1j*(ket_ev[:,np.newaxis]+e_center)*t[np.newaxis,:])
        
        bra_dict = {'bool_mask':bra_nonzero,'manifold_key':manifold1_key,'psi':bra_in}
        ket_dict = {'bool_mask':ket_nonzero,'manifold_key':manifold2_key,'psi':ket_in}

        if np.abs(manifold1_num - manifold2_num) != 1:
            warnings.warn('Dipole only connects manifolds 0 to 1 or 1 to 2')
            return None
        t0 = time.time()
        if manifold1_num > manifold2_num:
            bra_new_mask = ket_dict['bool_mask']
            bra_dict = self.dipole_down(bra_dict,new_manifold_mask = bra_new_mask,
                                        pulse_number = pulse_number)
        else:
            ket_new_mask = bra_dict['bool_mask']
            ket_dict = self.dipole_down(ket_dict,new_manifold_mask = ket_new_mask,
                                        pulse_number = pulse_number)
        
        bra_in = bra_dict['psi']
        ket_in = ket_dict['psi']

        exp_val = np.sum(np.conjugate(bra_in) * ket_in,axis=0)
        t1 = time.time()
        self.expectation_time += t1-t0
        
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

        center = - self.centers[local_oscillator_number]
        P_of_t = P_of_t# * np.exp(-1j*center*t)
        
        pulse_time_ind = np.argmin(np.abs(self.t))

        efield = self.get_local_oscillator()

        halfway = self.w.size//2
        pm = self.w.size//(2*undersample_factor)
        efield_min_ind = halfway - pm
        efield_max_ind = halfway + pm + self.w.size%2
        efield = efield[efield_min_ind:efield_max_ind]

        P_of_w = fftshift(ifft(ifftshift(P_of_t)))*P_of_t.size*dt

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

        # signal = np.trapz(P * np.conjugate(efield),x=efield_t)
        signal = np.sum(P * np.conjugate(efield))*(efield_t[1] - efield_t[0])
        if not self.return_complex_signal:
            return np.imag(signal)
        else:
            return -1j*signal

    def add_gaussian_linewidth(self,sigma):
        try:
            old_signal = self.old_signal
        except AttributeError:
            self.old_signal = self.signal.copy()
            old_signal = self.old_signal

        if len(self.signal.shape) == 3:

            sig_tau_t = fftshift(fft(ifftshift(old_signal,axes=(-1)),axis=-1),axes=(-1))
            sig_tau_t = sig_tau_t * (np.exp(-self.t**2/(2*sigma**2))[np.newaxis,np.newaxis,:]
                                     *np.exp(-self.t21_array**2/(2*sigma**2))[:,np.newaxis,np.newaxis])
            sig_tau_w = fftshift(ifft(ifftshift(sig_tau_t,axes=(-1)),axis=-1),axes=(-1))
            
            self.signal = sig_tau_w

        elif len(self.signal.shape) == 2:
            sig_t = fftshift(fft(ifftshift(old_signal,axes=(-1)),axis=-1),axes=(-1))
            sig_t = sig_t * (np.exp(-self.t**2/(2*sigma**2))[np.newaxis,:])
            sig_w = fftshift(ifft(ifftshift(sig_t,axes=(-1)),axis=-1),axes=(-1))

            self.signal = sig_w

    def save(self,file_name,pulse_delay_names = [],*,use_base_path=True,makedir=True):
        if use_base_path:
            file_name = os.path.join(self.base_path,file_name)
        if makedir:
            folder = os.path.split(file_name)[0]
            os.makedirs(folder,exist_ok=True)
        if len(pulse_delay_names) == 0:
            pulse_delay_names = ['t' + str(i) for i in range(len(self.all_pulse_delays))]
        save_dict = {}
        for name,delays in zip(pulse_delay_names,self.all_pulse_delays):
            save_dict[name] = delays
        if self.detection_type == 'polarization':
            save_dict['wt'] = self.w
        save_dict['signal'] = self.signal
        save_dict['signal_calculation_time'] = self.calculation_time
        np.savez(file_name,**save_dict)

    def load(self,file_name,pulse_delay_names=[],*,use_base_path=True):
        if use_base_path:
            file_name = os.path.join(self.base_path,file_name)
        arch = np.load(file_name)
        self.all_pulse_delays = []
        if len(pulse_delay_names) == 0:
            for key in arch.keys():
                if key[0] == 't':
                    pulse_delay_names.append(key)
        print(pulse_delay_names)
        for name in pulse_delay_names:
            self.all_pulse_delays.append(arch[name])
        if self.detection_type == 'polarization':
            self.w = arch['wt']
        self.signal = arch['signal']
        self.calculation_time = arch['signal_calculation_time']
            
        
