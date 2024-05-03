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
from scipy.sparse import csr_matrix, issparse

from ufss import DiagramGenerator
from ufss.UF2.heaviside_convolve import HeavisideConvolve

def set_identical_efields(obj):
    """This should contain more"""
    obj.efields = []

class rho_container:
    def __init__(self,t,rho,bool_mask,pulse_number,manifold_key,pdc,*,interp_kind='linear',
                 interp_left_fill=0):
        self.bool_mask = bool_mask
        self.pulse_number = pulse_number
        self.manifold_key = manifold_key
        self.pdc = pdc
        self.pdc_tuple = tuple(tuple(pdc[i,:]) for i in range(pdc.shape[0]))
        
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
            if t.size == 0:
                return np.array([])
            elif t[0] > self.t[-1]:
                if t.size <= self.M:
                    ans = self._rho[:,-t.size:]#.copy()
                else:
                    ans = np.ones(t.size,dtype='complex')[np.newaxis,:] * self.asymptote[:,np.newaxis]
            elif t[-1] < self.t[0]:
                if t.size <= self.M:
                    ans = self._rho[:,:t.size]#.copy()
                else:
                    ans = np.zeros((self.n,t.size),dtype='complex')
            elif t.size == self.M:
                if np.allclose(t,self.t):
                    ans = self.rho#.copy()
                else:
                    ans = self.rho_fun(t)
            else:
                ans = self.rho_fun(t)
        else:
                ans = self.rho_fun(t)
        return ans

    def __getitem__(self,inds):
        return self._rho[:,inds]#.copy()

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
                 detection_type = 'polarization',conserve_memory=False):
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
        self.efield_mask_time = 0
        self.dipole_down_dot_product_time = 0
        self.reshape_and_sum_time = 0

        self.next_order_counter = 0

        self.check_for_zero_calculation = False

        self.efield_mask_flag = True

        self.sparsity_threshold = .1

        self.conserve_memory = conserve_memory
        
        self.base_path = file_path

        self.undersample_factor = 1

        self.gamma_res = 6.91

        self.load_eigensystem()

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

        if detection_type == 'polarization':
            self.rho_to_signal = self.polarization_detection_rho_to_signal
            self.return_complex_signal = False
            
        elif detection_type == 'complex_polarization':
            self.rho_to_signal = self.polarization_detection_rho_to_signal
            self.return_complex_signal = True
            detection_type = 'polarization'
            
        elif detection_type == 'integrated_polarization':
            self.rho_to_signal = self.integrated_polarization_detection_rho_to_signal
            self.return_complex_signal = False
            
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
        self.composite_rhos = dict()

    def set_efield_mask_flag(self,efield_mask_flag):
        if efield_mask_flag == False:
            warnings.warn('Warning, turning off efield masking can speed up calculations, but may cause aliasing artifacts. Proceed with caution')
        self.efield_mask_flag = efield_mask_flag

    def set_pulse_delays(self,all_delays):
        """Must be a list of numpy arrays, where each array is a
            list of delay times between pulses
"""
        self.all_pulse_delays = all_delays
        num_delays = len(self.all_pulse_delays)
        num_pulses = len(self.efields)
        
        if num_delays == num_pulses - 1:
            pass
        elif (num_delays == num_pulses - 2 and
                            (self.detection_type == 'polarization' or
                             self.detection_type == 'integrated_polarization')):
            # If there is a local oscillator, it arrives simultaneously with the last pulse by default
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

        # for L interacting pulses, there should be L-1 delays
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
        save_dict = {'UF2_calculation_time':self.calculation_time}
        np.savez(os.path.join(self.base_path,'UF2_calculation_time.npz'),**save_dict)
        
    def calculate_signal_all_delays(self,*,composite_diagrams=False):
        if composite_diagrams:
            calculate_signal = self.calculate_signal_composite_diagrams
        else:
            calculate_signal = self.calculate_signal
        
        t0 = time.time()
        num_delays = len(self.all_pulse_delays)
        num_pulses = len(self.efields)

        all_delay_combinations = list(itertools.product(*self.all_pulse_delays))

        # for L interacting pulses, there should be L-1 delays
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
                signal[counter,:] = calculate_signal(arrival_times)
            else:
                signal[counter] = calculate_signal(arrival_times)
            counter += 1

        self.signal = signal.reshape(signal_shape)
        self.calculation_time = time.time() - t0
        return self.signal

    def set_t(self,optical_dephasing_rate,*,dt='auto'):
        """Sets the time grid upon which all frequency-detected signals will
be calculated on
"""
        if optical_dephasing_rate == 'auto':
            try:
                eigvals = self.eigenvalues['01']
            except KeyError:
                eigvals = self.eigenvalues['all_manifolds']
            optical_dephasing_rate = -np.min(np.real(eigvals))
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

    def add_rhos(self,ra,rb):
        """Add two density matrix objects together
"""
        tmin = min(ra.t[0],rb.t[0])
        tmax = max(ra.t[-1],rb.t[-1])
        dt = min(ra.t[1]-ra.t[0],rb.t[1]-rb.t[0])
        t = np.arange(tmin,tmax+dt*0.9,dt)
        rho = np.zeros((ra.bool_mask.size,t.size),dtype='complex')
        rho[ra.bool_mask,:] = ra(t)
        rho[rb.bool_mask,:] += rb(t)

        bool_mask = np.logical_or(ra.bool_mask,rb.bool_mask)
        rho = rho[bool_mask,:]

        if ra.pulse_number == rb.pulse_number:
            pulse_number = ra.pulse_number
        else:
            pulse_number = None

        if ra.manifold_key != rb.manifold_key:
            warnings.warn('Inconsistent manifolds')
        manifold_key = ra.manifold_key
        if not np.allclose(ra.pdc,rb.pdc):
            raise Exception('Cannot add density matrices with different phase-discrimination conditions')
        pdc = ra.pdc
        
        rab = rho_container(t,rho,bool_mask,pulse_number,manifold_key,pdc)
        return rab
        
    def execute_composite_diagrams(self):
        self.check_for_zero_calculation = True
        self.set_ordered_interactions()
        interactions = set(self.ordered_interactions)
        rhos = self.composite_rhos
        old_rhos = [self.rho0]
            
        for i in range(len(self.ordered_interactions)):
            new_rhos = []
            pdcs = set()
            for pulse_num, pm_flag in interactions:
                for old_rho in old_rhos:
                    output_pdc = self.get_output_pdc(old_rho.pdc,pulse_num,pm_flag)

                    # exclude cases with too many interactions with the same pulse
                    if np.all(self.pdc - output_pdc >= 0):

                        # Check to see if pulses overlap or are correctly ordered
                        remaining_pdc = self.pdc - output_pdc
                        remaining_pulse_interactions = np.sum(remaining_pdc,axis=1)
                        remaining_pulses = np.zeros(remaining_pulse_interactions.shape,dtype='bool')
                        remaining_pulses[:] = remaining_pulse_interactions

                        output_t = self.efield_times[pulse_num] + self.pulse_times[pulse_num]
                        output_t0 = output_t[0]
                        add_flag = True
                        for i in range(remaining_pulses.size):
                            if remaining_pulses[i]:
                                test_t = self.efield_times[i] + self.pulse_times[i]
                                if test_t[-1] < output_t0:
                                    add_flag = False
                        
                        if add_flag:
                            pdcs.add(tuple(tuple(output_pdc[i,:]) for i in range(output_pdc.shape[0])))
                    
                    ket_key,bra_key = self.wavevector_dict[pm_flag]
                    new_rho_k = self.KB_dict[ket_key](old_rho,pulse_number=pulse_num)
                    new_rho_b = self.KB_dict[bra_key](old_rho,pulse_number=pulse_num)
                    if new_rho_k != None:
                        new_rhos.append(new_rho_k)
                    if new_rho_b != None:
                        new_rhos.append(new_rho_b)
            
            for new_rho in new_rhos:
                if new_rho.pdc_tuple in rhos.keys():
                    partial_rho = rhos[new_rho.pdc_tuple]
                    new_rho = self.add_rhos(partial_rho,new_rho)
                rhos[new_rho.pdc_tuple] = new_rho

            old_rhos = [rhos[pdc] for pdc in pdcs]

        self.check_for_zero_calculation = False
        
        rho = rhos[self.pdc_tuple]
        # print(rhos.keys())
        sig = self.rho_to_signal(rho)
        return sig

    def calculate_signal_composite_diagrams(self,arrival_times):
        try:
            old_pulse_times = self.pulse_times
            for i in range(len(old_pulse_times)):
                if old_pulse_times[i] != arrival_times[i]:
                    try:
                        self.remove_composite_rhos_by_pulse_number(i)
                    except IndexError:
                        pass
        except AttributeError:
            pass
        
        self.pulse_times = arrival_times
        t0 = time.time()
        
        signal = self.execute_composite_diagrams()
        
        return signal

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

    def remove_composite_rhos_by_pulse_number(self,pulse_number):
        keys = self.composite_rhos.keys()
        keys_to_remove = []
        for key in keys:
            ket_ints,bra_ints = key[pulse_number]
            if ket_ints > 0 or bra_ints > 0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.composite_rhos.pop(key)

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

    def set_current_diagram_instructions(self,arrival_times):
        self.diagram_generation_counter += 1
        t0a = time.time()
        self.current_instructions = self.get_diagrams(arrival_times)
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

        self.set_current_diagram_instructions(arrival_times)

        t1 = time.time()
        
        if len(self.current_instructions) == 0:
            signal = 0
        else:
            instructions = self.current_instructions[0]
            signal = self.execute_diagram(instructions)
            for instructions in self.current_instructions[1:]:
                signal += self.execute_diagram(instructions)

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
        p_of_t = self.dipole_expectation(rho,ket_flag=True)
        return self.polarization_to_signal(p_of_t,local_oscillator_number=-1)

    def integrated_polarization_detection_rho_to_signal(self,rho):    
        p = self.integrated_dipole_expectation(rho,ket_flag=True)
        return self.integrated_polarization_to_signal(p,local_oscillator_number=-1)

    # def fluorescence_detection_rho_to_signal(self,rho):
    #     n_nonzero = rho['bool_mask']
    #     L_size = self.eigenvalues[0].size
    #     H_size = int(np.sqrt(L_size))

    #     # move back to the basis the Liouvillian was written in
    #     rho = self.eigenvectors['right'][:,n_nonzero].dot(rho['rho'][:,-1])

    #     # reshape rho into a normal density matrix representation
    #     rho = rho.reshape((H_size,H_size))

    #     fluorescence_yield = np.array([0,1,1,self.f_yield])

    #     signal = np.dot(np.diagonal(rho),fluorescence_yield)
        
    #     return signal

    def set_efields(self,times_list,efields_list,centers_list,phase_discrimination,*,reset_rhos = True,
                    plot_fields = False):
        self.efield_times = times_list
        self.efields = efields_list

        self.efield_masks = [dict() for efield in self.efields]
        
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

    def get_rho(self,t,key,*,reshape=True,original_L_basis=True):
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
        if original_L_basis:
            new_rho = ev.dot(rho)
        else:
            new_rho = rho.copy()
        if reshape:
            new_rho = new_rho.reshape(size_H,size_H,rho.shape[-1])
        return new_rho

    def get_rho_by_order(self,t,order,original_L_basis=True):
        keys = self.rhos.keys()
        order_keys = []
        counter = 0
        for key in keys:
            if len(key) == 3*order:
                order_keys.append(key)
        rho_total = self.get_rho(t,order_keys.pop(0),original_L_basis=original_L_basis)
        for key in order_keys:
            counter += 1
            rho_total += self.get_rho(t,key,original_L_basis=original_L_basis)
        # print(counter)

        return rho_total

    def set_rho0(self):
        """Creates the unperturbed density matarix by finding the 0 
            eigenvalue of the ground-state manifold, which should correspond
            to a thermal distribution
"""
        try:
            ev = self.eigenvalues['all_manifolds']
        except KeyError:
            ev = self.eigenvalues['00']

        t = self.efield_times[0] # can be anything of the correct length
        
        initial_state = np.where(np.isclose(ev,0,atol=1E-12))[0]
        rho0 = np.ones((1,t.size),dtype=complex)
        bool_mask = np.zeros(ev.size,dtype='bool')
        bool_mask[initial_state] = True

        if bool_mask.sum() != 1:
            warnings.warn('Could not automatically determine the initial thermal state. User must specify the initial condition, rho^(0), manually')
            return None

        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)

        self.rho0 = rho_container(t,rho0,bool_mask,None,'00',pdc,
                                  interp_kind='zero',interp_left_fill=1)

    def set_rho0_manual_L_eigenbasis(self,manifold_key,bool_mask,weights):
        """
"""
        ev = self.eigenvalues[manifold_key][bool_mask]

        t = self.efield_times[0] # can be anything of the correct length
        
        rho0 = np.ones((bool_mask.sum(),t.size),dtype=complex) * weights[:,np.newaxis]

        if manifold_key == 'all_manifolds':
            manifold_key = '00'

        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)

        self.rho0 = rho_container(t,rho0,bool_mask,None,manifold_key,pdc,
                                  interp_kind='zero',interp_left_fill=1)

    def set_rho0_manual(self,rho0,*,manifold_key = 'all_manifolds'):
        """Set the initial condition.  Must be done after setting the pulse shapes
        Args:
            rho0 (2D np.array) : the initial density matrix, in the basis that
                the Liouvillian was defined in
            manfifold_key (str) : manifold in which initial density matrix is 
                defined (usually 'all_manifolds' (default) or '00')
"""
        try:
            evl = self.left_eigenvectors[manifold_key]
        except AttributeError:
            self.load_left_eigenvectors()
            evl = self.left_eigenvectors[manifold_key]

        rho0_flat = rho0.flatten() # ufss always works with vectors
        

        rho0_eig = evl.dot(rho0_flat) # transform into Liouvillian eigenbasis
        
        nonzero_inds = np.where(np.abs(rho0_eig) > 1E-12)[0]

        bool_mask = np.zeros(rho0_eig.size,dtype='bool')
        bool_mask[nonzero_inds] = True

        rho0_trimmed = rho0_eig[nonzero_inds]

        t = self.efield_times[0]
        time_dependence = np.ones(t.size)
        rho0 = rho0_trimmed[:,np.newaxis] * time_dependence[np.newaxis,:]
        
        if manifold_key == 'all_manifolds':
            manifold_key = '00'

        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)

        self.rho0 = rho_container(t,rho0,bool_mask,None,manifold_key,pdc,
                                  interp_kind='zero',interp_left_fill=1)

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
        eigvec_save_name = os.path.join(self.base_path,'right_eigenvectors.npz')
        with np.load(eigval_save_name) as eigval_archive:
            self.manifolds = list(eigval_archive.keys())
            self.eigenvalues = {key:eigval_archive[key] for key in self.manifolds}
        with np.load(eigvec_save_name,allow_pickle=True) as eigvec_archive:
            self.eigenvectors = dict()
            for key in self.manifolds:
                ev = eigvec_archive[key]
                if ev.dtype == np.dtype('O'):
                    self.eigenvectors[key] = ev[()]
                elif self.check_sparsity(ev):
                    self.eigenvectors[key] = csr_matrix(ev)
                else:
                    self.eigenvectors[key] = ev

        if self.conserve_memory:
            self.load_left_eigenvectors()

    def load_left_eigenvectors(self):
        left_eigvec_save_name = os.path.join(self.base_path,'left_eigenvectors.npz')
        with np.load(left_eigvec_save_name,allow_pickle=True) as left_eigvec_archive:
            self.left_eigenvectors = dict()
            for key in self.manifolds:
                evl = left_eigvec_archive[key]
                if evl.dtype == np.dtype('O'):
                    self.left_eigenvectors[key] = evl[()]
                elif self.check_sparsity(evl):
                    self.left_eigenvectors[key] = csr_matrix(evl)
                else:
                    self.left_eigenvectors[key] = evl
                        
    def set_rho_shapes(self):
        self.rho_shapes = dict()
        if 'all_manifolds' in self.manifolds:
            L_size = self.eigenvalues['all_manifolds'].size
            H_size = int(np.sqrt(L_size))
            self.rho_shapes['all_manifolds'] = (H_size,H_size)
        else:
            H_sizes = dict()
            for key in self.manifolds:
                ket_key, bra_key = key
                if ket_key == bra_key:
                    L_size = self.eigenvalues[key].size
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
            indices: (new manifold eigenfunction, old manifold eigenfunction, 
            cartesian coordinate)."""
        file_name = os.path.join(self.base_path,'mu.npz')
        file_name_pruned = os.path.join(self.base_path,'mu_pruned.npz')
        file_name_bool = os.path.join(self.base_path,'mu_boolean.npz')
        try:
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

    def check_sparsity(self,mat):
        csr_mat = csr_matrix(mat)
        sparsity = csr_mat.nnz / (csr_mat.shape[0]*csr_mat.shape[1])
        if sparsity < self.sparsity_threshold:
            return True
        else:
            return False
        
    ### Setting the electric field to be used

    def set_polarization_sequence(self,polarization_list,*,reset_rhos=True):
        """Sets the sequences used for either parallel or crossed pump and probe
        
        Args:
            polarization_list (list): list of strings, can be 'x','y', or 
                'z' for linear polarizations or 'r' and 'l' for right and 
                left circularly polarized light, respectively
        Returns:
            None: sets the attribute polarization sequence
"""

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        r = np.array([1,-1j,0])/np.sqrt(2)
        l = np.conjugate(r)
        pol_options = {'x':x,'y':y,'z':z,'r':r,'l':l}

        self.polarization_sequence = [pol_options[pol] for pol in polarization_list]
        if reset_rhos:
            self.rhos = dict()


    ### Tools for recursively calculating perturbed wavepackets using TDPT

    def dipole_matrix(self,pulse_number,key,ket_flag=True,up_flag=True):
        """Calculates the dipole matrix given the electric field polarization vector,
            if ket_flag = False then uses the bra-interaction"""
        t0 = time.time()
        pol = self.polarization_sequence[pulse_number]

        if up_flag == ket_flag:
            # rotating term
            pass
        else:
            # counter-rotating term
            pol = np.conjugate(pol)
            
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
            overlap_matrix = mu[0]#.copy()
        elif np.all(pol == y):
            overlap_matrix = mu[1]#.copy()
        elif np.all(pol == z):
            overlap_matrix = mu[2]#.copy()
        else:
            overlap_matrix = mu[0]*pol[0] + mu[1]*pol[1] + mu[2]*pol[2]

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
        ta = time.time()

        try:
            mask = self.efield_masks[pulse_number][key]
        except KeyError:
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
            
            self.efield_masks[pulse_number][key] = mask

            tb = time.time()
            self.efield_mask_time += tb - ta

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
            the ket manifold, the second, the bra manifold.  If the density 
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

    def get_output_pdc(self,input_pdc,pulse_number,pm_flag):
        output_pdc = input_pdc.copy()
        if pm_flag == '+':
            output_pdc[pulse_number,0] += 1
        elif pm_flag == '-':
            output_pdc[pulse_number,1] += 1
        else:
            raise Exception('Cannot parse pm_flag')

        return output_pdc
    
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
        if ket_flag == up_flag:
            # Rotating term excites the ket and de-excites the bra
            conjugate_flag = False
        else:
            # Counter-rotating term
            conjugate_flag = True
            
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

        # # my system breaks above 9. Need to fix this...
        # if new_manifold[0] > 9:
        #     new_manifold[0] = 9
        #     warnings.warn('manifold_key tracking system breaks down after 9 excitations')
        # if new_manifold[1] > 9:
        #     new_manifold[1] = 9
        #     warnings.warn('manifold_key tracking system breaks down after 9 excitations')

        input_pdc = rho_in.pdc
        output_pdc = input_pdc.copy()
        if conjugate_flag:
            output_pdc[pulse_number][1] += 1
        else:
            output_pdc[pulse_number][0] += 1

        if self.check_for_zero_calculation:
            output_pdc_tuple = tuple(tuple(output_pdc[i,:]) for i in range(output_pdc.shape[0]))
            # print('Testing',output_pdc_tuple)
            if output_pdc_tuple in self.composite_rhos.keys():
                # do not redo unnecesary calculations
                # print("Don't redo calculations",output_pdc_tuple)
                return None
            if not np.all(self.pdc - output_pdc >= 0):
                # too many interactions with one of the pulses
                return None
            if np.any(new_manifold < self.minimum_manifold):
                return None
            elif np.any(new_manifold > self.maximum_manifold):
                return None

            remaining_pdc = self.pdc - output_pdc
            remaining_pulse_interactions = np.sum(remaining_pdc,axis=1)
            remaining_pulses = np.zeros(remaining_pulse_interactions.shape,dtype='bool')
            remaining_pulses[:] = remaining_pulse_interactions

            output_t0 = t[0]
            for i in range(remaining_pulses.size):
                if remaining_pulses[i]:
                    test_t = self.efield_times[i] + self.pulse_times[i]
                    if test_t[-1] < output_t0:
                        # print('Excluded due to pulse non-overlap',output_pdc)
                        return None
        
        new_manifold_key = self.manifold_array_to_key(new_manifold)
        mu_key = old_manifold_key + '_to_' + new_manifold_key

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

        if self.conserve_memory:
            # move back to the basis the Liouvillian was written in
            if 'all_manifolds' in self.manifolds:
                rho = self.eigenvectors['all_manifolds'][:,m_nonzero].dot(rho)
                ket_size,bra_size = self.rho_shapes['all_manifolds']
            else:
                rho = self.eigenvectors[old_manifold_key][:,m_nonzero].dot(rho)
                ket_size,bra_size = self.rho_shapes[old_manifold_key]
                
            t_size = t.size

            rho = rho.reshape(ket_size,bra_size,t_size)

            if ket_flag:
                old_ket_key = old_manifold_key[0]
                new_ket_key = new_manifold_key[0]
                if up_flag:
                    H_mu_key = old_ket_key + '_to_' + new_ket_key
                else:
                    H_mu_key = new_ket_key + '_to_' + old_ket_key

                rotating_flag = up_flag
            else:
                old_bra_key = old_manifold_key[1]
                new_bra_key = new_manifold_key[1]
                if up_flag:
                    H_mu_key = old_bra_key + '_to_' + new_bra_key
                else:
                    H_mu_key = new_bra_key + '_to_' + old_bra_key
                rotating_flag = not up_flag

            overlap_matrix = self.get_H_mu(pulse_number,H_mu_key,
                                           rotating_flag=rotating_flag)
            
            t0 = time.time()
            if ket_flag:
                rho = np.einsum('ij,jkl',overlap_matrix,rho)
            else:
                rho = np.einsum('ijl,jk',rho,overlap_matrix)
            t1 = time.time()
            
            rho_vec_size = rho.shape[0]*rho.shape[1]
            rho = rho.reshape(rho_vec_size,t_size)
            if 'all_manifolds' in self.manifolds:
                evl = self.left_eigenvectors['all_manifolds']
            else:
                evl = self.left_eigenvectors[new_manifold_key]
            rho = evl.dot(rho)
            n_nonzero = np.ones(rho.shape[0],dtype='bool')
            zero_inds = np.where(np.isclose(rho[:,-1],0,atol=1E-12))
            n_nonzero[zero_inds] = False
            rho = rho[n_nonzero,:]
        else:
            boolean_matrix, overlap_matrix = self.dipole_matrix(pulse_number,mu_key,ket_flag=ket_flag,up_flag=up_flag)

            if self.efield_mask_flag:
                e_mask = self.electric_field_mask(pulse_number,mu_key,conjugate_flag=conjugate_flag)
                if np.allclose(e_mask,True):
                    pass
                else:
                    boolean_matrix = boolean_matrix * e_mask
                    if issparse(overlap_matrix):
                        overlap_matrix = overlap_matrix.toarray()

                    overlap_matrix = csr_matrix(overlap_matrix * e_mask)
            else:
                pass

            overlap_matrix, n_nonzero = self.mask_dipole_matrix(boolean_matrix,overlap_matrix,m_nonzero,
                                                                next_manifold_mask = new_manifold_mask)


            t0 = time.time()
            rho = overlap_matrix.dot(rho)
            t1 = time.time()
            
        self.next_order_expectation_time += t1-t0

        exp_factor2 = np.exp(-ev2[n_nonzero,np.newaxis]*t[np.newaxis,:])

        rho = rho * exp_factor2

        t0 = time.time()

        M = self.efield_times[pulse_number].size

        fft_convolve_fun = self.heaviside_convolve_list[pulse_number].fft_convolve2
        # fft_convolve_fun = self.fft_convolve2

        if M == 1:
            rho = rho * self.efields[pulse_number]
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
            rho = rho * 1j
        else:
            rho = rho * -1j

        rho_out = rho_container(t,rho,n_nonzero,pulse_number,
                                new_manifold_key,output_pdc)

        self.next_order_counter += 1

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

    def load_H_mu(self):
        parent_dir = os.path.split(self.base_path)[0]
        file_name = os.path.join(parent_dir,'closed','mu.npz')

        with np.load(file_name) as mu_archive:
            self.H_mu = {key:mu_archive[key] for key in mu_archive.keys()}

    def get_H_mu(self,pulse_number,key,rotating_flag=True):
        """Calculates the dipole matrix given the electric field polarization vector"""
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
            overlap_matrix = mu[:,:,0]#.copy()
        elif np.all(pol == y):
            overlap_matrix = mu[:,:,1]#.copy()
        elif np.all(pol == z):
            overlap_matrix = mu[:,:,2]#.copy()
        else:
            overlap_matrix = np.tensordot(mu,pol,axes=(-1,0))

        if not rotating_flag:
            overlap_matrix = np.conjugate(overlap_matrix.T)

        t1 = time.time()
        self.dipole_time += t1-t0

        return overlap_matrix

    def dipole_down_H_mu(self,rho_dict,*,new_manifold_mask = None,
                     pulse_number = -1,ket_flag=True):
        """This method is similar to the method down, but does not involve 
            the electric field shape or convolutions. It is the action of the 
            dipole operator on the ket-side without TDPT effects.  It also includes
            the dot product of the final electric field polarization vector."""
        if not ket_flag:
            raise Exception('Not implemented for bra-side')
        old_manifold_key = rho_dict['manifold_key']
        old_ket_key = old_manifold_key[0]
        new_ket_key = str(int(old_ket_key)-1)
        mu_key = new_ket_key + '_to_' + old_ket_key

        if ket_flag:
            center = - self.centers[pulse_number]
            conjugate_flag = True
        else:
            center = self.centers[pulse_number]
            conjugate_flag = False

        rho_in = rho_dict['rho']
        t_size = rho_in.shape[-1]
        
        m_nonzero = rho_dict['bool_mask']

        # move back to the basis the Liouvillian was written in
        try:
            rho = self.eigenvectors['all_manifolds'][:,m_nonzero].dot(rho_in)
            L_size = rho.shape[0]
            H_size = int(np.sqrt(L_size))
            rho = rho.reshape(H_size,H_size,t_size)
        except KeyError:
            rho = self.eigenvectors[old_manifold_key][:,m_nonzero].dot(rho_in)
            ket_manifold_key = old_manifold_key[0] + old_manifold_key[0]
            ket_L_manifold_size = self.eigenvalues[ket_manifold_key].size
            ket_H_size = int(np.sqrt(ket_L_manifold_size))

            bra_manifold_key = old_manifold_key[1] + old_manifold_key[1]
            bra_L_manifold_size = self.eigenvalues[bra_manifold_key].size
            bra_H_size = int(np.sqrt(bra_L_manifold_size))

            rho = rho.reshape(ket_H_size,bra_H_size,t_size)
        
        overlap_matrix = self.get_H_mu(pulse_number,mu_key,rotating_flag=False)

        # e_mask = self.electric_field_mask(pulse_number,mu_key,conjugate_flag=conjugate_flag)
        # boolean_matrix = boolean_matrix * e_mask
        # overlap_matrix = overlap_matrix * e_mask

        # overlap_matrix, n_nonzero = self.mask_dipole_matrix(boolean_matrix,overlap_matrix,m_nonzero,
        #                                                         next_manifold_mask = new_manifold_mask)
            
        t0 = time.time()

        if issparse(overlap_matrix):
            overlap_matrix = overlap_matrix.toarray()
        
        polarization_field = np.einsum('ij,jik',overlap_matrix,rho)
                
        t1 = time.time()

        self.dipole_down_dot_product_time += t1 - t0

        return polarization_field

    def dipole_down_L_mu(self,rho_dict,*,new_manifold_mask = None,pulse_number = -1,
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

        if self.efield_mask_flag:
            e_mask = self.electric_field_mask(pulse_number,mu_key,conjugate_flag=conjugate_flag)
            if np.allclose(e_mask,True):
                pass
            else:
                boolean_matrix = boolean_matrix * e_mask
                if issparse(overlap_matrix):
                    overlap_matrix = overlap_matrix.toarray()

                overlap_matrix = csr_matrix(overlap_matrix * e_mask)
        else:
            pass

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

        rho_nonzero = rho_in.bool_mask
        try:
            ev = self.eigenvalues['all_manifolds'][rho_nonzero]
        except KeyError:
            ev = self.eigenvalues[rho_in.manifold_key][rho_nonzero]

        rho1 = rho1 * np.exp((ev[:,np.newaxis] - 1j*center)* t1[np.newaxis,:])

        rho2 = rho2 * np.exp((ev[:,np.newaxis] - 1j*center) * t2[np.newaxis,:])
        tb = time.time()

        # _u is an abbreviation for undersampled
        t_u = np.hstack((t1,t2))
        rho_u = np.hstack((rho1,rho2))
        
        
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

        # Initialize return array with zeros
        ret_val = np.zeros(self.t.size,dtype='complex')
        
        # set non-zero values using t_slice
        ret_val[pulse_start_ind:] = exp_val

        tb = time.time()
        self.interpolation_time += tb-t0
        
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

        elif e_dt > dt*1.00001:
            raise Exception("""Local oscillator dt is too large for desired 
                signal detection bandwidth.  You must either use method
                set_t with a larger dt, or supply local oscillator with 
                smaller value of dt""")
        
        # elif efield_t[-1] < self.t[-1]:
        #     f = sinterp1d(efield_t,efield,fill_value = (0,0),bounds_error=False,
        #                   kind='linear')
        #     full_efield = f(self.t)
        #     efield_ft = fftshift(ifft(ifftshift(full_efield)))*full_efield.size * dt
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
        P_of_t = P_of_t_in[undersample_slice]#.copy()
        t = self.t[undersample_slice]
        dt = t[1] - t[0]
        pulse_time = self.pulse_times[local_oscillator_number]

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

        signal = np.trapz(P * np.conjugate(efield),x=efield_t)
        if not self.return_complex_signal:
            return np.imag(signal)
        else:
            return -1j*signal

    def add_gaussian_linewidth(self,sigma):
        self.old_signal = self.signal.copy()

        sig_tau_t = fftshift(fft(ifftshift(self.old_signal,axes=(-1)),axis=-1),axes=(-1))
        sig_tau_t = sig_tau_t * (np.exp(-self.t**2/(2*sigma**2))[np.newaxis,np.newaxis,:]
                                 *np.exp(-self.t21_array**2/(2*sigma**2))[:,np.newaxis,np.newaxis])
        sig_tau_w = fftshift(ifft(ifftshift(sig_tau_t,axes=(-1)),axis=-1),axes=(-1))
        self.signal = sig_tau_w

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
