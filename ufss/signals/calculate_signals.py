#Standard python libraries
import os
import warnings
import copy
import time
import itertools
import functools

#Dependencies - numpy, scipy, matplotlib, pyfftw
import numpy as np
import numpy.polynomial.chebyshev as npch
import matplotlib.pyplot as plt
try:
    from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq
except ImportError:
    from numpy.fft import fft, fftshift, ifft, ifftshift, fftfreq
from scipy.interpolate import interp1d as sinterp1d
import scipy
from scipy.sparse import csr_matrix, issparse

from ufss.perturbative_calculations import UF2OpenEngine, UF2ClosedEngine
from ufss.perturbative_calculations import RKOpenEngine, RKClosedEngine
from ufss.perturbative_calculations import ChebPoly

class CalculateSignals:

    def __init__(self,*,detection_type = 'polarization'):
        
        self.undersample_factor = 1

        self.gamma_res = 6.91
        
        self.rho_to_polarization_time = 0
        self.add_lorentzian_linewidth = False
        if detection_type == 'polarization':
            self.rho_to_signal = self.polarization_detection_rho_to_signal
            self.return_complex_signal = False
            
        elif detection_type == 'complex_polarization':
            self.rho_to_signal = self.polarization_detection_rho_to_signal
            self.return_complex_signal = True
            
        elif detection_type == 'integrated_polarization':
            self.rho_to_signal = self.integrated_polarization_detection_rho_to_signal
            self.return_complex_signal = False
            
        elif detection_type == 'fluorescence':
            self.rho_to_signal = self.fluorescence_detection_rho_to_signal
            self.return_complex_signal = False

    def set_pulse_delays(self,all_delays):
        """Must be a list of numpy arrays, where each array is a
            list of delay times between pulses
"""
        self.all_pulse_delays = all_delays
        num_delays = len(self.all_pulse_delays)
        num_pulses = len(self.efields)
        
        if num_delays == num_pulses - 1:
            pass
        elif (num_delays == num_pulses - 2
              and 'polarization' in self.detection_type):
            # If there is a local oscillator, it arrives simultaneously with
            # the last pulse by default
            self.all_pulse_delays.append(np.array([0]))
        elif num_delays <= num_pulses -2:
            raise Exception('There are not enough delay times')
        elif num_delays >= num_pulses:
            raise Exception('There are too many delay times')

    def get_initial_and_final_signal_shapes(self,all_delay_combinations):
        # for L interacting pulses, there should be L-1 delays
        final_shape = [delays.size for delays in self.all_pulse_delays]
        if self.detection_type == 'polarization' or self.detection_type == 'complex_polarization':
            initial_shape = (len(all_delay_combinations),self.w.size)
            
            if len(final_shape) == self.pdc.shape[0]:
                # get rid of the "delay" between the last pulse and the local oscillator
                final_shape[-1] = self.w.size
            elif len(final_shape) == self.pdc.shape[0] - 1:
                # append the shape of the polariation-detection axis
                final_shape.append(self.w.size)
            else:
                raise Exception('Cannot automatically determine final signal shape')
                
        else:
            initial_shape = (len(all_delay_combinations),)

        return initial_shape, final_shape
        
    def calculate_diagrams_all_delays(self,diagrams):
        t0 = time.time()

        all_delay_combinations = list(itertools.product(*self.all_pulse_delays))
        initial_shape, final_shape = self.get_initial_and_final_signal_shapes(all_delay_combinations)
        signal = np.zeros(initial_shape,dtype='complex')

        counter = 0
        for delays in all_delay_combinations:
            arrival_times = [0]
            for delay in delays:
                arrival_times.append(arrival_times[-1]+delay)

            if self.detection_type == 'polarization' or self.detection_type == 'complex_polarization':
                signal[counter,:] = self.calculate_diagrams(diagrams,arrival_times)
            else:
                signal[counter] = self.calculate_diagrams(diagrams,arrival_times)
            counter += 1

        self.signal = signal.reshape(final_shape)
        if not self.return_complex_signal:
            self.signal = np.real(self.signal)
        self.calculation_time = time.time() - t0
        
        return self.signal

    def set_pulse_times(self,arrival_times):
        if hasattr(self,'pulse_times'):
            old_pulse_times = self.pulse_times
            for i in range(len(old_pulse_times)):
                if old_pulse_times[i] != arrival_times[i]:
                    self.remove_composite_calculations_by_pulse_number(i)
                    self.remove_calculations_by_pulse_number(i)
        else:
            self.reset()
        
        self.pulse_times = arrival_times
        
    def calculate_diagrams(self,diagram_instructions,arrival_times):
        self.set_pulse_times(arrival_times)
        
        self.current_instructions = diagram_instructions
        instructions = diagram_instructions[0]
        signal = self.execute_diagram(instructions)
        rho = self.execute_diagram(instructions)
        signal = self.rho_to_signal(rho)
        for instructions in diagram_instructions[1:]:
            rho = self.execute_diagram(instructions)
            signal += self.rho_to_signal(rho)
        return signal

    def calculate_signal_all_delays(self,*,composite_diagrams=False):
        
        def calculate_signal(t):
            return self.calculate_signal(t,composite_diagrams=composite_diagrams)
        
        t0 = time.time()

        all_delay_combinations = list(itertools.product(*self.all_pulse_delays))

        initial_shape, final_shape = self.get_initial_and_final_signal_shapes(all_delay_combinations)

        signal = np.zeros(initial_shape,dtype='complex')

        if not self.return_complex_signal:
            signal = np.real(signal)

        signal_dict = {key:signal.copy() for key in self.signal_pdcs}

        counter = 0
        for delays in all_delay_combinations:
            arrival_times = [0]
            for delay in delays:
                arrival_times.append(arrival_times[-1]+delay)

            if self.detection_type == 'polarization' or self.detection_type == 'complex_polarization':
                single_signal_dict = calculate_signal(arrival_times)
                for key in self.signal_pdcs:
                    signal_dict[key][counter,:] = single_signal_dict[key]
            else:
                single_signal_dict = calculate_signal(arrival_times)
                for key in self.signal_pdcs:
                    signal_dict[key][counter] = single_signal_dict[key]
            counter += 1

        for key in self.signal_pdcs:
            signal_dict[key] = signal_dict[key].reshape(final_shape)
            
        self.calculation_time = time.time() - t0
        
        self.signal_dict = signal_dict
        if len(self.signal_pdcs) == 1:
            key = self.signal_pdcs[0]
            self.signal = signal_dict[key]
            return self.signal
        else:
            return self.signal_dict

    def set_t_general(self,optical_dephasing_rate,*,dt='auto'):
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
        
    def polarization_detection_rho_to_signal(self,rho):
        t0 = time.time()
        p_of_t = self.dipole_expectation(rho,ket_flag=True)
        self.rho_to_polarization_time += time.time() - t0
        return self.polarization_to_signal(p_of_t,local_oscillator_number=-1)

    def integrated_polarization_detection_rho_to_signal(self,rho):
        t0 = time.time()
        p = self.integrated_dipole_expectation(rho,ket_flag=True)
        self.rho_to_polarization_time += time.time() - t0
        return self.integrated_polarization_to_signal(p,local_oscillator_number=-1)

    def fluorescence_detection_rho_to_signal(self,rho):
        t = np.array([rho.t0 + 1E-2])
        signal = self.fl_exp_val(t,rho)
        return signal
    
    def set_local_oscillator_phase(self,phase):
        self.efields[-1] = np.exp(1j*phase) * self.local_oscillator

    def calculate_signal(self,arrival_times,*,composite_diagrams=False):
        self.set_pulse_times(arrival_times)

        signal_dict = {}
        
        if composite_diagrams:
            self.execute_composite_diagrams()
            for key in self.signal_pdcs:
                signal = self.key_to_signal(key)
                if not self.return_complex_signal:
                    signal = np.real(signal)
                signal_dict[key] = signal

        else:
            t0 = time.time()
            self.set_current_diagram_instructions(arrival_times)
            t1 = time.time()

            for key in self.signal_pdcs:
                diagrams = self.current_instructions[key]
                if len(diagrams) == 0:
                    signal = 0
                else:
                    instructions = diagrams[0]
                    rho = self.execute_diagram(instructions)
                    signal = self.rho_to_signal(rho)
                    for instructions in diagrams[1:]:
                        rho = self.execute_diagram(instructions)
                        signal += self.rho_to_signal(rho)
                    if not self.return_complex_signal:
                        signal = np.real(signal)

                signal_dict[key] = signal

            t2 = time.time()
            self.automation_time += t1-t0
            self.diagram_to_signal_time += t2-t1

        return signal_dict

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

        # if rho_in.pulse_number == None:
        #     old_pulse_time = 0
        # else:
        #     old_pulse_time = self.pulse_times[rho_in.pulse_number]
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

        t0a = time.time()
        t_slice = slice(pulse_start_ind,None,None)

        t = self.t[t_slice] + pulse_time
        
        u = self.undersample_factor
        t1_slice = slice(pulse_start_ind,pulse_end_ind,None)
        t2_slice = slice(pulse_end_ind,None,u)

        t1 = self.t[t1_slice] + pulse_time
        t2 = self.t[t2_slice] + pulse_time

        t_u = np.hstack((t1,t2))
        

        t0b = time.time()
        if t1.size == 0:
            exp_val1 = np.array([])
        else:
            exp_val1 = self.mu_exp_val(t1,rho_in,pulse_number,mu_type='H',
                                       ket_flag = ket_flag, up_flag = False)
        # if self.method == 'chebyshev':
        #     dom = np.array([t2[0],t2[-1]])
        #     halfwidth = (dom[1] - dom[0])/2
        #     midpoint = (dom[1] + dom[0])/2
        #     M = 61
        #     tcheb = npch.chebpts1(M) * halfwidth + midpoint
        #     exp_val_ch=self.mu_exp_val(tcheb,rho_in,pulse_number,mu_type='H',
        #                                ket_flag = ket_flag, up_flag = False)
        #     exp_val_rot = exp_val_ch * np.exp(-1j * center * tcheb)
        #     exp_val_poly = ChebPoly(tcheb,exp_val_rot,dom=dom)
        #     exp_val2 = exp_val_poly(t2)[0,:]
        #     exp_val2 = exp_val2 * np.exp(+1j * center * t2)
        # else:
        exp_val2 = self.mu_exp_val(t2,rho_in,pulse_number,mu_type='H',
                                   ket_flag = ket_flag, up_flag = False)
        
        tb = time.time()
        self.expectation_time += tb-t0b

        exp_val_u = np.hstack((exp_val1,exp_val2))
        
        # rotate away carrier frequency
        exp_val_u  = exp_val_u * np.exp(-1j * center * t_u)

        ta = time.time()
        
        self.slicing_time += ta - t0a - (tb - t0b)

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

        t0 = time.time()
        exp_val = self.mu_exp_val(t,rho_in,pulse_number,mu_type='H',
                                  ket_flag = ket_flag, up_flag = False)

        # rotate away carrier frequency
        exp_val  = exp_val * np.exp(-1j * center * t)
        
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

        if self.method == 'chebyshev':
            dom = self.doms[local_oscillator_number]
            chp = ChebPoly(efield_t,efield[np.newaxis,:],dom = dom)
            full_efield = chp(self.t)[0,:]
            dt = self.t[1] - self.t[0]
            efield_ft = fftshift(ifft(ifftshift(full_efield)))*full_efield.size * dt
            return efield_ft
        
        e_dt = efield_t[1] - efield_t[0]
        dt = self.t[1] - self.t[0]

        if (not np.isclose(e_dt,dt) and e_dt > dt):
            raise Exception("""Local oscillator dt is too large for desired 
                signal detection bandwidth.  You must either use method
                set_t with a larger dt, or supply local oscillator with 
                smaller value of dt""")
        elif (np.isclose(e_dt,dt) and efield_t[-1] >= self.t[-1]):
            full_efield = np.zeros(self.t.size,dtype='complex')

            # the local oscillator sets the "zero" on the clock
            pulse_time_ind = np.argmin(np.abs(self.t))

            pulse_start_ind = pulse_time_ind - efield_t.size//2
            pulse_end_ind = pulse_time_ind + efield_t.size//2 + efield_t.size%2

            t_slice = slice(pulse_start_ind, pulse_end_ind,None)
            
            full_efield[t_slice] = efield
            efield_ft = fftshift(ifft(ifftshift(full_efield)))*full_efield.size * dt
        elif efield_t[-1] > self.t[-1]:
            f = sinterp1d(efield_t,efield,fill_value = (0,0),bounds_error=False,
                          kind='linear')
            full_efield = f(self.t)
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
        P_of_t = P_of_t_in[undersample_slice]#.copy()
        t = self.t[undersample_slice]
        dt = t[1] - t[0]
        if self.add_lorentzian_linewidth:
            P_of_t = P_of_t * np.exp(-self.gamma*np.abs(t))
        pulse_time = self.pulse_times[local_oscillator_number]

        efield = self.get_local_oscillator()

        halfway = self.w.size//2
        pm = self.w.size//(2*undersample_factor)
        efield_min_ind = halfway - pm
        efield_max_ind = halfway + pm + self.w.size%2
        efield = efield[efield_min_ind:efield_max_ind]

        P_of_w = fftshift(ifft(ifftshift(P_of_t)))*P_of_t.size*dt/np.pi

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

        if len(P) == 1:
            signal = P * np.conjugate(efield)

        else:
            signal = np.trapz(P * np.conjugate(efield),x=efield_t)
            
        if not self.return_complex_signal:
            return np.imag(signal)
        else:
            return 1j*signal

    def add_gaussian_linewidth(self,sigma):
        try:
            old_signal_dict = self.old_signal_dict
        except AttributeError:
            self.old_signal_dict = copy.deepcopy(self.signal_dict)
            old_signal_dict = self.old_signal_dict

        for pdc in self.signal_pdcs:
            signal = self.old_signal_dict[pdc]

            if len(signal.shape) == 3:

                sig_tau_t = fftshift(fft(ifftshift(signal,axes=(-1)),axis=-1),axes=(-1))
                sig_tau_t = sig_tau_t * (np.exp(-self.t**2/(2*sigma**2))[np.newaxis,np.newaxis,:]
                                         *np.exp(-self.t21_array**2/(2*sigma**2))[:,np.newaxis,np.newaxis])
                sig_tau_w = fftshift(ifft(ifftshift(sig_tau_t,axes=(-1)),axis=-1),axes=(-1))

                new_signal = sig_tau_w

            elif len(signal.shape) == 2:
                sig_t = fftshift(fft(ifftshift(signal,axes=(-1)),axis=-1),axes=(-1))
                sig_t = sig_t * (np.exp(-self.t**2/(2*sigma**2))[np.newaxis,:])
                sig_w = fftshift(ifft(ifftshift(sig_t,axes=(-1)),axis=-1),axes=(-1))

                new_signal = sig_w

            self.signal_dict[pdc] = new_signal

    def save(self,file_name,pulse_delay_names = [],*,use_base_path=True,makedir=True):
        if use_base_path:
            file_name = os.path.join(self.base_path,file_name)
        if makedir:
            folder = os.path.split(file_name)[0]
            os.makedirs(folder,exist_ok=True)
        if len(pulse_delay_names) == 0:
            pulse_delay_names = ['t' + str(i) for i in range(len(self.all_pulse_delays))]
        try:
            save_dict = {'signal':self.signal}
        except AttributeError:
            save_dict = {}
        if 'signal_dict' in self.__dict__.keys():
            for key in self.signal_dict.keys():
                save_dict[str(key)] = self.signal_dict[key]

        for name,delays in zip(pulse_delay_names,self.all_pulse_delays):
            save_dict[name] = delays
        save_dict['efields'] = self.efields
        save_dict['efield_times'] = self.efield_times
        save_dict['centers'] = self.centers
        save_dict['pdc'] = self.pdc
        save_dict['polarization_sequence'] = self.polarization_sequence
        if self.detection_type == 'polarization' or self.detection_type == 'complex_polarization':
            save_dict['wt'] = self.w
        save_dict['signal_calculation_time'] = self.calculation_time
        np.savez(file_name,**save_dict)

    def load(self,file_name,pulse_delay_names=[],*,use_base_path=True):
        if use_base_path:
            file_name = os.path.join(self.base_path,file_name)
        arch = np.load(file_name)
        print(list(arch.keys()))
        self.all_pulse_delays = []
        if len(pulse_delay_names) == 0:
            for key in arch.keys():
                if key[0] == 't':
                    pulse_delay_names.append(key)
        if len(pulse_delay_names) == 0:
            raise Exception('Pulse delay names not in standard format, must specify manually')
        for name in pulse_delay_names:
            self.all_pulse_delays.append(arch[name])
        if self.detection_type == 'polarization' or self.detection_type == 'complex_polarization':
            try:
                self.w = arch['wt']
                dw = self.w[1] - self.w[0]
                self.t = fftshift(fftfreq(self.w.size,d=dw)*2*np.pi)
            except KeyError:
                print('No detection frequency found')
        self.signal_dict = {}
        self.signal_pdcs = []
        for key in arch.keys():
            if key[0:2] == '((':
                self.signal_dict[eval(key)] = arch[key]
                self.signal_pdcs.append(eval(key))
        self.calculation_time = arch['signal_calculation_time']
        try:
            self.signal = arch['signal']
        except KeyError:
            pass
        try:
            self.efields = arch['efields']
            self.efield_times = arch['efield_times']
            self.centers = arch['centers']
            self.pdc = arch['pdc']
            self.polarization_sequence = arch['polarization_sequence']
        except KeyError:
            pass

class CalculateSignalsOpen(CalculateSignals):

    def set_t(self,optical_dephasing_rate,*,dt='auto'):
        """Sets the time grid upon which all frequency-detected signals will
            be calculated on
"""
        if optical_dephasing_rate == 'auto':
            try:
                eigvals = self.eigenvalues['0,1']
            except KeyError:
                eigvals = self.eigenvalues['all_manifolds']
            optical_dephasing_rate = -np.min(np.real(eigvals))
        self.gamma = optical_dephasing_rate
        self.set_t_general(optical_dephasing_rate,dt=dt)

    def key_to_signal(self,signal_key):
        rho = self.composite_rhos[signal_key]
        return self.rho_to_signal(rho)

class CalculateSignalsClosed(CalculateSignals):

    def set_t(self,optical_dephasing_rate,*,dt='auto',set_gamma=True):
        """Sets the time grid upon which all frequency-detected signals will
            be calculated on
"""
        if set_gamma:
            self.gamma = optical_dephasing_rate
        else:
            self.gamma = 0
        self.set_t_general(optical_dephasing_rate,dt=dt)

    def key_to_signal(self,signal_key):
        calculation_keys = []
        signal_pdc = self.pdc_tup_to_arr(signal_key)
        if self.detection_type == 'integrated_polarization':
            signal = np.zeros(1,dtype='complex')
        else:
            signal = np.zeros(self.w.size,dtype='complex')
        for key in self.composite_psis.keys():
            psi = self.composite_psis[key]
            ket_pdc_a = psi.pdc
            remaining_pdc_a = signal_pdc - ket_pdc_a
            bra_pdc_a = remaining_pdc_a[:,::-1]
            bra_key_a = self.pdc_arr_to_tup(bra_pdc_a)
            if bra_key_a in self.composite_psis.keys():
                calculation_key = (bra_key_a,key)
                if calculation_key in calculation_keys:
                    pass
                else:
                    calculation_keys.append(calculation_key)
                    psi_bra_a = self.composite_psis[bra_key_a]
                    rho = (psi_bra_a,psi)
                    signal += self.rho_to_signal(rho)

            bra_pdc_b = psi.pdc[:,::-1]
            ket_pdc_b = signal_pdc - bra_pdc_b
            ket_key_b = self.pdc_arr_to_tup(ket_pdc_b)
            if ket_key_b in self.composite_psis.keys():
                calculation_key = (key,ket_key_b)
                if calculation_key in calculation_keys:
                    pass
                else:
                    calculation_keys.append(calculation_key)
                    psi_ket_b = self.composite_psis[ket_key_b]
                    rho = (psi,psi_ket_b)
                    signal += self.rho_to_signal(rho)
        return signal

    
class UF2DensityMatrices(UF2OpenEngine,CalculateSignalsOpen):
    def __init__(self,file_path,*,detection_type='polarization',
                 conserve_memory=False):
        UF2OpenEngine.__init__(self,file_path,detection_type=detection_type,
                           conserve_memory=conserve_memory)
        CalculateSignalsOpen.__init__(self,detection_type=detection_type)

class RKDensityMatrices(RKOpenEngine,CalculateSignalsOpen):
    def __init__(self,file_path,*,detection_type='polarization',
                 conserve_memory=False):
        RKOpenEngine.__init__(self,file_path,detection_type=detection_type,
                           conserve_memory=conserve_memory)
        CalculateSignalsOpen.__init__(self,detection_type=detection_type)

class UF2Wavefunctions(UF2ClosedEngine,CalculateSignalsClosed):
    def __init__(self,file_path,*,detection_type='polarization',
                conserve_memory=None):
        UF2ClosedEngine.__init__(self,file_path,detection_type=detection_type)
        CalculateSignalsClosed.__init__(self,detection_type=detection_type)

        self.add_lorentzian_linewidth = True

class RKWavefunctions(RKClosedEngine,CalculateSignalsClosed):
    def __init__(self,file_path,*,detection_type='polarization',
                conserve_memory=None):
        RKClosedEngine.__init__(self,file_path,detection_type=detection_type)
        CalculateSignalsClosed.__init__(self,detection_type=detection_type)

        self.add_lorentzian_linewidth = True


class SpectroscopyBase:
    def __init__(self,file_name,*,engine_name = 'UF2',include_linear=False,
                 detection_type = 'polarization',conserve_memory=False):
        if file_name[-1] == '/':
            file_name = file_name[:-1]
        file_end = os.path.split(file_name)[-1]
        if file_end == 'closed':
            if engine_name == 'UF2':
                engine = UF2Wavefunctions
            elif engine_name == 'RK':
                engine = RKWavefunctions
                
        elif file_end == 'open':
            if engine_name == 'UF2':
                engine = UF2DensityMatrices
            elif engine_name == 'RK':
                engine = RKDensityMatrices
        self.calc = engine(file_name,conserve_memory = conserve_memory,
                           detection_type=detection_type)
        self.engine = self.calc
        self.save_name = 'signals'

    def set_phase_discrimination(self,pdc):
        self.pdc = pdc

    def set_efields(self,efield_times,efields,centers):
        self.engine.set_efields(efield_times,efields,centers,self.pdc)
        self.engine.set_polarization_sequence(['x']*len(self.pdc))

    def set_t(self,gamma,*,dt='auto'):
        self.engine.set_t(gamma,dt=dt)

    def set_identical_gaussians(self,sigma,center,Delta = 10, M = 41):
        self.engine.set_identical_gaussians(sigma,center,self.pdc,
                                          Delta = Delta, M = M)
        self.engine.set_polarization_sequence(['x']*len(self.pdc))
        self.engine.centers = [center]*len(self.pdc)

    def set_impulsive_pulses(self,center):
        self.engine.set_impulsive_pulses(self.pdc)
        self.engine.set_polarization_sequence(['x']*len(self.pdc))
        self.engine.centers = [center]*len(self.pdc)
            
    def set_highest_order(self,highest_order):
        self.engine.include_higher_orders((highest_order-1,1))
    
    def set_pulse_delays(self,pulse_delays):
        self.engine.set_pulse_delays(pulse_delays)

    def calculate_signal_all_delays(self,*,composite_diagrams=True):
        self.engine.calculate_signal_all_delays(composite_diagrams=composite_diagrams)
        self.engine.save(self.save_name)

    def get_signal(self,lam_list,*,max_order = np.inf):
        """lam is the perturbative parameter
        Args:
            lam (float) : unit-less electric field amplitude scaling factor
"""
        random_key = self.engine.signal_pdcs[0]
        signal = np.zeros(self.engine.signal_dict[random_key].shape,dtype='complex')
        for key in self.engine.signal_pdcs:
            pulse_orders = self.engine.pdc_tup_to_arr(key).sum(axis=1)
            if np.sum(pulse_orders) <= max_order:
                lam_powers = [l**p_ord for l,p_ord in zip(lam_list,pulse_orders)]
                perturbative_parameter = np.prod(lam_powers)
                signal += self.engine.signal_dict[key] * perturbative_parameter

        return signal

    def get_signal_order(self,order):
        random_key = self.engine.signal_pdcs[0]
        signal = np.zeros(self.engine.signal_dict[random_key].shape,dtype='complex')
        for key in self.engine.signal_pdcs:
            if self.engine.pdc_tup_to_arr(key).sum() == order:
                signal += self.engine.signal_dict[key]

        return signal
