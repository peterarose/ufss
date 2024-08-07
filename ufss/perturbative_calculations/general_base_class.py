import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as npch
from pyfftw.interfaces.numpy_fft import fftshift, ifft, ifftshift, fftfreq
import copy
import warnings
import os

from ufss.perturbative_calculations.heaviside_convolve import HeavisideConvolve, HeavisideConvolveSP

class BaseClass:
    """Contains methods that are used by all perturbative calculation engines
        (both UF2 and direct propagation).
"""
    def __init__(self,file_path,*,detection_type = 'polarization'):
        self.detection_type = detection_type
        self.base_path = file_path
        self.method = 'UF2'

        # Code will not actually function until the following empty lists are set by the user
        self.efields = [] #initialize empty list of electric field shapes
        self.efield_times = [] #initialize empty list of times assoicated with each electric field shape
        self.dts = [] #initialize empty list of time spacings associated with each electric field shape
        self.polarization_sequence = [] #initialize empty polarization sequence
        self.pulse_times = [] #initialize empty list of pulse arrival times
        self.centers = [] #initialize empty list of pulse center frequencies
        self.efield_wavevectors = []
        self.heaviside_convolve_list = []

    def set_efields(self,times_list,efields_list,centers_list,
                    phase_discrimination,*,reset_calculations = True,
                    plot_fields = False):
        """Use to set electric field pulse shapes to be used in calculations. 
            Electric fields can include the carrier frequency, in which case the
            centers in centers_list should be 0, or can be specified in the 
            rotating frame, in which case center frequencies should be specified
            in the centers_list.
            
        Args:
            times_list (list) : list of 1D np.ndarrays specifying the time points
                on which each pulse is defined.
            efields_list (list) : list of 1D np.ndarrays specifying the complex 
                amplitude of each pulse
            centers_list (list) : list of floats specifying the center frequency
                of each pulse
            phase_discrimination (tuple) : tuple of tuples, ((n0r,n0c),(n1r,n1c),...)
                that specify the number of interactions with the rotating and
                counter-rotating terms of each pulse
                
        Keyword Args:
            reset_calculations (bool) : True if all stored perturbative calculations
                should be deleted (setting this to False is dangerous, as old 
                calculations done with previous pulse-shapes will be reused without
                warning)
            plot_fields (bool) : True if you wish to see plots of the pulses in both
                time- and frequency-domain. This will also initiate a check of the
                time- and frequency-resolution of the pulses
            """
        self.efield_times = times_list
        self.efields = efields_list
        self.efield_masks = [dict() for efield in self.efields]
        self.centers = centers_list
        self.set_phase_discrimination(phase_discrimination)
        self.dts = []
        self.efield_frequencies = []
        self.heaviside_convolve_list = []
        if reset_calculations:
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
            if self.interaction_picture_calculations:
                self.heaviside_convolve_list.append(HeavisideConvolve(t.size))
            else:
                self.heaviside_convolve_list.append(HeavisideConvolveSP(t.size))
        self.dt = self.dts[0]
        self.set_dft_efields()

        if self.detection_type == 'polarization' or 'integrated_polarization':
            try:
                self.local_oscillator = self.efields[-1].copy()
            except:
                self.local_oscillator = copy.deepcopy(self.efields[-1])

        for i in range(len(self.efields)):
            self.check_efield_resolution(i,plot_fields = plot_fields)

    def set_dft_efields(self):
        """Calculates the dft of the efields and saves the results as a list 
            to the attribute "efields_fts". Also calculates the associated 
            angular frequencies, and stores the results as a list to the
            attribute "efield_omegas" """
        self.efield_omegas = []
        self.efield_fts = []
        for pulse_number in range(len(self.efields)):
            dt = self.dts[pulse_number]
            efield_t = self.efield_times[pulse_number]
            efield_w = fftshift(fftfreq(efield_t.size,d=dt))*2*np.pi
            self.efield_omegas.append(efield_w)
            ifft_norm = dt*efield_t.size
            efield = self.efields[pulse_number]
            efield_ft = fftshift(ifft(ifftshift(efield)))*ifft_norm
            self.efield_fts.append(efield_ft)

    def check_efield_resolution(self,pulse_number,*,plot_fields = False):
        """Checks the resolution in time- and frequency-domain of the specified
            pulse, and also optionally plots the fields for view

        Args:
            pulse_number (int) : which pulse to check

        Keyword Args:
            plot_fields (bool) : if True, plots the specified pulse in time- and
                frequency-domains
        """
        efield = self.efields[pulse_number]
        if len(efield) == 1:
            return None
        
        efield_tail = np.max(np.abs([efield[0],efield[-1]]))

        if efield_tail > np.max(np.abs(efield))/100:
            warnings.warn('''Consider using larger time interval, pulse does not decay
                              to less than 1% of maximum value in time domain''')
        efield_ft = self.efield_fts[pulse_number]
        efield_ft_tail = np.max(np.abs([efield_ft[0],efield_ft[-1]]))
        
        if efield_ft_tail > np.max(np.abs(efield_ft))/100:
            warnings.warn('''Consider using smaller value of dt, pulse does not decay 
                              to less than 1% of maximum value in frequency domain''')

        if plot_fields:
            self.plot_efield(pulse_number)

    def plot_efield(self,pulse_number):
        """Plots the specified pulse in both time and frequency domain

        Args:
            pulse_number (int) : which pulse to plot
        """
        efield_t = self.efield_times[pulse_number]
        efield = self.efields[pulse_number]
        efield_w = self.efield_omegas[pulse_number]
        efield_ft = self.efield_fts[pulse_number]
        fig, axes = plt.subplots(1,2)
        l1,l2, = axes[0].plot(efield_t,np.real(efield),efield_t,np.imag(efield))
        plt.legend([l1,l2],['Real','Imag'])
        axes[1].plot(efield_w,np.real(efield_ft),efield_w,np.imag(efield_ft))

        axes[0].set_ylabel('Electric field Amp')
        axes[0].set_xlabel('Time ($\omega_0^{-1})$')
        axes[1].set_xlabel('Frequency ($\omega_0$)')

        fig.suptitle('Check that efield is well-resolved in time and frequency')
        plt.show()

    def set_polarization_sequence(self,polarization_list,*,
                                  reset_calculations=True):
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
        if reset_calculations:
            self.reset()

    def set_impulsive_pulses(self,phase_discrimination):
        """Automatically sets L impulsive pulses, where L is the length of the 
            input phase discrimination

        Args:
            phase_discrimination (tuple) : tuple of tuples, ((n0r,n0c),(n1r,n1c),...)
                that specify the number of interactions with the rotating and
                counter-rotating terms of each pulse
        """
        L = len(phase_discrimination) # number of pulses
        # Delta = 10 and M = 41 hard-coded in
        efield_t = np.array([0])
        times = [efield_t] * L
        self.set_polarization_sequence(['x'] * L)
        centers = [0] * L
        ef = np.array([1])
        efields = [ef] * L

        self.set_efields(times,efields,centers,phase_discrimination,
                         reset_calculations = True,plot_fields = False)
        

    def set_identical_gaussians(self,sigma_t,c,phase_discrimination,*,
                                Delta = 10, M = 41):
        """Automatically sets L identical gaussian pulses, where L is the length 
            of the input phase discrimination

        Args:
            sigma_t (float) : standard deviation of the Gaussian shape in time-domain
            c (float) : center frequency of the pulse
            phase_discrimination (tuple) : tuple of tuples, ((n0r,n0c),(n1r,n1c),...)
                that specify the number of interactions with the rotating and
                counter-rotating terms of each pulse
        Keyword Args:
            Delta (float) : full time duration for the pulses, in multiples of sigma_t
            M (int) : number of time points to use for the pulses
        """
        L = len(phase_discrimination) # number of pulses
        if self.method == 'chebyshev':
            efield_t = npch.chebpts1(M) * Delta/2 * sigma_t
            dom = np.array([-Delta/2*sigma_t,Delta/2*sigma_t])
            self.doms = [dom] * L
        else:
            efield_t = np.linspace(-Delta/2,Delta/2,num=M)*sigma_t
            
        ef = np.exp(-efield_t**2/(2*sigma_t**2))/(np.sqrt(2*np.pi)*sigma_t)
        times = [efield_t] * L
        self.set_polarization_sequence(['x'] * L)
        centers = [c] * L
        efields = [ef] * L

        self.set_efields(times,efields,centers,phase_discrimination,
                         reset_calculations = True,plot_fields = False)

    def get_closest_index_and_value(self,value,array):
        """Given an array and a desired value, finds the closest actual value
            stored in that array, and returns that value, along with its corresponding 
            array index
        
        Args:
            value (complex) : value to look for in the given array
            array (np.ndarray) : 1D array to look in
"""
        index = np.argmin(np.abs(array - value))
        value = array[index]
        return index, value
    
class UF2BaseClass(BaseClass):
    def save_timing(self):
        save_dict = {'UF2_calculation_time':self.calculation_time}
        np.savez(os.path.join(self.base_path,'UF2_calculation_time.npz'),**save_dict)

class DPBaseClass(BaseClass):
    def save_timing(self):
        save_dict = {'DP_calculation_time':self.calculation_time}
        np.savez(os.path.join(self.base_path,'DP_calculation_time.npz'),**save_dict)
