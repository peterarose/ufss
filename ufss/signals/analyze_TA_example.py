#Standard python libraries
import copy
import os
import time

#Dependencies
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq

#UF2
from .plotting_tools import Plotter
from .signal_processing import SignalProcessing

class AnalyzeTransientAbsorption(SignalProcessing,Plotter):

    def __init__(self,parameter_file_path,*,load_file_name='default'):
        self.base_path = parameter_file_path
        self.load(load_file_name=load_file_name)

    def load(self,*,load_file_name='default'):
        if load_file_name == 'default':
            load_name = os.path.join(self.base_path,'TA_spectra_iso_ave.npz')
        else:
            load_name = os.path.join(self.base_path,load_file_name)
        arch = np.load(load_name)
        self.signal = arch['signal']
        if 'delay_times' in arch.keys():
            self.delay_times = arch['delay_times']
        elif 't0' in arch.keys():
            self.delay_times = arch['t0']
        elif 'T' in arch.keys():
            self.delay_times = arch['T']
        else:
            raise Exception('Could not load delay times from signal archive')
        try:
            self.w = arch['frequencies']
        except KeyError:
            self.w = arch['wt']
        try:
            self.center = arch['pulse_center']
        except KeyError:
            warnings.warn('Pulse center was not saved in archive, setting center = 0')
            self.center = 0

    def get_closest_index_and_value(self,value,array):
        """Given an array and a desired value, finds the closest actual value
            stored in that array, and returns that value, along with its 
            corresponding array index
"""
        index = np.argmin(np.abs(array - value))
        value = array[index]
        return index, value

    def phase_diff(self,signal,w1,w2,wT):
        w1ind, w1 = self.get_closest_index_and_value(w1,self.w)
        w2ind, w2 = self.get_closest_index_and_value(w2,self.w)
        wTind, wT = self.get_closest_index_and_value(wT,self.wT)
        signal_ft = self.ft_axis1(signal)
        return self.phase_diff_by_inds(signal_ft,w1ind,w2ind,wTind)

    def mag_ratio(self,signal,w1,w2,wT):
        w1ind, w1 = self.get_closest_index_and_value(w1,self.w)
        w2ind, w2 = self.get_closest_index_and_value(w2,self.w)
        wTind, wT = self.get_closest_index_and_value(wT,self.wT)
        signal_ft = self.ft_axis1(signal)
        return self.mag_ratio_by_inds(signal_ft,w1ind,w2ind,wTind)

    def integrated_signal(self,*,frequency_range='all'):
        return self.integrate_TA(self.w, self.signal,
                                 x_range = frequency_range)

    def integrated_ft(self,*,frequency_range='all',delay_time_start = -1,delay_time_stop = 300,gamma_T = 0):
        delay_time_indices = np.where((self.delay_times > delay_time_start)
                                      & (self.delay_times < delay_time_stop))[0]
        delay_times = self.delay_times[delay_time_indices]
        signal = self.signal[delay_time_indices,:].copy()
        if gamma_T != 0:
            signal *= np.exp(-gamma_T*np.abs(delay_times))[:,np.newaxis]
        w_T, signal = self.ft1D(delay_times,signal,axis=0)
        return w_T,self.integrate_TA(self.w, signal,
                                 x_range = frequency_range)

    def find_node(self):
        """Finds the location of the vibrational node
"""
        zeros = np.zeros(self.signal.shape[0])
        T, sig = self.remove_DC(self.delay_times,self.signal,axis=0)
        for i in range(zeros.size):
            zero = self.find_zero_closest_to(self.w,np.real(sig[i,:]))
            zeros[i] = zero
        return zeros

    def find_average_node(self,*,max_delay_time = 50):
        nodes = self.find_node()
        Tind, Tmax = self.get_closest_index_and_value(max_delay_time,self.delay_times)
        nodes_to_fit = nodes[:Tind]
        finite_indices = np.where(np.isfinite(nodes_to_fit))[0]
        return np.mean(nodes_to_fit[finite_indices])

    def plot_node(self):
        zeros = self.find_node()
        plt.plot(self.delay_times,zeros,'--r')
        
    def find_node_wT(self,wT):
        w_T, sig = self.ft1D(self.delay_times,self.signal,axis=0)
        wTind, wT = self.get_closest_index_and_value(wT,w_T)
        imag_zero = self.find_zero_closest_to(self.w,np.imag(sig[wTind,:]))
        real_zero = self.find_zero_closest_to(self.w,np.real(sig[wTind,:]))
        return real_zero,imag_zero

    def kappa_measure(self,wT,delay_time_start = 1,gamma_T = 0,node_position=0):
        ave_node = self.find_average_node()
        m = self.kappa_measure_manual_node(wT,delay_time_start = delay_time_start,
                                           gamma_T = gamma_T,node_position=ave_node)
        return m

    def kappa_measure_manual_node(self,wT,delay_time_start = 1,gamma_T = 0,node_position=0):
        w_T, neg_integral = self.integrated_ft(frequency_range=[-np.inf,node_position+0.0001],delay_time_start=delay_time_start,
                                               gamma_T=gamma_T)
        w_T, pos_integral = self.integrated_ft(frequency_range=[node_position-0.0001,np.inf],delay_time_start=delay_time_start,
                                               gamma_T = gamma_T)
        wTind, wT = self.get_closest_index_and_value(wT,w_T)
        neg = np.abs(neg_integral[wTind])
        pos = np.abs(pos_integral[wTind])
        ave = (neg + pos)/2
        diff = (pos - neg)/2
        return diff/ave

    def kappa_measure_old(self,wT,delay_time_start = 1,gamma_T = 0):
        w_T, neg_integral = self.integrated_ft(frequency_range=[-np.inf,0.0001],delay_time_start=delay_time_start,
                                               gamma_T=gamma_T)
        w_T, pos_integral = self.integrated_ft(frequency_range=[-0.0001,np.inf],delay_time_start=delay_time_start,
                                               gamma_T = gamma_T)
        wTind, wT = self.get_closest_index_and_value(wT,w_T)
        neg = neg_integral[wTind]
        pos = pos_integral[wTind]
        if np.abs(pos) > np.abs(neg):
            k = +1
        else:
            k = -1
        ave = np.abs(neg + pos)/2
        diff = np.abs(neg - pos)/2
        return k*diff/ave

    def add_gaussian_linewidth(self,sigma):
        self.old_signal = self.signal.copy()

        t,sig_t = self.ft1D(self.w,self.old_signal,axis=-1)
        sig_t = sig_t * np.exp(-t**2/(2*sigma**2))[np.newaxis,:]
        w,sig_w = self.ift1D(t,sig_t,axis=-1)
        self.signal = sig_w

    def plot(self,*,remove_DC = True,frequency_range=[-3,3]):
        self.plotTA(subtract_DC = remove_DC, create_figure=True,
                    color_range = 'auto',draw_colorbar = True,save_fig=False,
                    frequency_range=frequency_range)
        # if remove_DC:
        #     T, sig = self.remove_DC(self.delay_times,self.signal,axis=0)
        # else:
        #     sig = self.signal
        # ufss.signals.plot2D(self.delay_times,sig,part='real')
