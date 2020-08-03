#Standard python libraries
import os

#Dependencies
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq
from scipy.optimize import brentq as sbrentq
from scipy.interpolate import interp1d as sinterp1d


class SignalProcessing(object):
    @staticmethod
    def subtract_DC(signal,return_ft = False, axis = 1):
        """Use discrete fourier transform to remove the DC component of a 
            signal.
        Args:
            signal (np.ndarray): real signal to be processed
            return_ft (bool): if True, return the Fourier transform of the 
                              input signal
            axis (int): axis along which the fourier trnasform is to be taken
"""
        sig_fft = fft(ifftshift(signal,axes=(axis)),axis=axis)
        
        nd_slice = [slice(None) for i in range(len(sig_fft.shape))]
        nd_slice[axis] = slice(0,1,1)
        nd_slice = tuple(nd_slice)
        sig_fft[nd_slice] = 0
        
        if not return_ft:
            sig = fftshift(ifft(sig_fft),axes=(axis))
        else:
            sig = sig_fft
        return sig

    @staticmethod
    def remove_DC(x,y,*,axis=0):
        """Redundant with subtract_DC
"""
        k,f = SignalProcessing.ft1D(x,y,axis=axis,zero_DC=True)
        return SignalProcessing.ift1D(k,f,axis=axis,zero_DC=False)

    def integrate_TA(self,x,signal,*,x_range = 'all'):
        if x_range == 'all':
            pass
        else:
            inds = np.where((x > x_range[0]) & (x < x_range[1]))[0]
            x = x[inds]
            signal = signal[:,inds]
        return np.trapz(signal,x=x,axis=1)
    
    @staticmethod
    def ft1D(x,y,*,axis=0,zero_DC=False):
        """Takes in x and y = y(x), and returns k and the Fourier transform 
        of y(x) -> f(k) along a single (1D) axis
        Handles all of the annoyances of fftshift and ifftshift, and gets the 
        normalization right
        Args:
            x (np.ndarray) : independent variable, must be 1D
            y (np.ndarray) : dependent variable, can be nD
        Kwargs:
            axis (int) : which axis to perform FFT
            zero_DC (bool) : if true, sets f(0) = 0
    """
        dx = x[1]-x[0]
        k = fftshift(fftfreq(x.size,d=dx))*2*np.pi
        fft_norm = dx

        shifted_x = ifftshift(x)
        if np.isclose(shifted_x[0],0):
            f = fft(ifftshift(y,axes=(axis)),axis=axis)*fft_norm
        else:
            f = fft(y,axis=axis)*fft_norm

        if zero_DC:
            nd_slice = [slice(None) for i in range(len(f.shape))]
            nd_slice[axis] = slice(0,1,1)
            nd_slice = tuple(nd_slice)
            f[nd_slice] = 0

        f = fftshift(f,axes=(axis))

        return k, f

    @staticmethod
    def ift1D(k,f,*,axis=0,zero_DC=False):
        """Takes in k and f = f(k), and returns x and the discrete Fourier 
        transform of f(k) -> y(x).
        Handles all of the annoyances of fftshift and ifftshift, and gets the 
        normalization right
        Args:
            x (np.ndarray): independent variable
            y (np.ndarray): dependent variable
        Kwargs:
            axis (int) : which axis to perform FFT
    """
        dk = k[1]-k[0]
        x = fftshift(fftfreq(k.size,d=dk))*2*np.pi
        ifft_norm = dk*k.size/(2*np.pi)

        shifted_k = ifftshift(k)
        if np.isclose(shifted_k[0],0):
            y = ifft(ifftshift(f,axes=(axis)),axis=axis)*ifft_norm
        else:
            y = ifft(f,axis=axis)*ifft_norm

        if zero_DC:
            nd_slice = [slice(None) for i in range(len(y.shape))]
            nd_slice[axis] = slice(0,1,1)
            nd_slice = tuple(nd_slice)
            y[nd_slice] = 0

        y = fftshift(y,axes=(axis))

        return x, y

    def phase_2d(self,signal,ind0,ind1):
        val = signal[ind0,ind1]
        phase = np.arctan2(np.real(val),np.imag(val))
        return phase

    def phase_diff_by_inds(self,signal,w1_ind,w2_ind,wT_ind):
        ph1 = self.phase_2d(signal,w1_ind,wT_ind)
        ph2 = self.phase_2d(signal,w2_ind,wT_ind)
        diff = ((ph1-ph2 + np.pi) % (2*np.pi) - np.pi)
        return diff

    def mag_ratio_by_inds(self,signal,w1_ind,w2_ind,wT_ind):
        mag1 = np.abs(signal[w1_ind,wT_ind])
        mag2 = np.abs(signal[w2_ind,wT_ind])
        return mag1/mag2

    def integrated_ft(self,delay_time_start = 1,delay_time_stop = 300):
        delay_time_indices = np.where((self.delay_times > delay_time_start) & (self.delay_times < delay_time_stop))[0]
        delay_times = self.delay_times[delay_time_indices]
        sig = self.signal_vs_delay_times[:,delay_time_indices]
        integrated = np.trapz(sig,x=self.TA.w,axis=0)
        w_T = fftshift(fftfreq(delay_times.size,d=(delay_times[1] - delay_times[0])))*2*np.pi
        integrated_fft = fft(integrated)
        integrated_fft[0] = 0
        integrated_fft = fftshift(integrated_fft)
        return w_T, integrated_ft
        
    def find_zero(self,x,arr):
        """Given an input 1d array, extrapolate a zero crossing
"""
        y = sinterp1d(x,arr)
        try:
            zero = sbrentq(y,-0.5,0.5)
        except:
            zero = np.nan
        return zero
        
