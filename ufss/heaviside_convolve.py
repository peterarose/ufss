#Standard python libraries
import os
import warnings
import copy
import time
import itertools

#Dependencies - numpy, scipy, matplotlib, pyfftw
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq
import scipy
from scipy.interpolate import interp1d as sinterp1d


class HeavisideConvolve:
    """This class calculates the discrete convolution of an array with the 
        heaviside step function

    Attributes:
        size (int) : number of linear convolution points
        theta_fft (numpy.ndarray) : discrete fourier transform of the step 
            function
        a : aligned array of zeros for use with the fftw algorithm
        b : empty aligned array for use with the fftw algorithm
        c : empty aligned array for use with the fftw algorithm
        fft : method for calculating the FFT of a (stores the result in b)
        ifft : method for calculating the IFFT of b (stores the result in c)
        
"""
    def __init__(self,arr_size):
        """
        Args:
            arr_size (int) : number of points desired for the linear 
                convolution
"""
        self.size = arr_size
        self.theta_fft = self.heaviside_fft()
        # The discrete convolution is inherently circular. Therefore we
        # perform the convolution using 2N-1 points
        self.a = pyfftw.empty_aligned(2*self.size - 1, dtype='complex128', n=16)
        self.b = pyfftw.empty_aligned(2*self.size - 1, dtype='complex128', n=16)
        self.c = pyfftw.empty_aligned(2*self.size - 1, dtype='complex128', n=16)
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(10)
        self.fft = pyfftw.FFTW(self.a, self.b)
        self.ifft = pyfftw.FFTW(self.b,self.c,direction='FFTW_BACKWARD')
        self.a[:] = 0

    def heaviside_fft(self,*,value_at_zero=0.5):
        """This method calculates the FFT of the heaviside step function
        
        Args:
            value_at_zero (float): value of the heaviside step function at 
                x = 0

        Returns:
            numpy.ndarray: the FFT of the heaviside step function
"""
        # The discrete convolution is inherently circular. Therefore we perform the
        # convolution using 2N-1 points. Spacing dx is irrelevant for evaluating
        # the heaviside step function. However it is crucial that the x = 0 is included
        t = np.arange(-self.size+1,self.size)
        y = np.heaviside(t,value_at_zero)
        return fft(y)

    def fft_convolve(self,arr,*,d=1):
        """This method calculates the linear convolution of an input with 
            the heaviside step function
        
        Args:
            arr (numpy.ndarray): 1d array of input function values f(x)
            d (float): spacing size of grid f(x) is evaluated on, dx

        Returns:
            numpy.ndarray: linear convolution of arr with heaviside step 
                function
"""
        self.a[:arr.size] = arr

        self.b = self.fft()
        
        self.b *= self.theta_fft

        self.b = self.b

        self.c = self.ifft()
        # Return only the results of the linear convolution
        return self.c[-arr.size:] * d

    def fft_convolve2(self,arr,*,d=1):
        """This method loops over fft_convolve in order to perform the convolution of input array with the heaviside step function along the second axis of arr

        Args:
            arr (numpy.ndarray): 2d array of input function values f_i(x), 
                where i is the 1st index of the array
            d (float): spacing size of grid f_i(x) is evaluated on, dx

        Returns:
            numpy.ndarray: 2d array of linear convolution of arr with 
                heaviside step function along the second axis of arr
"""
        size0,size1 = arr.shape
        for i in range(size0):
            arr[i,:] = self.fft_convolve(arr[i,:],d=d)
        return arr
