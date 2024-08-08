#Dependencies - numpy
import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift, fftfreq


class HeavisideConvolve:
    """This class calculates the discrete convolution of an array with the 
        heaviside step function

    Attributes:
        size (int) : number of linear convolution points
        theta_fft (numpy.ndarray) : discrete fourier transform of the step 
            function
        a : array of zeros for use with the fft algorithm
        
"""
    def __init__(self,arr_size):
        """
        Args:
            arr_size (int) : number of points desired for the linear 
                convolution
"""
        self.size = arr_size
        self.theta_fft = self.heaviside_fft()

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
        # The discrete convolution is inherently circular. Therefore we
        # perform the convolution using 2N-1 points
        a = np.zeros(2*self.size - 1, dtype='complex128')
        a[:arr.size] = arr

        b = fft(a) * self.theta_fft

        c = ifft(b)
        # Return only the results of the linear convolution
        return c[-arr.size:] * d

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
        # The discrete convolution is inherently circular. Therefore we
        # perform the convolution using 2N-1 points
        a = np.zeros((size0,2*self.size - 1), dtype='complex128')
        a[:,:size1] = arr
        b = fft(a,axis=-1) * self.theta_fft[np.newaxis,:]
        c = ifft(b,axis=-1)
        # Return only the results of the linear convolution
        ans = c[:,-size1:] * d
        return ans

class HeavisideConvolveSP:
    """This class calculates the discrete convolution of an array with the 
        heaviside step function times an exponential decay/oscillation 
        envelope

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

        t = np.arange(-self.size+1,self.size)
        self.theta = np.heaviside(t,0.5)

    def fft_convolve(self,arr1,arr2,*,d=1):
        """This method calculates the linear convolution of an input with 
            the heaviside step function
        
        Args:
            arr (numpy.ndarray): 1d array of input function values f(x)
            d (float): spacing size of grid f(x) is evaluated on, dx

        Returns:
            numpy.ndarray: linear convolution of arr with heaviside step 
                function
"""
        # The discrete convolution is inherently circular. Therefore we
        # perform the convolution using 2N-1 points
        a = np.zeros(2*self.size - 1, dtype='complex128')
        a[:arr2.size] = arr2

        a_theta = self.theta.copy()
        a_theta[-arr1.size:] *= arr1
        
        b = fft(a) * fft(a_theta)

        c = ifft(b)

        # Return only the results of the linear convolution
        return c[-arr2.size:] * d

    def fft_convolve2(self,arr1,arr2,*,d=1):
        """This method loops over fft_convolve in order to perform the convolution of input array with the heaviside step function along the second axis of arr

        Args:
            arr (numpy.ndarray): 2d array of input function values f_i(x), 
                where i is the 1st index of the array
            d (float): spacing size of grid f_i(x) is evaluated on, dx

        Returns:
            numpy.ndarray: 2d array of linear convolution of arr with 
                heaviside step function along the second axis of arr
"""
        size0,size1 = arr2.shape
        # The discrete convolution is inherently circular. Therefore we
        # perform the convolution using 2N-1 points
        a = np.zeros((size0,2*self.size - 1), dtype='complex128')
        a[:,:size1] = arr2

        a_theta = np.zeros((size0,2*self.size - 1), dtype='complex128')
        a_theta[:,-arr1.size] = arr1
        a_theta = self.theta[np.newaxis,:] * a_theta
        
        b = fft(a,axis=-1) * fft(a_theta,axis=-1)

        c = ifft(b,axis=-1)

        # Return only the results of the linear convolution
        return c[:,-size1:] * d
