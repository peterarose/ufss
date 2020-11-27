import numpy as np

# Different electric field shapes that all approach a delta function in time

def gaussian(t,sigma):
    """t is time.  Gaussian pulse, with time-domain standard deviation sigma, 
    normalized to behave like a delta function as sigma -> 0"""
    pre = 1/(np.sqrt(2*np.pi)*sigma)
    return pre * np.exp(-t**2/(2*sigma**2))

def exponential(t,sigma):
    """t is time.  Exponential pulse, with frequency-domain decay constant 
    gamma, normalized to behave like a delta function as sigma -> 0"""
    pre = 1/(2*sigma)
    return pre * np.exp(-np.abs(t)/sigma)

def lorentzian(t,sigma):
    """t is time.  Lorentzian pulse, with time-domain width sigma, 
    normalized to behave like a delta function as sigma -> 0"""
    pre = sigma/2/np.pi
    return pre / (t**2 + (sigma/2)**2)

def sech(t,sigma):
    pre = 1/(np.pi*sigma)
    return pre * 1/np.cosh(t/sigma)

def constant_bandwidth_chirped_gaussian(t,bandwidth,chirp):
    sigma = np.sqrt(1+chirp**2)/bandwidth
    newsig = sigma/np.sqrt(1+1j*chirp)
    return gaussian(t,newsig)

def razorblade_gaussian(t,width,sigma):
    dt = t[1] - t[0]
    a = np.sinc(t*width/np.pi/2)*width/(2*np.pi)
    if sigma == 0:
        return a
    b = gaussian(t,sigma)
    return fftconvolve(a,b,mode='same')*dt
