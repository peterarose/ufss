import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

import ufss

ift = ufss.signals.SignalProcessing.ift1D
ft = ufss.signals.SignalProcessing.ft1D

folder = 'v_3LS'

# "v" 3LS from Smallwood 2017 paper

# the first step is to create a directory to work in
folder = 'v_3LS'
os.makedirs(folder,exist_ok=True)

# next we construct L0 based upon Smallwood et al.'s 2017 paper
# we are considering their v 3 level system (see their Figure 3)
# since this consists of 3 energy levels, we construct three occupation operators:
#Ground state:
n0 = np.zeros((3,3))
n0[0,0] = 1
#first excited state:
n1 = np.zeros((3,3))
n1[1,1] = 1
#second excited state:
n2 = np.zeros((3,3))
n2[2,2] = 1

# the Hamiltonian is given as
c = 0 #optical gap, which is rotated away
H = n0*0 + n1*(c-0.5) + n2*(c+0.5)
II = np.eye(H.shape[0]) #Identity of same dimension as H

#####

L_closed = -1j*np.kron(H,II.T) + 1j*np.kron(II,H.T)

#####

#Smallwood's paper uses the following definition of the pulse shape
def g(t,sigma):
    """t is time.  Gaussian pulse, with time-domain standard deviation sigma, normalized to behave like
    a delta function as sigma -> 0"""
    pre = 1/(np.sqrt(2*np.pi)*sigma)
    return pre * np.exp(-t**2/(2*sigma**2))

def G(w,sigma):
    """w is angular frequency.  Analytical Fourier transform of g(t,sigma), where sigma is the time-domain 
    standard deviation sigma"""
    return np.exp(-w**2*sigma**2/2)

#Where they define the frequency bandwidth beta as
def beta(sigma):
    """Bandwidth of the pulse (for a Gaussian, this is the frequency-domain standard deviation)"""
    return 1/sigma

#####

#They work with sigma = 1, and define the bath coupling relative to the bandwidth of the pulse:
sigma = 1
gamma_dephasing = 0.2*beta(sigma)
gamma_decay = 0.1*beta(sigma)

# Now let's add the bath coupling as described in the paper.  This is a 9x9 matrix with mostly zeros.
R = np.zeros(L_closed.shape)
# Because they are not keeping track of where lost population goes, R will be entirely diagonal
# They use two rates, a dephasing rate, and a population decay rate
R[1,1] = gamma_dephasing
R[2,2] = gamma_dephasing
R[3,3] = gamma_dephasing
R[4,4] = gamma_decay
R[5,5] = gamma_decay
R[6,6] = gamma_dephasing
R[7,7] = gamma_decay
R[8,8] = gamma_decay

#####

#The total Liouvillian is therefore
L0 = L_closed - R

#####

mu_ag = np.zeros((3,3))
mu_ag[1,0] = 1
mu_ga = mu_ag.T

mu_bg = np.zeros((3,3))
mu_bg[2,0] = 1
mu_gb = mu_bg.T
# The total dipole operator
mu = mu_ag + mu_ga + mu_bg + mu_gb
# In order to enforce the rotating wave approximation (RWA) we only use mu_up and mu_down = mu_up.T
mu_ket_up = mu_ag + mu_bg
print(mu_ket_up)

#####

manL = ufss.HLG.ManualL(L0,mu_ket_up,savedir=folder,output='uf2')

##### Get the same w-values that will be used by uf2

re = ufss.DensityMatrices(os.path.join(folder,'open'),
                          detection_type='complex_polarization')

M = 25 # number of points used to resolve optical pulses
Delta = 6 # pulse interval
t = np.linspace(-Delta/2,Delta/2,num=M)
dt = t[1] - t[0]
ef = g(t,sigma)

# Smallwood et al. use a delta function for the local oscillator
lo_dt = 0.25 #### This must never change, because it defines t and tau, by default
lo_t =  np.arange(-5,5.2,lo_dt)    
lo_dt = lo_t[1] - lo_t[0]
lo = np.zeros(lo_t.size,dtype='float')
lo[lo.size//2] = 1/lo_dt

re.set_efields([t,t,t,lo_t],[ef,ef,ef,lo],[0,0,0,0],[(0,1),(1,0),(1,0)])
    
re.gamma_res = 20
re.set_t(gamma_dephasing)

#############################
# Now make Smallwood's result
#############################

def Pij(w,Wij):
    return 1j/(w-Wij)

from scipy.special import erf

def Da(wt,T,wtau,eta,Wjk,Whi,Wfg,mu,sigma):
    wt = wt[:,np.newaxis,np.newaxis]
    T = T[np.newaxis,:,np.newaxis]
    wtau = wtau[np.newaxis,np.newaxis,:]
    pre = 1j*mu[2] * mu[1] * mu[0] / (2*np.pi) #Note we are using a different normalization factor
    wt_terms = Pij(wt,Wjk) * G(wt-eta[2]*c-Whi,sigma)
    wtau_terms = Pij(wtau,Wfg) * G(wtau-eta[0]*c,sigma) * G(wtau+eta[1]*c-Whi,sigma)
    T_terms = np.exp(-1j*Whi*T)
    mixed = 0.5*(1+erf((T+1j*sigma**2*(wt+wtau-eta[2]*c+eta[1]*c-2*Whi))/(2*sigma)))
    return pre*wt_terms*wtau_terms*T_terms*mixed

# We will be using
eta = [-1,1,1]
# which refers to the phase-matching direction -k1 + k2 + k3
# We will be using
mu = [1,1,1]
# because all dipole transitions have equal strength in this model

# L0 is diagonal, so the eigenvalues are trivially found to be
eigvals = L0.diagonal()/-1j

W = eigvals.reshape((3,3))

wt = re.w
wtau = wt.copy()
T = np.arange(0,1,1)
c = 0
analytical_sig = -1j * (Da(wt+c,T,wtau-c,eta,W[1,0],W[1,1],W[0,1],mu,sigma)
                        + Da(wt+c,T,wtau-c,eta,W[1,0],W[1,2],W[0,2],mu,sigma)
                        + Da(wt+c,T,wtau-c,eta,W[2,0],W[2,2],W[0,2],mu,sigma)
                        + Da(wt+c,T,wtau-c,eta,W[2,0],W[2,1],W[0,1],mu,sigma)
                        + Da(wt+c,T,wtau-c,eta,W[2,0],W[0,0],W[0,1],mu,sigma)
                        + Da(wt+c,T,wtau-c,eta,W[2,0],W[0,0],W[0,2],mu,sigma)
                        + Da(wt+c,T,wtau-c,eta,W[1,0],W[0,0],W[0,2],mu,sigma)
                        + Da(wt+c,T,wtau-c,eta,W[1,0],W[0,0],W[0,1],mu,sigma) )
save_file_name = os.path.join('v_3LS','analytical_signal.npy')
np.save(save_file_name,analytical_sig.T)
