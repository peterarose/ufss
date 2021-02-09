import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

#Import UFSS
import ufss

# these are helper functions defined in ufss for convenient forward and backward dft
# they use pyfftw
ift = ufss.signals.SignalProcessing.ift1D
ft = ufss.signals.SignalProcessing.ft1D

folder = 'circular_test'
os.makedirs(folder,exist_ok=True)

# since this system consists of 3 energy levels, we construct three occupation operators:
# Ground state:
n0 = np.zeros((3,3))
n0[0,0] = 1
# first excited state:
n1 = np.zeros((3,3))
n1[1,1] = 1
#second excited state:
n2 = np.zeros((3,3))
n2[2,2] = 1

# create reference signal
e = np.array([1])

for i in range(len(e)):
    print(f'{i}/{len(e)}')
    print(e[i])
    
    # the Hamiltonian is given as
    epsilon = e[i] # energy gap between ground and lowest excited state (rotated away)
    depsilon = 0 # energy gap between lowest excited state and next excited state
    H = 0*n0 + epsilon * n1 + (epsilon + depsilon) * n2
    II = np.eye(H.shape[0]) #Identity of same dimension as H


    # Take H and split it into the ground-state manifold (GSM) and the singly-excited manifold (SEM)
    H_GSM = H[:1,:1]
    H_SEM = H[1:,1:]

    # Normally we would now diagonalize H_GSM anad H_SEM separately.  However, they are already diagonal!

    # eigenvalues are the diagonal entries of H
    eigenvalues = {'0':H_GSM.diagonal(),'1':H_SEM.diagonal()}

    # matrix of eigenvectors is simply the identity
    eigenvectors = {'0':np.eye(eigenvalues['0'].size),'1':np.eye(eigenvalues['1'].size)}


    # right and left-circularly polarized light are described as
    r = np.array([1,-1j,0])/np.sqrt(2)
    l = np.conjugate(r)

    # we want r to excite from 0 -> 1:
    mu_01 = np.array([1,1j,0])/np.sqrt(2)

    # we want l to excite from 0 -> 2:
    mu_02 = np.array([1,-1j,0])/np.sqrt(2)

    mu = np.zeros((3,3,3),dtype='complex')
    mu[1,0,:] = mu_01
    mu[2,0,:] = mu_02

    # Now in principle we should do the following to force mu to be Hermitian
    mu_H = mu + np.conjugate(np.transpose(mu,axes=(1,0,2)))
    

    # However, in UFSS, we usually enforce the rotating-wave approximation (RWA) by using what is defined here as
    # "mu", and not "mu_H".  mu excites the ket wavefunction, and the complex conjugate transpose of mu de-excites
    # the ket wavefunction.  Taking the conjugate transpose is handled automatically in UF2 (may not be implemented
    # correctly in RKE yet)

    np.einsum('ijk,k',mu,r)

    # now a quick test. Begin with initial wavefunction in ground state
    psi0 = np.zeros(3)
    psi0[0] = 1
    mu_dot_e_r = np.einsum('ijk,k',mu,r)
    mu_dot_e_l = np.einsum('ijk,k',mu,l)
    

    psi1 = mu_dot_e_r.dot(psi0)
    psi2 = mu_dot_e_l.dot(psi0)



    # Looks like this has worked!  Now we actually only need the part of mu that connects the GSM to the SEM:
    mu_dict = {'0_to_1':mu[1:,:1,:]}

    # in this case we can represent psi0 as a vector of length one:
    psi0 = np.array([1])

    mu_dot_e_r = np.einsum('ijk,k',mu_dict['0_to_1'],r)
    mu_dot_e_l = np.einsum('ijk,k',mu_dict['0_to_1'],l)

    psi1 = mu_dot_e_r.dot(psi0)
    psi2 = mu_dot_e_l.dot(psi0)


    np.savez(os.path.join(folder,'eigenvalues'),**eigenvalues)
    np.savez(os.path.join(folder,'eigenvectors'),**eigenvectors)
    np.savez(os.path.join(folder,'mu'),**mu_dict)


    #re is short for rephasing 2DPE
    re = ufss.Wavepackets(folder,detection_type='complex_polarization')


    # defining the optical pulses in the RWA
    # sigma = 1
    # M = 21 # number of points used to resolve optical pulses
    # Delta = 10 # pulse interval
    # t = np.linspace(-Delta/2,Delta/2,num=M)*sigma
    # dt = t[1] - t[0]
    # ef = ufss.gaussian(t,sigma)
    # c = 1 # center frequency of the laser pulse

    # # Smallwood et al. use a delta function for the local oscillator
    # lo_dt = 1 #### This must never change, because it defines t and tau, by default
    # lo_t =  np.arange(-5,5.2,lo_dt)    
    # lo_dt = lo_t[1] - lo_t[0]
    # lo = np.zeros(lo_t.size,dtype='float')
    # lo[lo.size//2] = 1/lo_dt

    # defining the impulsive pulses in the RWA
    c = 1 # center frequency ofthe laser pulse

    pulse1 = np.array([1]) # pulse 1 amplitude is 1
    t1 = np.array([0]) # impulsive limit, nonzero at a single time point

    pulse2 = np.array([1]) # pulse 2 amplitude is 1
    t2 = np.array([0]) # impulsive limit, nonzero at a single time point

    lo = np.array([1]) # local oscillator amplitude is 1
    lo_t = np.array([0]) # impulsive limit, nonzero at a single time point

    pmc = [(0,1),(2,0)] #pmc stands for phase-matching condition. must include 1 tuple for each pulse
    # excluding the local oscillator

    re.maximum_manifold = 1
    alpha = 0
    re.set_polarization_sequence(['r', 'r', 'r', 'r'])

    re.set_efields([t1,t2,lo_t],[pulse1,pulse2,lo],[c,c,c],pmc)

    re.gamma_res = 20
    re.set_t(0.2, dt = 1) #set tmax for polarization (only thing here that I think is opaque)
    re.pulse_times = [0, 0, 0]

    tau = re.t.copy()[re.t > -15] #dtau is the same as the dt for local oscillator
    T = np.arange(0, 1, 1)
    re.set_pulse_delays([tau, T])


    full_sig_tau_T_wt = re.calculate_signal_all_delays()
    # the signal returned is the signal as a function of tau, T and the fourier conjugate of t -> omega_t
    # inverse fourier transform to get as a function of t:

    t, full_sig_tau_T_t = ft(re.w,full_sig_tau_T_wt,axis = 1)
    true_signal = full_sig_tau_T_t * np.exp(-(tau[:,np.newaxis] - t[np.newaxis,:])**2/(2*1/0.2**2))

e = np.random.normal(1, 0.2, 1000)
signal = []

for i in range(len(e)):
    print(f'{i}/{len(e)}')
    print(e[i])
    
    # the Hamiltonian is given as
    epsilon = e[i] # energy gap between ground and lowest excited state (rotated away)
    depsilon = 0 # energy gap between lowest excited state and next excited state
    H = 0*n0 + epsilon * n1 + (epsilon + depsilon) * n2
    II = np.eye(H.shape[0]) #Identity of same dimension as H


    # Take H and split it into the ground-state manifold (GSM) and the singly-excited manifold (SEM)
    H_GSM = H[:1,:1]
    H_SEM = H[1:,1:]

    # Normally we would now diagonalize H_GSM anad H_SEM separately.  However, they are already diagonal!

    # eigenvalues are the diagonal entries of H
    eigenvalues = {'0':H_GSM.diagonal(),'1':H_SEM.diagonal()}

    # matrix of eigenvectors is simply the identity
    eigenvectors = {'0':np.eye(eigenvalues['0'].size),'1':np.eye(eigenvalues['1'].size)}


    # right and left-circularly polarized light are described as
    r = np.array([1,-1j,0])/np.sqrt(2)
    l = np.conjugate(r)

    # we want r to excite from 0 -> 1:
    mu_01 = np.array([1,1j,0])/np.sqrt(2)

    # we want l to excite from 0 -> 2:
    mu_02 = np.array([1,-1j,0])/np.sqrt(2)

    mu = np.zeros((3,3,3),dtype='complex')
    mu[1,0,:] = mu_01
    mu[2,0,:] = mu_02

    # Now in principle we should do the following to force mu to be Hermitian
    mu_H = mu + np.conjugate(np.transpose(mu,axes=(1,0,2)))
    

    # However, in UFSS, we usually enforce the rotating-wave approximation (RWA) by using what is defined here as
    # "mu", and not "mu_H".  mu excites the ket wavefunction, and the complex conjugate transpose of mu de-excites
    # the ket wavefunction.  Taking the conjugate transpose is handled automatically in UF2 (may not be implemented
    # correctly in RKE yet)

    np.einsum('ijk,k',mu,r)

    # now a quick test. Begin with initial wavefunction in ground state
    psi0 = np.zeros(3)
    psi0[0] = 1
    mu_dot_e_r = np.einsum('ijk,k',mu,r)
    mu_dot_e_l = np.einsum('ijk,k',mu,l)
    

    psi1 = mu_dot_e_r.dot(psi0)
    psi2 = mu_dot_e_l.dot(psi0)



    # Looks like this has worked!  Now we actually only need the part of mu that connects the GSM to the SEM:
    mu_dict = {'0_to_1':mu[1:,:1,:]}

    # in this case we can represent psi0 as a vector of length one:
    psi0 = np.array([1])

    mu_dot_e_r = np.einsum('ijk,k',mu_dict['0_to_1'],r)
    mu_dot_e_l = np.einsum('ijk,k',mu_dict['0_to_1'],l)

    psi1 = mu_dot_e_r.dot(psi0)
    psi2 = mu_dot_e_l.dot(psi0)


    np.savez(os.path.join(folder,'eigenvalues'),**eigenvalues)
    np.savez(os.path.join(folder,'eigenvectors'),**eigenvectors)
    np.savez(os.path.join(folder,'mu'),**mu_dict)


    #re is short for rephasing 2DPE
    re = ufss.Wavepackets(folder,detection_type='complex_polarization')


    # defining the optical pulses in the RWA
    # sigma = 1
    # M = 21 # number of points used to resolve optical pulses
    # Delta = 10 # pulse interval
    # t = np.linspace(-Delta/2,Delta/2,num=M)*sigma
    # dt = t[1] - t[0]
    # ef = ufss.gaussian(t,sigma)
    # c = 1 # center frequency of the laser pulse

    # # Smallwood et al. use a delta function for the local oscillator
    # lo_dt = 1 #### This must never change, because it defines t and tau, by default
    # lo_t =  np.arange(-5,5.2,lo_dt)    
    # lo_dt = lo_t[1] - lo_t[0]
    # lo = np.zeros(lo_t.size,dtype='float')
    # lo[lo.size//2] = 1/lo_dt

    # defining the impulsive pulses in the RWA
    c = 1 # center frequency ofthe laser pulse

    pulse1 = np.array([1]) # pulse 1 amplitude is 1
    t1 = np.array([0]) # impulsive limit, nonzero at a single time point

    pulse2 = np.array([1]) # pulse 2 amplitude is 1
    t2 = np.array([0]) # impulsive limit, nonzero at a single time point

    lo = np.array([1]) # local oscillator amplitude is 1
    lo_t = np.array([0]) # impulsive limit, nonzero at a single time point

    pmc = [(0,1),(2,0)] #pmc stands for phase-matching condition. must include 1 tuple for each pulse
    # excluding the local oscillator

    re.maximum_manifold = 1
    alpha = 0
    re.set_polarization_sequence(['r', 'r', 'r', 'r'])

    re.set_efields([t1,t2,lo_t],[pulse1,pulse2,lo],[c,c,c],pmc)

    re.gamma_res = 20
    re.set_t(0.2, dt = 1) #set tmax for polarization (only thing here that I think is opaque)
    re.pulse_times = [0, 0, 0]

    tau = re.t.copy()[re.t > -15] #dtau is the same as the dt for local oscillator
    T = np.arange(0, 1, 1)
    re.set_pulse_delays([tau, T])


    full_sig_tau_T_wt = re.calculate_signal_all_delays()
    # the signal returned is the signal as a function of tau, T and the fourier conjugate of t -> omega_t
    # inverse fourier transform to get as a function of t:

    t, full_sig_tau_T_t = ft(re.w,full_sig_tau_T_wt,axis = 1)

    signal.append(full_sig_tau_T_t)


def L2_norm(a,b):
    return np.sum(np.abs(b-a)**2)/(np.sum(np.abs(b)**2))

diffs = []

for i in range(e.size):
    averaged_signal = np.mean(np.array(signal[:i+1]), axis = 0)
    diffs.append(L2_norm(averaged_signal,true_signal))

num_simulations = np.arange(e.size)
plt.figure()
plt.semilogy(num_simulations,diffs)
plt.xlabel('Number of simulations')
plt.ylabel('$L_2$ norm difference')
plt.title('Average random sampling')

signal = np.mean(np.array(signal), axis = 0)

plt.figure()
plt.pcolormesh(t, tau, np.abs(signal))
plt.xlabel('t')
plt.ylabel(r'$\tau$')
plt.colorbar()
plt.title('Random averaged signal')

plt.figure()
plt.pcolormesh(t, tau, np.abs(true_signal))
plt.xlabel('t')
plt.ylabel(r'$\tau$')
plt.title('True Signal')
plt.colorbar()

plt.figure()
plt.pcolormesh(t, tau, np.abs(signal-true_signal))
plt.xlabel('t')
plt.ylabel(r'$\tau$')
plt.colorbar()
plt.title('Difference')
plt.show()

