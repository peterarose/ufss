#Standard python libraries
import os
import warnings
import copy
import itertools
import time

#Dependencies - numpy, scipy, matplotlib, pyfftw
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from scipy.sparse import csr_matrix, identity, kron
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq
from scipy.integrate import RK45

#Other parts of this code
from ufss.HLG.base_class import DataOrganizer
from ufss.HLG.eigenstates import PolymerVibrations

class RKE_Wavepackets(PolymerVibrations,DataOrganizer):
    """This class is designed to calculate perturbative wavepackets in the
        light-matter interaction given the hamiltonians of each optical
        manifold

    Args:
        file_path (string): path to folder containing eigenvalues and the
            dipole operator for the system Hamiltonian
        num_conv_points (int): number of desired points for linear 
            convolution. Also number of points used to resolve all optical
            pulse shapes
        dt (float): time spacing used to resolve the shape of all optical
            pulses
        initial_state (int): index of initial state for psi^0 NOT IMPLEMENTED YET
        total_num_time_poitns (int): total number of time points used for
            the spectroscopic calculations.

"""
    def __init__(self,file_path,*, num_conv_points=138, dt=0.1,center = 0,
                 initial_state=0, total_num_time_points = 2000):
        self.size = num_conv_points
        # Initialize time array to be used for all desired delay times
        self.t = np.arange(-(total_num_time_points//2),total_num_time_points//2+total_num_time_points%2)*dt
        self.t += self.t[-(self.size//2+1)]
        self.dt = dt
        parameter_file = os.path.join(file_path,'params.yaml')
        super().__init__(parameter_file,mask_by_occupation_num=True)
        self.base_path = file_path

        self.load_params()

        self.set_diagrams_and_manifolds()

        self.set_molecular_dipoles()

        self.set_bottom_eigensystem()

        self.set_H()

        self.zero_hamiltonians()

        self.rtol = 1E-6
        self.atol = 1E-6

        self.time_to_extend = 0
        self.time_for_next_order = 0

        ############### Optical part

        self.set_homogeneous_linewidth(0.05)

        self.efield_t = np.arange(-(num_conv_points//2),num_conv_points//2+num_conv_points%2) * dt
        self.efield_w = 2*np.pi*fftshift(fftfreq(self.efield_t.size,d=dt))

        # Code will not actually function until the following three empty lists are set by the user
        self.efields = [] #initialize empty list of electric field shapes
        self.polarization_sequence = [] #initialize empty polarization sequence
        self.pulse_times = [] #initialize empty list of pulse arrival times
        
        f = fftshift(fftfreq(self.t.size-self.t.size%2,d=self.dt))
        self.w = 2*np.pi*f

        # Define the unitary operator for each manifold in the RWA given the rotating frequency center
        self.recenter(new_center = center)

    def set_diagrams_and_manifolds(self):
        if len(self.energies) == 2:
            self.diagrams = ['GSB', 'SE']
            self.manifolds = ['GSM','SEM']
            self.diagrams_by_manifold = {'GSM':['GSB'],'SEM':['SE']}
        elif len(self.energies) == 3:
            self.diagrams = ['GSB','SE','ESA']
            self.manifolds = ['GSM','SEM','DEM']
            self.diagrams_by_manifold = {'GSM':['GSB'],'SEM':['SE','ESA']}

    def set_H(self,*,truncation_size = None):
        if truncation_size:
            self.truncation_size = truncation_size
            self.set_vibrations()
        self.H0 = self.manifold_hamiltonian(0)
        self.H1 = self.manifold_hamiltonian(1)
        
        if 'DEM' in self.manifolds:
            self.H2 = self.manifold_hamiltonian(2)

    def make_H_dense(self):
        self.H0 = self.H0.toarray()
        self.H1 = self.H1.toarray()
        if 'DEM' in self.manifolds:
            self.H2 = self.H2.toarray()

    def dH0(self,t,psi_in):
        return -1j*self.H0.dot(psi_in)

    def dH1(self,t,psi_in):
        return -1j*self.H1.dot(psi_in)

    def dH2(self,t,psi_in):
        return -1j*self.H2.dot(psi_in)

    def RKU_onetimestep(self,psi0,t0,tf,*,manifold_num = 0):
        dH = [self.dH0,self.dH1,self.dH2]
        fun = dH[manifold_num]
        rk45 = RK45(fun,t0,psi0,tf,atol=self.atol,rtol=self.rtol,vectorized=True)
        while rk45.t < tf:
            rk45.step()
        psi_final = rk45.y
        return psi_final

    def RKU(self,psi0,t0,tvalues,*,manifold_num = 0):
        t_last = t0
        states = np.zeros((psi0.size,tvalues.size),dtype='complex')
        psi_last = psi0
        for i in range(tvalues.size):
            t_next = tvalues[i]
            psi_next = self.RKU_onetimestep(psi_last,t_last,t_next,
                                               manifold_num=manifold_num)
            states[:,i] = psi_next
            t_last = t_next
            psi_last = psi_next

        return states

    def set_molecular_dipoles(self,*,dipoles = None):
        """Load molecular dipoles from params file, or override with input
dipoles - must be a numpy ndarray, with shape (n,3) where n is the number of sites"""
        if type(dipoles) is np.ndarray:
            self.molecular_dipoles = dipoles
        else:
            self.molecular_dipoles = np.array(self.params['dipoles'],dtype='float')

        self.set_single_to_double_dipole_matrix()

    def set_single_to_double_dipole_matrix(self):
        """Given a set of dipoles for transitions from the ground to the
singly excited states, constructs the dipole transitions that take the
system from the singly excited states to the various doubly excited states
"""
        singly_excited = np.arange(self.molecular_dipoles.shape[0])
        doubly_excited = list(itertools.combinations(singly_excited,2))
        mat = np.zeros((len(singly_excited),len(doubly_excited),3))
        for i in range(len(singly_excited)):
            for j in range(len(doubly_excited)):
                tup = doubly_excited[j]
                if i == tup[0]:
                    mat[i,j,:] = self.molecular_dipoles[singly_excited[tup[1]]]
                elif i == tup[1]:
                    mat[i,j,:] = self.molecular_dipoles[singly_excited[tup[0]]]
        self.molecular_dipoles_SEM_to_DEM = mat

    def set_bottom_eigensystem(self):
        """Calculate or load lowest ground-state eigenvalue and eigenvector,
            and lowest singly excited eigenvalue.
            Note: would like to expand this to allow initial conditions other 
            than bottom ground state
"""
        try:
            self.load_bottom_eigensystem()
        except:
            self.calculate_bottom_eigensystem()
            self.save_bottom_eigensystem()

        self.setup_eigensystem()

    def calculate_bottom_eigensystem(self):
        ge0, gv0 = self.eigs(1,0)
        se0, sv0 = self.eigs(1,1)
        self.eigenvalues = [ge0,se0]
        self.eigenvectors = gv0

    def setup_eigensystem(self):
        psi0 = self.eigenvectors[:,0]
        psi0_extension = np.ones((psi0.size,self.t.size),dtype='complex')
        psi0 = psi0[:,np.newaxis] * psi0_extension
        self.psi0 = {'psi':psi0,'manifold_num':0,'bool_mask':np.array([None])}
        ge0 = self.eigenvalues[0][0]
        se0 = self.eigenvalues[1][0]
        self.ground_ZPE = ge0
        self.ground_to_excited_transition = se0 - ge0

    def zero_hamiltonians(self):
        # Subtract out GSM zpe
        self.H0 -= identity(self.H0.shape[0]) * self.ground_ZPE
        self.H1 -= identity(self.H1.shape[0]) * self.ground_ZPE
        if 'DEM' in self.manifolds:
            self.H2 -= identity(self.H2.shape[0]) * self.ground_ZPE

        # Default RWA is the gap from lowest energy ground state
        # and lowest energy singly excited state
        self.H1 -= identity(self.H1.shape[0]) * self.ground_to_excited_transition
        if 'DEM' in self.manifolds:
            self.H2 -= identity(self.H2.shape[0]) * 2 * self.ground_to_excited_transition

    def save_bottom_eigensystem(self):
        np.savez(os.path.join(self.base_path,'bottom_eigenvalues'),GSM=self.eigenvalues[0],SEM=self.eigenvalues[1])
        np.save(os.path.join(self.base_path,'bottom_eigenvectors'),self.eigenvectors)

    def load_bottom_eigensystem(self):
        arch = np.load(os.path.join(self.base_path,'bottom_eigenvalues.npz'))
        self.eigenvalues = [arch['GSM'],arch['SEM']]
        self.eigenvectors = np.load(os.path.join(self.base_path,'bottom_eigenvectors.npy'))

    def set_homogeneous_linewidth(self,gamma):
        self.gamma = gamma

    def recenter(self,new_center = 0):
        self.H1 -= identity(self.H1.shape[0]) * new_center
        if 'DEM' in self.manifolds:
            self.H2 -= identity(self.H2.shape[0]) * new_center

    def get_closest_index_and_value(self,value,array):
        """Given an array and a desired value, finds the closest actual value
            stored in that array, and returns that value, along with its 
            corresponding  array index
"""
        index = np.argmin(np.abs(array - value))
        value = array[index]
        return index, value

    def extend_wavefunction(self,psi_dict,pulse_start_ind,pulse_end_ind,
                            gamma_end_ind = None):
        """Perturbative wavefunctions are calculated only during the time where the given pulse
            is non-zero.  This function extends the wavefunction beyond those bounds by taking 
            all values before the interaction to be zero, and using RK45 method to extend the 
            wavefunction forward
"""
        t_start = time.time()

        pulse_t_slice = slice(pulse_start_ind, pulse_end_ind,1)

        psi = psi_dict['psi']

        m_num = psi_dict['manifold_num']

        total_psi = np.zeros((psi.shape[0],self.t.size),dtype='complex')
        total_psi[:,pulse_t_slice] = psi
        psi_last = total_psi[:,pulse_end_ind-1]
        t_last = self.t[pulse_end_ind-1]
        total_psi[:,pulse_end_ind:gamma_end_ind] = self.RKU(psi_last,t_last,
                                                            self.t[pulse_end_ind:gamma_end_ind],
                                                            manifold_num = m_num)

        psi_dict['psi'] = total_psi
        self.time_to_extend += time.time() - t_start
        return psi_dict

    def set_polarization_sequence(self,polarization_list,*,reset_psis=True):
        """Sets the sequences used for either parallel or crossed pump and probe
        
        Args:
            polarization_list (list): list of four strings, can be 'x' or 'y'
        Returns:
            None: sets the attribute polarization sequence
"""

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        pol_options = {'x':x,'y':y,'z':z}

        self.polarization_sequence = [pol_options[pol] for pol in polarization_list]

        if reset_psis:
            self.psis = dict()


    def dipole_matrix(self,starting_manifold_num,next_manifold_num,pulse_number):
        """Calculates the dipole matrix that connects from one 
            manifold to the next, using the known dipole moments and the efield 
            polarization, determined by the pulse number.
"""
        pol = self.polarization_sequence[pulse_number]
        upper_manifold_num = max(starting_manifold_num,next_manifold_num)
        
        if abs(starting_manifold_num - next_manifold_num) != 1:
            warnings.warn('Can only move from manifolds 0 to 1 or 1 to 2')
            return None
        
        # Condon approximation
        vib_identity = identity(self.H0.shape[0])
        
        if upper_manifold_num == 1:
            d_vec = self.molecular_dipoles.dot(pol)
            d_mat = d_vec[:,np.newaxis]
            overlap_matrix = kron(d_mat,vib_identity)
            
        elif upper_manifold_num == 2:
            d_mat = self.molecular_dipoles_SEM_to_DEM.dot(pol)
            overlap_matrix = kron(d_mat.T,vib_identity)

        if starting_manifold_num > next_manifold_num:
            # Take transpose if transition is down rather than up
            overlap_matrix = np.conjugate(overlap_matrix.T)

        return overlap_matrix.tocsr()

    def next_order(self,psi_in_dict,manifold_change,*,pulse_number = 0,gamma = 0,
                   new_manifold_mask = None):
        """This function connects psi_p to psi+pj^(*)

        Args:
            psi_in_dict (dict): input wavefunction dictionary
            manifold_change (int): is either +/-1 (up or down)
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
        
        Return:
            psi_dict (dict): next-order wavefunction
"""
        t_start = time.time()
        pulse_time = self.pulse_times[pulse_number]
        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2

        t_slice = slice(pulse_start_ind, pulse_end_ind,1)
        t = self.t[t_slice]

        if gamma != 0:
            gamma_end_ind = pulse_time_ind + int(6.91/self.gamma/self.dt)
        else:
            gamma_end_ind = None
        
        psi_in = psi_in_dict['psi'][:,t_slice].copy()
        
        starting_manifold_num = psi_in_dict['manifold_num']
        next_manifold_num = starting_manifold_num + manifold_change
        
        dipole_matrix = self.dipole_matrix(starting_manifold_num,next_manifold_num,
                                            pulse_number)
        
        if pulse_number == 'impulsive':
            # This is broken!
            psi_imp = dipole_matrix.dot(psi_in[:,self.size//2])
            psi = np.zeros((psi_imp.size,t.size),dtype='complex')
            psi[:,self.size//2] = psi_imp
            t_last = t[self.size//2]
            t_request = t[self.size//2+1:]
            psi[:,self.size//2+1:] = self.RKU(psi_imp,t_last,t_request,
                                              manifold_num=next_manifold_num)
        else:
            if next_manifold_num > starting_manifold_num:
                efield = self.efields[pulse_number]
            else:
                efield = np.conjugate(self.efields[pulse_number])
            psi = np.zeros((dipole_matrix.shape[0],t.size),dtype='complex')
            psi[:,0] = 1j*dipole_matrix.dot(psi_in[:,0]) * efield[0] * self.dt
            for i in range(1,efield.size):
                psi[:,i] = self.RKU_onetimestep(psi[:,i-1],t[i-1],t[i],
                                                manifold_num = next_manifold_num)
                psi[:,i] += 1j*dipole_matrix.dot(psi_in[:,i]) * efield[i] * self.dt
            

        psi_dict = {'psi':psi,'manifold_num':next_manifold_num,'bool_mask':np.array([None])}

        self.time_for_next_order += time.time() - t_start

        psi_dict = self.extend_wavefunction(psi_dict,pulse_start_ind,pulse_end_ind,
                                            gamma_end_ind = gamma_end_ind)

        return psi_dict

    def up(self,psi_in_dict,*,pulse_number = 0,gamma=0,new_manifold_mask = None):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            psi_in_dict (dict): input wavefunction dictionary
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            gamma (float): optical dephasing (only use with final interaction)
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (dict): output from method next_order
"""

        return self.next_order(psi_in_dict,1,gamma=gamma,
                               pulse_number = pulse_number)

    def down(self,psi_in_dict,*,pulse_number = 0,gamma=0,new_manifold_mask = None):
        """This method connects psi_p to psi_pj^* where the next order psi
            is one manifold below the current manifold.

        Args:
            psi_in_dict (dict): input wavefunction dictionary
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            gamma (float): optical dephasing (only use with final interaction)
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (dict): output from method next_order
"""

        return self.next_order(psi_in_dict,-1,gamma=gamma,
                               pulse_number = pulse_number)

    ### Tools for taking the expectation value of the dipole operator with perturbed wavepackets
    
    def dipole_down(self,psi_in_dict,*,pulse_number = -1):
        """This method is similar to the method down, but does not
involve the electric field shape or convolutions. It is the action of 
the dipole operator on a ket without TDPT effects.  It also includes
the dot product of the final electric field polarization vector."""
        psi_in = psi_in_dict['psi']
        starting_manifold_num = psi_in_dict['manifold_num']
        next_manifold_num = starting_manifold_num - 1
        
        # This function is always used as the final interaction to
        # produce the polarization field, which is a vector quantity
        # However we assume that we will calculate a signal, which
        # invovles the dot product of the polarization field with the
        # local oscillator vector. We do this now to avoid carrying
        # around the cartesian coordinates of the polarization field
        
        overlap_matrix = self.dipole_matrix(starting_manifold_num,next_manifold_num,
                                                            pulse_number = pulse_number)

        psi = overlap_matrix.dot(psi_in)

        psi_dict = {'psi':psi,'manifold_num':next_manifold_num,'bool_mask':np.array([None])}

        return psi_dict

    def dipole_expectation(self,bra_dict_original,ket_dict_original,*,pulse_number = -1):
        """Computes the expectation value of the two wavefunctions with respect 
            to the dipole operator.  Both wavefunctions are taken to be kets, and the one named 'bra' is
converted to a bra by taking the complex conjugate."""
        pulse_time = self.pulse_times[pulse_number]
        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        gamma_end_ind = pulse_time_ind + int(6.91/self.gamma/self.dt)

        # The signal is zero before the final pulse arrives, and persists
        # until it decays. Therefore we avoid taking the sum at times
        # where the signal is zero. This is captured by t_slice
        t_slice = slice(pulse_start_ind, gamma_end_ind,None)

        bra_in = bra_dict_original['psi'][:,t_slice].copy()
        ket_in = ket_dict_original['psi'][:,t_slice].copy()
        
        manifold1_num = bra_dict_original['manifold_num']
        manifold2_num = ket_dict_original['manifold_num']

        bra_dict = {'manifold_num':manifold1_num,'psi':bra_in}
        ket_dict = {'manifold_num':manifold2_num,'psi':ket_in}

        if np.abs(manifold1_num - manifold2_num) != 1:
            warnings.warn('Dipole only connects manifolds 0 to 1 or 1 to 2')
            return None

        if manifold1_num > manifold2_num:
            bra_dict = self.dipole_down(bra_dict,
                                        pulse_number = pulse_number)
        else:
            ket_dict = self.dipole_down(ket_dict,
                                        pulse_number = pulse_number)

        bra = bra_dict['psi']
        ket = ket_dict['psi']

        exp_val = np.sum(np.conjugate(bra) * ket,axis=0)
        
        # Initialize return array with zeros
        ret_val = np.zeros(self.t.size,dtype='complex')
        # set non-zero values using t_slice
        ret_val[t_slice] = exp_val
        return ret_val

    def integrated_dipole_expectation(self,bra_dict_original,ket_dict_original,*,pulse_number = -1):
        """Given two wavefunctions, this computes the expectation value of the two with respect 
to the dipole operator.  Both wavefunctions are taken to be kets, and the one named 'bra' is
converted to a bra by taking the complex conjugate.  This assumes that the signal will be
frequency integrated."""
        pulse_time = self.pulse_times[pulse_number]
        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2
        
        # The signal is zero before the final pulse arrives, and persists
        # until it decays. However, if no frequency information is
        # required, fewer time points are needed for this t_slice
        t_slice = slice(pulse_start_ind, pulse_end_ind,None)

        bra_in = bra_dict_original['psi'][:,t_slice].copy()
        ket_in = ket_dict_original['psi'][:,t_slice].copy()
        
        
        manifold1_num = bra_dict_original['manifold_num']
        manifold2_num = ket_dict_original['manifold_num']

        bra_dict = {'manifold_num':manifold1_num,'psi':bra_in,'bool_mask':np.array([None])}
        ket_dict = {'manifold_num':manifold2_num,'psi':ket_in,'bool_mask':np.array([None])}

        if np.abs(manifold1_num - manifold2_num) != 1:
            warnings.warn('Dipole only connects manifolds 0 to 1 or 1 to 2')
            return None

        if manifold1_num > manifold2_num:
            bra_new_mask = ket_dict['bool_mask']
            bra_dict = self.dipole_down(bra_dict,
                                        pulse_number = pulse_number)
        else:
            ket_new_mask = bra_dict['bool_mask']
            ket_dict = self.dipole_down(ket_dict,
                                        pulse_number = pulse_number)

        bra = bra_dict['psi']
        ket = ket_dict['psi']

        exp_val = np.sum(np.conjugate(bra) * ket,axis=0)
        return exp_val

    def polarization_to_signal(self,P_of_t_in,*,return_polarization=False,
                               local_oscillator_number = -1):
        """This function generates a frequency-resolved signal from a polarization field
local_oscillator_number - usually the local oscillator will be the last pulse in the list self.efields"""
        pulse_time = self.pulse_times[local_oscillator_number]
        if self.gamma != 0:
            exp_factor = np.exp(-self.gamma * (self.t-pulse_time))
            P_of_t_in *= exp_factor
        P_of_t = P_of_t_in
        if return_polarization:
            return P_of_t

        if local_oscillator_number == 'impulsive':
            efield = np.exp(1j*self.w*(pulse_time))
        else:
            pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

            pulse_start_ind = pulse_time_ind - self.size//2
            pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2

            t_slice = slice(pulse_start_ind, pulse_end_ind,None)
            efield = np.zeros(self.t.size,dtype='complex')
            efield[t_slice] = self.efields[local_oscillator_number]
            efield = fftshift(ifft(ifftshift(efield)))*len(P_of_t)*(self.t[1]-self.t[0])/np.sqrt(2*np.pi)

        if P_of_t.size%2:
            P_of_t = P_of_t[:-1]
            efield = efield[:len(P_of_t)]

        P_of_w = fftshift(ifft(ifftshift(P_of_t)))*len(P_of_t)*(self.t[1]-self.t[0])/np.sqrt(2*np.pi)
        
        signal = np.imag(P_of_w * np.conjugate(efield))
        return signal

    def polarization_to_integrated_signal(self,P_of_t,*,
                                          local_oscillator_number = -1):
        """This function generates a frequency-integrated signal from a polarization field
local_oscillator_number - usually the local oscillator will be the last pulse in the list self.efields
"""
        pulse_time = self.pulse_times[local_oscillator_number]
        pulse_time_ind = np.argmin(np.abs(self.t - self.delay_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2

        t_slice = slice(pulse_start_ind, pulse_end_ind,1)
        t = self.t[t_slice]
        if self.gamma != 0:
            exp_factor = np.exp(-self.gamma * (t-pulse_time))
            P_of_t *= exp_factor

        if local_oscillator_number == 'impulsive':
            signal = P_of_t[self.size//2]
        else:
            efield = self.efields[local_oscillator_number]
            signal = np.trapz(np.conjugate(efield)*P_of_t,x=t)
        
        return np.imag(signal)

    
