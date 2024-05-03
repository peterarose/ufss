import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import eigs, eigsh
import itertools
from scipy.linalg import block_diag, eig, expm, eigh
from scipy.sparse import save_npz, load_npz, csr_matrix, csc_matrix
import scipy.sparse as sp
from scipy.special import binom
import yaml
import copy
import warnings
import os
import time

from .Hamiltonians import PolymerVibrations

from .general_Liouvillian_classes import LiouvillianConstructor

class OhmicSpectralDensity:
    """Creates an ohmic spectral density function
"""

    def __init__(self,lam,gam,kT,*,cutoff='lorentz-drude'):
        """
        Args:
            lam (float): bath coupling strength
            gam (float): bath decay rate or cutoff frequency
            kT (float): temperature
            cutoff (str): high-frequency cutoff function (options: 'lorentz-drude or 
                exponential)
"""
        self.lam = lam
        self.gam = gam
        self.kT = kT

        if cutoff == 'lorentz-drude':
            self.J = self.lorentz_drude
        elif cutoff == 'exponential':
            self.J = self.exponential

    def ohmic(self,w):
        return 2 * w * self.lam/self.gam

    def lorentz_drude(self,w):
        return self.ohmic(w) * self.gam**2 / (w**2 + self.gam**2)
    
    def exponential(self,w):
        return self.ohmic(w) * np.exp( -np.abs(w) / self.gam)

    def temp_dist(self,w):
        return (1+self.coth(w))/2

    def coth(self,w):
        if self.kT == 0:
            return np.sign(w)
        else:
            return np.cosh(w/(2*self.kT))/np.sinh(w/(2*self.kT))

    def C_vec(self,w):
        f = self.J(w)
        zero_ind = np.where(w==0)[0]
        if zero_ind.size == 1:
            zi = zero_ind[0]
            f[:zi] *= self.temp_dist(w[:zi])
            f[zi+1:] *= self.temp_dist(w[zi+1:])
            f[zi] = 2*self.kT*self.lam/self.gam
        elif zero_ind.size == 0:
            f = f * self.temp_dist(w)
        else:
            raise Exception("Can't handle more than one value of w = 0")
        return f

    def C_scal(self,w):
        if w == 0:
            f = 2*self.kT*self.lam/self.gam
        else:
            f = self.J(w) * self.temp_dist(w)
        return f

    def __call__(self,w):
        if type(w) is np.ndarray:
            return self.C_vec(w)
        else:
            return self.C_scal(w)

class WhiteNoiseSpectralDensity:
    """Creates a white noie (flat) spectral density function
"""

    def __init__(self,dephasing_rate,relaxation_rate,kT):
        """
        Args:
            dephasing_rate (float): value of spectral density at 0 frequency
            relaxation_rate (float): value of spectral density at all 
                positive frequencies
            kT (float): temperature
"""
        self.gamma_d = dephasing_rate
        self.gamma_r = relaxation_rate
        self.kT = kT

        self.constant = self.gamma_r

    def C_vec(self,w):
        f = self.constant * np.ones(w.size)
        neg_inds = np.where(w<0)[0]
        zero_inds = np.where(np.isclose(w,0,atol=1E-12))
        
        if self.kT > 0:
            f[neg_inds] *= np.exp(w[neg_inds]/self.kT)
        else:
            f[neg_inds] = 0
        f[zero_inds] = self.gamma_d
        return f

    def C_scal(self,w):
        if np.isclose(w,0,atol=1E-12):
            f = self.gamma_d
        elif w < 0:
            if self.kT > 0:
                f = self.constant * np.exp(w/self.kT)
            else:
                f = 0
        else:
            f = self.constant
        return f

    def __call__(self,w):
        if type(w) is np.ndarray:
            return self.C_vec(w)
        else:
            return self.C_scal(w)

class RedfieldConstructor:

    def __init__(self,folder,*,conserve_memory = False,do_nothing=False):
        self.base_path = folder
        self.load_path = os.path.join(self.base_path,'closed')
        self.save_path = os.path.join(self.base_path,'open')
        os.makedirs(self.save_path,exist_ok=True)
        with open(os.path.join(folder,'params.yaml')) as yaml_stream:
            self.params = yaml.load(yaml_stream,Loader=yaml.SafeLoader)

        self.load_eigensystem()
        self.load_mu()

        if self.manifolds[0] == 'all_manifolds':
            separable_manifolds = False
        else:
            separable_manifolds = True
            
        self.PolyVib = PolymerVibrations(os.path.join(folder,'params.yaml'),
                                         separable_manifolds=separable_manifolds)

        self.set_spectral_densities()

        try:
            self.secular = self.params['bath']['secular']
        except KeyError:
            self.secular = False
            
        if do_nothing:
            pass
        else: 
            self.set_Y()
            self.set_L()
            self.save_L()

            if not conserve_memory:
                self.load_mu()

                self.set_mu_Liouville_space()
                self.save_mu()

    def set_spectral_densities(self):
        self.SD = {}
        self.SD['site_bath'] = self.make_spectral_density(self.params['bath']['site_bath'])
        self.SD['vibration_bath'] = self.make_spectral_density(self.params['bath']['site_bath'])
        try:
            self.SD['site_internal_conversion_bath'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath'])
            if self.PolyVib.separable_manifolds == True:
                raise Exception('Cannot model site internal conversion processes when input Hamiltonian has been divided into separate manifolds')

        except KeyError:
            pass

    def make_spectral_density(self,bath_dict):
        kT = bath_dict['temperature']
        spectrum_type = bath_dict['spectrum_type']

        if spectrum_type == 'ohmic':
            lam = bath_dict['coupling']
            gam = bath_dict['cutoff_frequency']
            cutoff = bath_dict['cutoff_function']
            SD = OhmicSpectralDensity(lam,gam,kT,cutoff=cutoff)
        elif spectrum_type == 'white-noise':
            deph_rate = bath_dict['dephasing_rate']
            relax_rate = bath_dict['relaxation_rate']
            SD = WhiteNoiseSpectralDensity(deph_rate,relax_rate,kT)
        else:
            raise Exception('spectrum_type not supported')

        return SD

    def load_eigensystem(self):
        """Load eigenvalues and eigenvectors of Hamiltonian
"""
        e_save_name = os.path.join(self.load_path,'eigenvalues.npz')
        v_save_name = os.path.join(self.load_path,'eigenvectors.npz')
        with np.load(e_save_name) as e_arch:
            self.manifolds = list(e_arch.keys())
            self.eigenvalues = {key:e_arch[key] for key in self.manifolds}
        with np.load(v_save_name) as v_arch:
            self.eigenvectors = {key:v_arch[key] for key in self.manifolds}

        return None

    def load_mu(self):
        mu_save_name = os.path.join(self.load_path,'mu.npz')
        with np.load(mu_save_name) as mu_arch:
            self.mu_keys = list(mu_arch.keys())
            self.mu = {key:mu_arch[key] for key in self.mu_keys}

    def make_dissipation_tensor(self,CO,manifold_key,SD):
        """Make a Redfield-style dissipation tensor using the eigenbasis
            decomposition of the system-bath coupling operator (CO)
        Args:
            CO (np.ndarray): system-bath coupling operator
            manifold_key (str): string representing the manifold
            SD (callable): spectral density function of the bath
        Returns:
            np.ndarray: dissipation tensor, from which R may be made
"""
        e = self.eigenvalues[manifold_key]
        v = self.eigenvectors[manifold_key]

        CO = np.conjugate(v.T).dot(CO).dot(v)

        CO_dagger = np.conjugate(CO.T)

        eki = e[np.newaxis,np.newaxis,:,np.newaxis] - e[:,np.newaxis,np.newaxis,np.newaxis]

        cki = np.zeros(eki.shape)
        for i in range(e.size):
            for k in range(e.size):
                cki[i,0,k,0] = SD(eki[i,0,k,0])

        Y = cki * CO[:,np.newaxis,:,np.newaxis] * CO_dagger[np.newaxis,:,np.newaxis,:]

        return Y

    def make_Y1(self,manifold_key):
        """Creates vibrational dissipation tensor, based upon a bilinear
            coupling of the explicit vibrations to the bath
        Args:
            manifold_key (str): can be 'all_manifolds' or '0','1','2',...
"""
        spectral_density = self.SD['vibration_bath']
        size = self.eigenvalues[manifold_key].size
        Y1 = np.zeros((size,size,size,size))
        if manifold_key == 'all_manifolds':
            pass
        else:
            manifold_num = int(manifold_key)

        for p in range(self.PolyVib.num_vibrations):
            # creation operator
            ad_p = self.PolyVib.total_ups[p]
            if manifold_key == 'all_manifolds':
                x_p = (ad_p + ad_p.T)/np.sqrt(2)
            else:
                ad_p_man = self.PolyVib.extract_vibronic_manifold(ad_p,manifold_num)
                x_p = (ad_p_man + ad_p_man.T)/np.sqrt(2)
            Y1 += self.make_dissipation_tensor(x_p,manifold_key,spectral_density)

        
        return Y1

    def make_Y2(self,manifold_key):
        """Creates an electronic dissipation tensor, based upon 
            uncorrelated baths associated with each diabatic excited state
        Args:
            manifold_key (str): can be 'all_manifolds' or '0','1','2',...
"""
        spectral_density = self.SD['site_bath']
        size = self.eigenvalues[manifold_key].size
        Y2 = np.zeros((size,size,size,size))
 
        for n in range(self.PolyVib.Polymer.num_sites):
            # site projector operators
            P_n = self.PolyVib.Polymer.occupied_list[n]
            P_n = self.extract_electronic_operator(P_n,manifold_key)

            P_n = np.kron(P_n,self.PolyVib.vibrational_identity)
            Y2 += self.make_dissipation_tensor(P_n,manifold_key,spectral_density)

        return Y2

    def extract_electronic_operator(self,O,manifold_key):
        if manifold_key == 'all_manifolds':
            O = self.PolyVib.Polymer.extract_electronic_subspace(O,0,self.PolyVib.maximum_manifold)
        else:
            manifold_num = int(manifold_key)
            O = self.PolyVib.Polymer.extract_manifold(O,manifold_num)

        return O

    def make_Y3(self):
        """Creates non-adiabatic dissipation tensor by using the sigma_x
            Pauli spin matrix as the coupling operator for each 2LS
"""
        spectral_density = self.SD['site_internal_conversion_bath']
        manifold_key = 'all_manifolds'
        size = self.eigenvalues[manifold_key].size
        Y3 = np.zeros((size,size,size,size))
        for n in range(self.PolyVib.Polymer.num_sites):
            # electronic excitation creation operator
            up_n = self.PolyVib.Polymer.up_list[n]
            up_n = self.extract_electronic_operator(up_n,manifold_key)
            up_n = np.kron(up_n,self.PolyVib.vibrational_identity)
            sigma_x_n = up_n + up_n.T

            Y3 += self.make_dissipation_tensor(sigma_x_n,manifold_key,spectral_density)
                
        return Y3

    def make_R_from_Y(self,Y,manifold_key):
        """Given the dissipation operator Y, constructs the Redfield tensor
        Args:
            Y (np.ndarray): 4-index array dissipation tensor
            manifold_key (str): can be 'all_manifolds' or '0','1','2',...
"""
        R = np.zeros(Y.shape)

        e = self.eigenvalues[manifold_key]

        R = -(Y + np.conjugate(Y.transpose((1,0,3,2))))
        delta_jl = np.eye(e.size)[np.newaxis,:,np.newaxis,:]
        delta_ik = np.eye(e.size)[:,np.newaxis,:,np.newaxis]
        Y_trace = np.einsum('nijn',Y)
        R += delta_jl * Y_trace[:,np.newaxis,:,np.newaxis]
        R += delta_ik * Y_trace[np.newaxis,:,np.newaxis,:]

        if self.secular:

            diffs = np.abs(e[:,np.newaxis,np.newaxis,np.newaxis]
                         - e[np.newaxis,:,np.newaxis,np.newaxis]
                         - e[np.newaxis,np.newaxis,:,np.newaxis]
                         + e[np.newaxis,np.newaxis,np.newaxis,:])

            nonzero_inds = np.where(diffs>1E-12)

            R[nonzero_inds] = 0

        return R

    def make_R_from_Y_coherence(self,Yket,Ybra,ket_manifold_key,bra_manifold_key):
        """Makes Redfield tensor for optical coherence states, in the case
            that each electronic excitation manifold is separable.
        Args:
            Yket (np.ndarray): 4-index array dissipation tensor for the ket
                excitation manifold
            Ybra (np.ndarray): 4-index array dissipation tensor for the bra
                excitation manifold
            ket_manifold_key (str): bra manifold, can be '0','1','2',...
            bra_manifold_key (str): ket manifold, can be '0','1','2',...
"""
        if ket_manifold_key == bra_manifold_key:
            raise Exception('You must use make_R_from_Y for ket and bra manifold keys equal to each other')
        ket_size = Yket.shape[0]
        bra_size = Ybra.shape[0]
        eket = self.eigenvalues[ket_manifold_key]
        ebra = self.eigenvalues[bra_manifold_key]

        delta_jl = np.eye(ebra.size)[np.newaxis,:,np.newaxis,:]
        delta_ik = np.eye(eket.size)[:,np.newaxis,:,np.newaxis]
        Yket_trace = np.einsum('nijn',Yket)
        Ybra_trace = np.einsum('nijn',Ybra)
        R = delta_jl * Yket_trace[:,np.newaxis,:,np.newaxis]
        R += delta_ik * Ybra_trace[np.newaxis,:,np.newaxis,:]

        if self.secular:

            diffs = np.abs(eket[:,np.newaxis,np.newaxis,np.newaxis]
                         - ebra[np.newaxis,:,np.newaxis,np.newaxis]
                         - eket[np.newaxis,np.newaxis,:,np.newaxis]
                         + ebra[np.newaxis,np.newaxis,np.newaxis,:])

            nonzero_inds = np.where(diffs>1E-12)

            R[nonzero_inds] = 0

        return R

    def flatten_R(self,R):
        new_shape = (R.shape[0] * R.shape[1], R.shape[2] * R.shape[3])
        new_R = np.zeros(new_shape)

        for k,l in itertools.product(range(R.shape[2]),range(R.shape[3])):
            n = k*R.shape[3] + l
            new_rho = R[:,:,k,l]
            new_R[:,n] += new_rho.flatten()

        return new_R

    def set_L(self):
        self.L = {}
        if self.manifolds[0] == 'all_manifolds':
            L_key = 'all_manifolds'
            R = self.make_flat_R(L_key,L_key)
            U = self.make_U(L_key,L_key)
            self.L[L_key] = U - R
        else:
            for ket_man,bra_man in itertools.product(self.manifolds,
                                                     self.manifolds):
                L_key = ket_man + bra_man
                R = self.make_flat_R(ket_man,bra_man)
                U = self.make_U(ket_man,bra_man)
                self.L[L_key] = U - R

    def save_L(self):
        np.savez(os.path.join(self.save_path,'L.npz'),**self.L)

    def make_flat_R(self,ket_manifold,bra_manifold):
        if ket_manifold == 'all_manifolds':
            Y = self.Y['all_manifolds']
            R = self.make_R_from_Y(Y,ket_manifold)
        elif ket_manifold == bra_manifold:
            Y = self.Y[ket_manifold]
            R = self.make_R_from_Y(Y,ket_manifold)
        elif ket_manifold != bra_manifold:
            Y_ket = self.Y[ket_manifold]
            Y_bra = self.Y[bra_manifold]
            R = self.make_R_from_Y_coherence(Y_ket,Y_bra,ket_manifold,bra_manifold)
            
        R2d = self.flatten_R(R)
        return R2d

    def set_Y(self):
        self.Y = {}
        for manifold in self.manifolds:
            self.Y[manifold] = (self.make_Y1(manifold)
                                + self.make_Y2(manifold) )
            if manifold == 'all_manifolds':
                self.Y[manifold] += self.make_Y3()

    def make_U(self,ket_manifold,bra_manifold):
        H_ket = np.diag(self.eigenvalues[ket_manifold])
        H_bra = np.diag(self.eigenvalues[bra_manifold])
        U_ins = LiouvillianConstructor.make_commutator_instructions2(-1j*H_ket,-1j*H_bra)
        U = LiouvillianConstructor.make_Liouvillian(U_ins,sparse=True)
        return U

    def mu_key_to_manifold_keys(self,key):
        if self.manifolds[0] == 'all_manifolds':
            starting_key = 'all_manifolds'
            ending_key = 'all_manifolds'
        else:
            starting_key, ending_key = key.split('_to_')
        return starting_key, ending_key


    def make_mu_by_manifold_ket(self,old_manifold,change):
        i,j = old_manifold
        i2 = i + change
        j2 = j
        
        if i2 >= 0 and i2 <= self.PolyVib.maximum_manifold:
            pass
        else:
            return None, None

        if i2 > i:
            mu_key = str(i) + '_to_' + str(i2)
            mu = self.mu[mu_key]
        else:
            mu_key = str(i2) + '_to_' + str(i)
            mu = self.mu[mu_key].transpose(1,0,2)

        j_size = self.eigenvalues[str(j)].size
        i_size = self.eigenvalues[str(i)].size
        i2_size = self.eigenvalues[str(i2)].size
        
        bra_eye = np.eye(j_size)
        old_key = str(i) + str(j)
        new_key = str(i2) + str(j2)
        
        mu_shape = (i2_size*j_size,i_size*j_size,3)
        new_mu = np.zeros(mu_shape,dtype='complex')
        for i in range(3):
            mu_i = np.kron(mu[:,:,i],bra_eye)
            new_mu[:,:,i] = mu_i

        if np.allclose(np.imag(new_mu),0):
            new_mu = np.real(new_mu)
        mu_key = old_key + '_to_' + new_key
        return mu_key, new_mu

    def make_mu_by_manifold_bra(self,old_manifold,change):
        i,j = old_manifold
        i2 = i
        j2 = j + change
        
        if j2 >= 0 and j2 <= self.PolyVib.maximum_manifold:
            pass
        else:
            return None, None

        if j2 > j:
            mu_key = str(j) + '_to_' + str(j2)
            mu = self.mu[mu_key].transpose(1,0,2)
        else:
            mu_key = str(j2) + '_to_' + str(j)
            mu = self.mu[mu_key]

        i_size = self.eigenvalues[str(i)].size
        j_size = self.eigenvalues[str(j)].size
        j2_size = self.eigenvalues[str(j2)].size
        
        ket_eye = np.eye(i_size)
        old_key = str(i) + str(j)
        new_key = str(i2) + str(j2)
        
        mu_shape = (i_size*j2_size,i_size*j_size,3)
        new_mu = np.zeros(mu_shape,dtype='complex')
        for i in range(3):
            mu_i = np.kron(ket_eye,mu[:,:,i].T)
            new_mu[:,:,i] = mu_i

        if np.allclose(np.imag(new_mu),0):
            new_mu = np.real(new_mu)
        mu_key = old_key + '_to_' + new_key
        return mu_key, new_mu

    def make_mu_unseparable_manifolds(self,change,ket_flag):
        mu = self.mu['up']
        if ket_flag:
            mu_key = 'ket'
        else:
            mu = mu.transpose(1,0,2)
            mu_key = 'bra'
        if change == 1:
            mu_key += '_up'
        elif change == -1:
            mu = mu.transpose(1,0,2)
            mu_key += '_down'
        else:
            raise Exception('change must be either +1 or -1')
        H_size = self.eigenvalues['all_manifolds'].size
        
        H_eye = np.eye(H_size)
        
        mu_shape = (H_size**2,H_size**2,3)
        new_mu = np.zeros(mu_shape,dtype='complex')
        for i in range(3):
            if ket_flag:
                mu_i = np.kron(mu[:,:,i],H_eye.T)
            else:
                mu_i = np.kron(H_eye,mu[:,:,i].T)
                
            new_mu[:,:,i] = mu_i

        if np.allclose(np.imag(new_mu),0):
            new_mu = np.real(new_mu)
            
        return mu_key, new_mu

    def append_mu_by_manifold(self,old_manifold,change,ket_flag):
        if ket_flag:
            f = self.make_mu_by_manifold_ket
        else:
            f = self.make_mu_by_manifold_bra
        key, mu = f(old_manifold,change)
        if key == None:
            pass
        else:
            self.mu_L_basis[key] = mu

    def set_mu_Liouville_space_unseparable_manifolds(self):
        self.mu_L_basis = dict()
        changes = [-1,1]
        for change,ket_flag in itertools.product(changes,[True,False]):
            mu_key, mu = self.make_mu_unseparable_manifolds(change,ket_flag)
            self.mu_L_basis[mu_key] = mu

    def set_mu_Liouville_space_separable_manifolds(self):
        self.mu_L_basis = dict()
        for i_key in self.manifolds:
            for j_key in self.manifolds:
                manifold = (int(i_key),int(j_key))
                self.append_mu_by_manifold(manifold,1,True)
                self.append_mu_by_manifold(manifold,-1,True)
                self.append_mu_by_manifold(manifold,1,False)
                self.append_mu_by_manifold(manifold,-1,False)

    def set_mu_Liouville_space(self):
        if self.manifolds[0] == 'all_manifolds':
            self.set_mu_Liouville_space_unseparable_manifolds()
        else:
            self.set_mu_Liouville_space_separable_manifolds()
                
    def save_mu(self):
        np.savez(os.path.join(self.save_path,'mu_original_L_basis.npz'),**self.mu_L_basis)



class SecularRedfieldConstructor:

    def __init__(self,folder,*,conserve_memory = False,do_nothing=False):
        self.base_path = folder
        self.load_path = os.path.join(self.base_path,'closed')
        self.save_path = os.path.join(self.base_path,'open')
        os.makedirs(self.save_path,exist_ok=True)
        with open(os.path.join(folder,'params.yaml')) as yaml_stream:
            self.params = yaml.load(yaml_stream,Loader=yaml.SafeLoader)

        self.load_eigensystem()
        self.load_mu()

        if self.manifolds[0] == 'all_manifolds':
            separable_manifolds = False
        else:
            separable_manifolds = True
            
        self.PolyVib = PolymerVibrations(os.path.join(folder,'params.yaml'),
                                         separable_manifolds=separable_manifolds)

        self.set_spectral_densities()

        try:
            self.secular = self.params['bath']['secular']
        except KeyError:
            self.secular = False
            
        if do_nothing:
            pass
        else: 
            self.set_Y()
            self.set_L()
            self.save_L()
 
            if not conserve_memory:
                self.load_mu()

                self.set_mu_Liouville_space()
                self.save_mu()

    def set_spectral_densities(self):
        self.SD = {}
        self.SD['site_bath'] = self.make_spectral_density(self.params['bath']['site_bath'])
        self.SD['vibration_bath'] = self.make_spectral_density(self.params['bath']['site_bath'])
        try:
            self.SD['site_internal_conversion_bath'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath'])
            if self.PolyVib.separable_manifolds == True:
                raise Exception('Cannot model site internal conversion processes when input Hamiltonian has been divided into separate manifolds')

        except KeyError:
            pass

    def make_spectral_density(self,bath_dict):
        kT = bath_dict['temperature']
        spectrum_type = bath_dict['spectrum_type']

        if spectrum_type == 'ohmic':
            lam = bath_dict['coupling']
            gam = bath_dict['cutoff_frequency']
            cutoff = bath_dict['cutoff_function']
            SD = OhmicSpectralDensity(lam,gam,kT,cutoff=cutoff)
        elif spectrum_type == 'white-noise':
            deph_rate = bath_dict['dephasing_rate']
            relax_rate = bath_dict['relaxation_rate']
            SD = WhiteNoiseSpectralDensity(deph_rate,relax_rate,kT)
        else:
            raise Exception('spectrum_type not supported')

        return SD

    def load_eigensystem(self):
        """Load eigenvalues and eigenvectors of Hamiltonian
"""
        e_save_name = os.path.join(self.load_path,'eigenvalues.npz')
        v_save_name = os.path.join(self.load_path,'eigenvectors.npz')
        with np.load(e_save_name) as e_arch:
            self.manifolds = list(e_arch.keys())
            self.eigenvalues = {key:e_arch[key] for key in self.manifolds}
        with np.load(v_save_name) as v_arch:
            self.eigenvectors = {key:v_arch[key] for key in self.manifolds}

        return None

    def load_mu(self):
        mu_save_name = os.path.join(self.load_path,'mu.npz')
        with np.load(mu_save_name) as mu_arch:
            self.mu_keys = list(mu_arch.keys())
            self.mu = {key:mu_arch[key] for key in self.mu_keys}

    def make_dissipation_tensor(self,CO,manifold_key,SD):
        """Make a Redfield-style dissipation tensor using the eigenbasis
            decomposition of the system-bath coupling operator (CO)
        Args:
            CO (np.ndarray): system-bath coupling operator
            manifold_key (str): string representing the manifold
            SD (callable): spectral density function of the bath
        Returns:
            np.ndarray: dissipation tensor, from which R may be made
"""
        e = self.eigenvalues[manifold_key]
        v = self.eigenvectors[manifold_key]

        CO = np.conjugate(v.T).dot(CO).dot(v)

        CO_dagger = np.conjugate(CO.T)

        eki = e[np.newaxis,:] - e[:,np.newaxis]

        cki = np.zeros(eki.shape)
        for i in range(e.size):
            for k in range(e.size):
                cki[i,k] = SD(eki[i,k])

        Yiikk = CO * CO_dagger * cki

        CO_diag = CO.diagonal()

        Yijij = CO_diag[:,np.newaxis]*CO_diag[np.newaxis,:] * SD(0)

        Yijji = CO * CO * cki

        return Yiikk, Yijij, Yijji

    def make_Y1(self,manifold_key):
        """Creates vibrational dissipation tensor, based upon a bilinear
            coupling of the explicit vibrations to the bath
        Args:
            manifold_key (str): can be 'all_manifolds' or '0','1','2',...
"""
        spectral_density = self.SD['vibration_bath']
        size = self.eigenvalues[manifold_key].size
        Y1_iikk = np.zeros((size,size))
        Y1_ijij = np.zeros((size,size))
        Y1_ijji = np.zeros((size,size))
        if manifold_key == 'all_manifolds':
            pass
        else:
            manifold_num = int(manifold_key)

        for p in range(self.PolyVib.num_vibrations):
            # creation operator
            ad_p = self.PolyVib.total_ups[p]
            if manifold_key == 'all_manifolds':
                x_p = (ad_p + ad_p.T)/np.sqrt(2)
            else:
                ad_p_man = self.PolyVib.extract_vibronic_manifold(ad_p,manifold_num)
                x_p = (ad_p_man + ad_p_man.T)/np.sqrt(2)
            a,b,c = self.make_dissipation_tensor(x_p,manifold_key,spectral_density)
            Y1_iikk += a
            Y1_ijij += b
            Y1_ijji += c

        
        return Y1_iikk, Y1_ijij, Y1_ijji

    def make_Y2(self,manifold_key):
        """Creates an electronic dissipation tensor, based upon 
            uncorrelated baths associated with each diabatic excited state
        Args:
            manifold_key (str): can be 'all_manifolds' or '0','1','2',...
"""
        spectral_density = self.SD['site_bath']
        size = self.eigenvalues[manifold_key].size
        Y2_iikk = np.zeros((size,size))
        Y2_ijij = np.zeros((size,size))
        Y2_ijji = np.zeros((size,size))
 
        for n in range(self.PolyVib.Polymer.num_sites):
            # site projector operators
            P_n = self.PolyVib.Polymer.occupied_list[n]
            P_n = self.extract_electronic_operator(P_n,manifold_key)

            P_n = np.kron(P_n,self.PolyVib.vibrational_identity)
            a,b,c = self.make_dissipation_tensor(P_n,manifold_key,spectral_density)
            Y2_iikk += a
            Y2_ijij += b
            Y2_ijji += c

        return Y2_iikk, Y2_ijij, Y2_ijji

    def extract_electronic_operator(self,O,manifold_key):
        if manifold_key == 'all_manifolds':
            O = self.PolyVib.Polymer.extract_electronic_subspace(O,0,self.PolyVib.maximum_manifold)
        else:
            manifold_num = int(manifold_key)
            O = self.PolyVib.Polymer.extract_manifold(O,manifold_num)

        return O

    def make_Y3(self):
        """Creates non-adiabatic dissipation tensor by using the sigma_x
            Pauli spin matrix as the coupling operator for each 2LS
"""
        spectral_density = self.SD['site_internal_conversion_bath']
        manifold_key = 'all_manifolds'
        size = self.eigenvalues[manifold_key].size
        
        Y3_iikk = np.zeros((size,size))
        Y3_ijij = np.zeros((size,size))
        Y3_ijji = np.zeros((size,size))
        
        for n in range(self.PolyVib.Polymer.num_sites):
            # electronic excitation creation operator
            up_n = self.PolyVib.Polymer.up_list[n]
            up_n = self.extract_electronic_operator(up_n,manifold_key)
            up_n = np.kron(up_n,self.PolyVib.vibrational_identity)
            sigma_x_n = up_n + up_n.T

            a,b,c = self.make_dissipation_tensor(sigma_x_n,manifold_key,spectral_density)
            Y3_iikk += a
            Y3_ijij += b
            Y3_ijji += c
                
        return Y3_iikk, Y3_ijij, Y3_ijji

    def make_R_from_Y(self,Yiikk,Yijij,Yijji):
        """Given the dissipation operator Y, constructs the Redfield tensor
        Args:
            Y (np.ndarray): 4-index array dissipation tensor
            manifold_key (str): can be 'all_manifolds' or '0','1','2',...
"""
        Riikk = -2*np.real(Yiikk) + np.eye(Yiikk.shape[0]) * np.sum(2*np.real(Yijji),axis=0)
        Rijij = -(Yijij + np.conjugate(Yijij.T)) + np.sum(Yijji,axis=0)[:,np.newaxis] + np.sum(np.conjugate(Yijji),axis=0)[np.newaxis,:]

        if not np.allclose(Riikk.diagonal(),Rijij.diagonal()):
            warnings.warn('Two methods of making Riiii yield different answers')

        R = sp.diags(Rijij.flatten(),format='lil')

        H_size = Riikk.shape[0]
        pop_inds = np.arange(H_size)*(H_size+1)

        for i in range(H_size):
            R[pop_inds[i],pop_inds] = Riikk[i,:]

        return R.tocsr()

    def make_R_from_Y_coherence(self,Yket_ijji,Ybra_ijji,ket_manifold_key,bra_manifold_key):
        """Makes Redfield tensor for optical coherence states, in the case
            that each electronic excitation manifold is separable.
        Args:
            Yket (np.ndarray): 4-index array dissipation tensor for the ket
                excitation manifold
            Ybra (np.ndarray): 4-index array dissipation tensor for the bra
                excitation manifold
            ket_manifold_key (str): bra manifold, can be '0','1','2',...
            bra_manifold_key (str): ket manifold, can be '0','1','2',...
"""
        if ket_manifold_key == bra_manifold_key:
            raise Exception('You must use make_R_from_Y for ket and bra manifold keys equal to each other')

        Rijij = np.sum(Yket_ijji,axis=0)[:,np.newaxis] + np.sum(np.conjugate(Ybra_ijji),axis=0)[np.newaxis,:]

        R = sp.diags(Rijij.flatten(),format='csr')
        
        return R

    def set_L(self):
        self.L = {}
        if self.manifolds[0] == 'all_manifolds':
            L_key = 'all_manifolds'
            R = self.make_flat_R(L_key,L_key)
            U = self.make_U(L_key,L_key)
            self.L[L_key] = U - R
        else:
            for ket_man,bra_man in itertools.product(self.manifolds,
                                                     self.manifolds):
                L_key = ket_man + bra_man
                R = self.make_flat_R(ket_man,bra_man)
                U = self.make_U(ket_man,bra_man)
                self.L[L_key] = U - R

    def save_L(self):
        np.savez(os.path.join(self.save_path,'L.npz'),**self.L)

    def make_flat_R(self,ket_manifold,bra_manifold):
        if ket_manifold == 'all_manifolds':
            Ys = self.Y['all_manifolds']
            R = self.make_R_from_Y(*Ys)
        elif ket_manifold == bra_manifold:
            Ys = self.Y[ket_manifold]
            R = self.make_R_from_Y(*Ys)
        elif ket_manifold != bra_manifold:
            a,b,Yket_ijji = self.Y[ket_manifold]
            a,b,Ybra_ijji = self.Y[bra_manifold]
            R = self.make_R_from_Y_coherence(Yket_ijji,Ybra_ijji,ket_manifold,bra_manifold)
            
        return R

    def set_Y(self):
        self.Y = {}
        for manifold in self.manifolds:
            Y1_iikk, Y1_ijij, Y1_ijji = self.make_Y1(manifold)
            Y2_iikk, Y2_ijij, Y2_ijji = self.make_Y2(manifold)
            
            self.Y[manifold] = (Y1_iikk + Y2_iikk,
                                Y1_ijij + Y2_ijij,
                                Y1_ijji + Y2_ijji)
            if manifold == 'all_manifolds':
                Y3_iikk, Y3_ijij, Y3_ijji = self.make_Y3()
                self.Y[manifold] = (Y1_iikk + Y2_iikk + Y3_iikk,
                                    Y1_ijij + Y2_ijij + Y3_ijij,
                                    Y1_ijji + Y2_ijji + Y3_ijji)

    def make_U(self,ket_manifold,bra_manifold):
        e_ket = self.eigenvalues[ket_manifold]
        e_bra = self.eigenvalues[bra_manifold]
        
        U_vec = -1j * (np.kron(e_ket,np.ones(e_bra.size)) - np.kron(np.ones(e_ket.size),e_bra))
        U = sp.diags(U_vec,format='csr')
        return U

    def mu_key_to_manifold_keys(self,key):
        if self.manifolds[0] == 'all_manifolds':
            starting_key = 'all_manifolds'
            ending_key = 'all_manifolds'
        else:
            starting_key, ending_key = key.split('_to_')
        return starting_key, ending_key


    def make_mu_by_manifold_ket(self,old_manifold,change):
        i,j = old_manifold
        i2 = i + change
        j2 = j
        
        if i2 >= 0 and i2 <= self.PolyVib.maximum_manifold:
            pass
        else:
            return None, None

        if i2 > i:
            mu_key = str(i) + '_to_' + str(i2)
            mu = self.mu[mu_key]
        else:
            mu_key = str(i2) + '_to_' + str(i)
            mu = self.mu[mu_key].transpose(1,0,2)

        j_size = self.eigenvalues[str(j)].size
        i_size = self.eigenvalues[str(i)].size
        i2_size = self.eigenvalues[str(i2)].size
        
        bra_eye = np.eye(j_size)
        old_key = str(i) + str(j)
        new_key = str(i2) + str(j2)
        
        mu_shape = (i2_size*j_size,i_size*j_size,3)
        new_mu = np.zeros(mu_shape,dtype='complex')
        for i in range(3):
            mu_i = np.kron(mu[:,:,i],bra_eye)
            new_mu[:,:,i] = mu_i

        if np.allclose(np.imag(new_mu),0):
            new_mu = np.real(new_mu)
        mu_key = old_key + '_to_' + new_key
        return mu_key, new_mu

    def make_mu_by_manifold_bra(self,old_manifold,change):
        i,j = old_manifold
        i2 = i
        j2 = j + change
        
        if j2 >= 0 and j2 <= self.PolyVib.maximum_manifold:
            pass
        else:
            return None, None

        if j2 > j:
            mu_key = str(j) + '_to_' + str(j2)
            mu = self.mu[mu_key].transpose(1,0,2)
        else:
            mu_key = str(j2) + '_to_' + str(j)
            mu = self.mu[mu_key]

        i_size = self.eigenvalues[str(i)].size
        j_size = self.eigenvalues[str(j)].size
        j2_size = self.eigenvalues[str(j2)].size
        
        ket_eye = np.eye(i_size)
        old_key = str(i) + str(j)
        new_key = str(i2) + str(j2)
        
        mu_shape = (i_size*j2_size,i_size*j_size,3)
        new_mu = np.zeros(mu_shape,dtype='complex')
        for i in range(3):
            mu_i = np.kron(ket_eye,mu[:,:,i].T)
            new_mu[:,:,i] = mu_i

        if np.allclose(np.imag(new_mu),0):
            new_mu = np.real(new_mu)
        mu_key = old_key + '_to_' + new_key
        return mu_key, new_mu

    def make_mu_unseparable_manifolds(self,change,ket_flag):
        mu = self.mu['up']
        if ket_flag:
            mu_key = 'ket'
        else:
            mu = mu.transpose(1,0,2)
            mu_key = 'bra'
        if change == 1:
            mu_key += '_up'
        elif change == -1:
            mu = mu.transpose(1,0,2)
            mu_key += '_down'
        else:
            raise Exception('change must be either +1 or -1')
        H_size = self.eigenvalues['all_manifolds'].size
        
        H_eye = np.eye(H_size)
        
        mu_shape = (H_size**2,H_size**2,3)
        new_mu = np.zeros(mu_shape,dtype='complex')
        for i in range(3):
            if ket_flag:
                mu_i = np.kron(mu[:,:,i],H_eye.T)
            else:
                mu_i = np.kron(H_eye,mu[:,:,i].T)
                
            new_mu[:,:,i] = mu_i

        if np.allclose(np.imag(new_mu),0):
            new_mu = np.real(new_mu)
            
        return mu_key, new_mu

    def append_mu_by_manifold(self,old_manifold,change,ket_flag):
        if ket_flag:
            f = self.make_mu_by_manifold_ket
        else:
            f = self.make_mu_by_manifold_bra
        key, mu = f(old_manifold,change)
        if key == None:
            pass
        else:
            self.mu_L_basis[key] = mu

    def set_mu_Liouville_space_unseparable_manifolds(self):
        self.mu_L_basis = dict()
        changes = [-1,1]
        for change,ket_flag in itertools.product(changes,[True,False]):
            mu_key, mu = self.make_mu_unseparable_manifolds(change,ket_flag)
            self.mu_L_basis[mu_key] = mu

    def set_mu_Liouville_space_separable_manifolds(self):
        self.mu_L_basis = dict()
        for i_key in self.manifolds:
            for j_key in self.manifolds:
                manifold = (int(i_key),int(j_key))
                self.append_mu_by_manifold(manifold,1,True)
                self.append_mu_by_manifold(manifold,-1,True)
                self.append_mu_by_manifold(manifold,1,False)
                self.append_mu_by_manifold(manifold,-1,False)

    def set_mu_Liouville_space(self):
        if self.manifolds[0] == 'all_manifolds':
            self.set_mu_Liouville_space_unseparable_manifolds()
        else:
            self.set_mu_Liouville_space_separable_manifolds()
                
    def save_mu(self):
        np.savez(os.path.join(self.save_path,'mu_original_L_basis.npz'),**self.mu_L_basis)

        

class DiagonalizeLiouvillian:
    def __init__(self,folder,*,conserve_memory = False,secular=False):
        self.secular = secular
        self.base_path = folder
        self.load_path = os.path.join(self.base_path,'open')
        self.save_path = self.load_path
        self.prune = True
        self.load_L()
        self.diagonalize_L()
        self.save_L_eigensystem()
        self.save_timing()

        if not conserve_memory:
            self.load_mu()
            self.transform_mu()
            self.save_mu()

    def save_timing(self):
        save_dict = {'L_diagonalization_time':self.L_diagonalization_time}
        np.savez(os.path.join(self.save_path,'Liouvillian_diagonalizaation_time.npz'),**save_dict)
        
    def load_L(self):
        """Load Liouvillian
"""
        L_save_name = os.path.join(self.load_path,'L.npz')
        with np.load(L_save_name,allow_pickle=True) as L_arch:
            self.L_keys = list(L_arch.keys())
            self.L = {key:L_arch[key] for key in self.L_keys}

    def load_mu(self):
        """Load dipole operator
"""
        mu_save_name = os.path.join(self.load_path,'mu_original_L_basis.npz')
        with np.load(mu_save_name) as mu_arch:
            self.mu_keys = list(mu_arch.keys())
            self.mu = {key:mu_arch[key] for key in self.mu_keys}

    def eig(self,L,*,check_eigenvectors = True,populations_only = False):
        
        eigvals, eigvecs_left, eigvecs = eig(L,left=True,right=True)

        eigvecs_left = np.conjugate(eigvecs_left.T)
        
        for i in range(eigvals.size):
            max_index = np.argmax(np.abs(eigvecs[:,i]))
            if np.real(eigvecs[max_index,i]) < 0:
                eigvecs[:,i] *= -1
                
            if np.isclose(eigvals[i],0,atol=1E-12):
                # eigenvalues of 0 correspond to thermal distributions,
                # which should have unit trace in the Hamiltonian space
                if populations_only:
                    trace_norm = eigvecs[:,i].sum()
                    eigvecs[:,i] = eigvecs[:,i] / trace_norm
                else:
                    shape = int(np.sqrt(eigvals.size))
                    trace_norm = eigvecs[:,i].reshape(shape,shape).trace()
                    eigvecs[:,i] = eigvecs[:,i] / trace_norm

        for i in range(eigvals.size):
            norm = np.dot(eigvecs_left[i,:],eigvecs[:,i])
            eigvecs_left[i,:] *= 1/norm

        if check_eigenvectors:
            LV = L.dot(eigvecs)
            D = eigvecs_left.dot(LV)
            if np.allclose(D,np.diag(eigvals),rtol=1E-10,atol=1E-10):
                pass
            else:
                warnings.warn('Using eigenvectors to diagonalize Liouvillian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(np.max(np.abs(D - np.diag(eigvals)))))

        return eigvals, eigvecs, eigvecs_left

    def eig2(self,L_key,*,check_eigenvectors = True):
        if L_key == 'all_manifolds':
            ket_key = 'all_manifolds'
            bra_key = 'all_manifolds'
        else:
            ket_key,bra_key = L_key
        L = self.L[L_key]
        if L.dtype == np.dtype('O'):
            L = L[()]
        E = L.diagonal()
        V = sp.eye(E.size,format='lil',dtype='complex')
        VL = sp.eye(E.size,format='lil',dtype='complex')
        
        if ket_key == bra_key:
            H_size = int(np.sqrt(E.size))
            pop_inds = np.arange(H_size)*(H_size+1)
            L_pop = np.zeros((H_size,H_size),dtype='complex')
            for i in range(H_size):
                L_pop[i,:] = L[pop_inds[i],pop_inds].toarray()
            e, v, vl = self.eig(L_pop,populations_only=True)
            E[pop_inds] = e[:]
            for i,j in zip(pop_inds,range(len(pop_inds))):
                
                V[pop_inds,i] = v[:,j]
                
                VL[pop_inds,i] = vl[:,j]

        if check_eigenvectors:
            LV = L.dot(V)
            D = VL.dot(LV)
            check = D - sp.diags(E)
            if check.data.size > 0:
                max_check_value = np.max(np.abs(check.data))
                if max_check_value > 1E-10:
                    warnings.warn('Using eigenvectors to diagonalize Liouvillian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(max_check_value))

        return E,V.tocsr(),VL.tocsr()

    def diagonalize_L(self):
        ta = time.time()
        self.L_eigenvalues = {}
        self.L_right_eigenvectors = {}
        self.L_left_eigenvectors = {}
        for L_key in self.L.keys():
            L = self.L[L_key]
            if self.secular:
                e, ev, evl = self.eig2(L_key)
            else:
                e, ev, evl = self.eig(L)
            self.L_eigenvalues[L_key] = e
            self.L_right_eigenvectors[L_key] = ev
            self.L_left_eigenvectors[L_key] = evl
        tb = time.time()
        self.L_diagonalization_time = tb-ta

    def save_L_eigensystem(self):
        np.savez(os.path.join(self.save_path,'right_eigenvectors.npz'),**self.L_right_eigenvectors)
        np.savez(os.path.join(self.save_path,'left_eigenvectors.npz'),**self.L_left_eigenvectors)
        np.savez(os.path.join(self.save_path,'eigenvalues.npz'),**self.L_eigenvalues)

    def transform_mu(self):
        self.transformed_mu = dict()
        if self.prune:
            self.boolean_mu = dict()
        for key in self.mu_keys:
            mu = self.mu[key]
            if 'all_manifolds' in self.L_keys:
                old_L_key = 'all_manifolds'
                new_L_key = 'all_manifolds'
            else:
                old_L_key,new_L_key = key.split('_to_')
            vl = self.L_left_eigenvectors[new_L_key]
            if sp.issparse(vl):
                vl = vl.toarray()
            vr = self.L_right_eigenvectors[old_L_key]
            if sp.issparse(vr):
                vr = vr.toarray()
            new_mu = np.zeros(mu.shape,dtype='complex')
            for i in range(3):
                new_mu[:,:,i] = vl.dot(mu[:,:,i]).dot(vr)
            self.transformed_mu[key] = new_mu

            if self.prune:
                boolean_mu = np.zeros(mu.shape[:2],dtype='bool')
                boolean_mu[:,:] = np.round(np.sum(np.abs(new_mu)**2,axis=-1),12)
                mu = mu * boolean_mu[:,:,np.newaxis]
                self.boolean_mu[key] = boolean_mu

    def save_mu(self):
        if self.prune:
            np.savez(os.path.join(self.save_path,'mu_pruned.npz'),**self.transformed_mu)
            np.savez(os.path.join(self.save_path,'mu_boolean.npz'),**self.boolean_mu)
        else:
            np.savez(os.path.join(self.save_path,'mu.npz'),**self.transformed_mu)
