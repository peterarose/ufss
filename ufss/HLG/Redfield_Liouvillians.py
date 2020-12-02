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

from .Hamiltonians import DisplacedAnharmonicOscillator, PolymerVibrations, Polymer, DiagonalizeHamiltonian, LadderOperators
from .params_converter import convert

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
        
        Y2_iikk = np.zeros((size,size))
        Y2_ijij = np.zeros((size,size))
        Y2_ijji = np.zeros((size,size))
        
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
            vr = self.L_right_eigenvectors[old_L_key]
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

            
class LindbladConstructor:

    @staticmethod
    def make_Lindblad_instructions(gamma,O):
        """O must be square
"""
        II = np.eye(O.shape[0])
        Od = np.conjugate(O.T)
        leftright = gamma * (-np.dot(Od,O)/2)
        return [(gamma*O,Od),(leftright,II),(II,leftright)]

    @staticmethod
    def make_Lindblad_instructions2(gamma,Oket,Obra):
        IIket = np.eye(Oket.shape[0])
        IIbra = np.eye(Obra.shape[0])
        Oketd = np.conjugate(Oket.T)
        Obrad = np.conjugate(Obra.T)
        left = gamma * (-np.dot(Oketd,Oket)/2)
        right = gamma * (-np.dot(Obrad,Obra)/2)
        return [(gamma*Oket,Obrad),(left,IIbra),(IIket,right)]

    @staticmethod
    def make_Lindblad_instructions2_Obra0(gamma,Oket,Obra):
        IIbra = np.eye(Obra.shape[0])
        Oketd = np.conjugate(Oket.T)
        left = gamma * (-np.dot(Oketd,Oket)/2)
        return [(left,IIbra)]

    @staticmethod
    def make_Lindblad_instructions2_Oket0(gamma,Oket,Obra):
        IIket = np.eye(Oket.shape[0])
        Obrad = np.conjugate(Obra.T)
        right = gamma * (-np.dot(Obrad,Obra)/2)
        return [(IIket,right)]

class LiouvillianConstructor(LindbladConstructor):

    @staticmethod
    def make_commutator_instructions(O):
        """O must be square
"""
        II = np.eye(O.shape[0])
        return [(O,II),(II,-O)]

    @staticmethod
    def make_commutator_instructions2(Oket,Obra):
        """
"""
        IIket = np.eye(Oket.shape[0])
        IIbra = np.eye(Obra.shape[0])
        return [(Oket,IIbra),(IIket,-Obra)]

    @staticmethod
    def make_Liouvillian(instruction_list,*,sparse=False):
        if sparse:
            kron = sp.kron
        else:
            kron = np.kron
        left, right = instruction_list[0]
        L = kron(left,right.T)
        for left,right in instruction_list[1:]:
            L = L + kron(left,right.T)
        return L

class OpenPolymer(Polymer,LiouvillianConstructor):

    def __init__(self,site_energies,site_couplings,dipoles):
        """Extends Polymer object to an open systems framework, 
        using the Lindblad formalism to describe bath coupling
"""
        super().__init__(site_energies,site_couplings,dipoles)

        # Values that need to be set
        self.optical_dephasing_gamma = 0
        self.optical_relaxation_gamma = 0
        self.site_to_site_dephasing_gamma = 0
        self.site_to_site_relaxation_gamma = 0
        self.exciton_relaxation_gamma = 0
        self.exciton_exciton_dephasing_gamma = 0
        self.kT = 0
        
    def optical_dephasing_operator(self):
        total_deph = self.occupied_list[0].copy()
        for i in range(1,len(self.occupied_list)):
            total_deph += self.occupied_list[i]
        return total_deph

    def optical_dephasing_instructions(self):
        O = self.optical_dephasing_operator()
        gamma = self.optical_dephasing_gamma
        return self.make_Lindblad_instructions(gamma,O)

    def optical_dephasing_Liouvillian(self):
        instructions = self.optical_dephasing_instructions()
        return self.make_Liouvillian(instructions)

    def boltzmann_factors(self,E1,E2):
        if E1 == E2:
            return 0.5,0.5

        if E1 < E2:
            return self.boltzmann_factors_ordered_inputs(E1,E2)
        else:
            E1_to_E2, E2_to_E1 = self.boltzmann_factors_ordered_inputs(E2,E1)
            return E2_to_E1, E1_to_E2

    def boltzmann_factors_ordered_inputs(self,E1,E2):
        """E1 must be less than E2"""
        if self.kT == 0:
            return 1, 0
        Z = np.exp(-E1/self.kT) + np.exp(-E2/self.kT)
        if np.isclose(Z,0):
            E2_to_E1 = 1
            E1_to_E2 = 0
        else:
            E2_to_E1 = np.exp(-E1/self.kT)/Z
            E1_to_E2 = np.exp(-E2/self.kT)/Z
        return E2_to_E1, E1_to_E2

    def optical_relaxation_instructions(self):
        eg = 0
        ins_list = []
        gamma = self.optical_relaxation_gamma
        for n in range(len(self.energies)):
            en = self.energies[n]
            bg, bn = self.boltzmann_factors(eg,en)
            O = self.up_list[n]
            instructions2 = self.make_Lindblad_instructions(gamma * bg,O.T)
            ins_list += instructions2
            if np.isclose(bn,0):
                pass
            else:
                instructions1 = self.make_Lindblad_instructions(gamma * bn,O)
                ins_list += instructions1

        return ins_list

    def optical_relaxation_Liouvillian(self):
        inst_list = self.optical_relaxation_instructions()
        L = self.make_Liouvillian(inst_list)
        return L

    def site_to_site_relaxation_instructions(self):
        nm = itertools.combinations(range(len(self.energies)),2)
        i = 0
        ins_list = []
        gamma = self.site_to_site_relaxation_gamma
        for n,m in nm:
            en = self.energies[n]
            em = self.energies[m]
            bn,bm = self.boltzmann_factors(en,em)
            O = self.exchange_list[i]
            instructions1 = self.make_Lindblad_instructions(gamma * bn,O)
            instructions2 = self.make_Lindblad_instructions(gamma * bm,O.T)
            ins_list += instructions1
            ins_list += instructions2
            i+=1

        return ins_list

    def site_to_site_relaxation_Liouvillian(self):
        inst_list = self.site_to_site_relaxation_instructions()
        L = self.make_Liouvillian(inst_list)
        return L

    def site_to_site_dephasing_operator_list(self):
        s_deph_list = []
        for (i,j) in itertools.combinations(range(self.num_sites),2):
            s_deph_list.append(self.occupied_list[i] - self.occupied_list[j])
        return s_deph_list

    def all_site_dephasing_instructions(self):
        s_deph_list = self.site_to_site_dephasing_operator_list()
        Lindblad_instruction_list = []
        gamma = self.site_to_site_dephasing_gamma
        for O in s_deph_list:
            Lindblad_instruction_list += self.make_Lindblad_instructions(gamma,O)
        return Lindblad_instruction_list

    def all_site_dephasing_Liouvillian(self):
        inst_list = self.all_site_dephasing_instructions()
        L = self.make_Liouvillian(inst_list)
        return L/(2*self.num_sites)

    def set_electronic_dissipation_instructions(self):
        inst_list = []
        
        if self.optical_dephasing_gamma != 0:
            inst_list += self.optical_dephasing_instructions()

        if self.site_to_site_dephasing_gamma != 0:
            inst_list += self.all_site_dephasing_instructions()
            
        if self.site_to_site_relaxation_gamma != 0:
            inst_list += self.site_to_site_relaxation_instructions()
            
        if self.optical_relaxation_gamma != 0:
            inst_list += self.optical_relaxation_instructions()
            
        self.electronic_dissipation_instructions = inst_list

    def make_manifold_hamiltonian_instructions(self,ket_manifold,bra_manifold):
        Hket = self.get_electronic_hamiltonian(manifold_num = ket_manifold)
        Hbra = self.get_electronic_hamiltonian(manifold_num = bra_manifold)
        return self.make_commutator_instructions2(-1j*Hket,-1j*Hbra)

    def make_total_Liouvillian(self):
        drho = self.make_Liouvillian(self.make_manifold_hamiltonian_instructions('all','all'))
        if self.num_sites > 1:
            drho += self.all_exciton_dephasing_Liouvillian()
            drho += self.exciton_relaxation_Liouvillian()
        # drho += self.optical_relaxation_Liouvillian()
        drho += self.optical_dephasing_Liouvillian()
        
        self.L = drho

    def eigfun(self,L,*,check_eigenvectors = True,invert = True,populations_only = False):
        eigvals, eigvecs = np.linalg.eig(L)

        eigvals = np.round(eigvals,12)
        sort_indices = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:,sort_indices]
        for i in range(eigvals.size):
            max_index = np.argmax(np.abs(eigvecs[:,i]))
            if np.real(eigvecs[max_index,i]) < 0:
                eigvecs[:,i] *= -1
            if eigvals[i] == 0:
                # eigenvalues of 0 correspond to thermal distributions,
                # which should have unit trace in the Hamiltonian space
                if populations_only:
                    trace_norm = eigvecs[:,i].sum()
                    eigvecs[:,i] = eigvecs[:,i] / trace_norm
                else:
                    shape = int(np.sqrt(eigvals.size))
                    trace_norm = eigvecs[:,i].reshape(shape,shape).trace()
                    if np.isclose(trace_norm,0):
                        pass
                    else:
                        eigvecs[:,i] = eigvecs[:,i] / trace_norm

        if invert:
            eigvecs_left = np.linalg.pinv(eigvecs)
        else:
            eigvals_left, eigvecs_left = np.linalg.eig(L.T)

            eigvals_left = np.round(eigvals_left,12)
            sort_indices_left = eigvals_left.argsort()
            eigvals_left.sort()
            eigvecs_left = eigvecs_left[:,sort_indices_left]
            eigvecs_left = eigvecs_left.T
            for i in range(eigvals_left.size):
                    norm = np.dot(eigvecs_left[i,:],eigvecs[:,i])
                    eigvecs_left[i,:] *= 1/norm

        if check_eigenvectors:
            LV = L.dot(eigvecs)
            D = eigvecs_left.dot(LV)
            if np.allclose(D,np.diag(eigvals),rtol=1E-10,atol=1E-10):
                pass
            else:
                warnings.warn('Using eigenvectors to diagonalize Liouvillian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(np.max(np.abs(D - np.diag(eigvals)))))

        self.eigenvalues = eigvals
        self.eigenvectors = {'left':eigvecs_left,'right':eigvecs}

        return eigvals, eigvecs, eigvecs_left

    def save_L(self,dirname):
        save_npz(os.path.join(dirname,'L.npz'),csr_matrix(self.L))

    def save_L_by_manifold(self):
        np.savez(os.path.join(self.base_path,'L.npz'),**self.L_by_manifold)

    def save_eigsystem(self,dirname):
        np.savez(os.path.join(dirname,'right_eigenvectors.npz'),all_manifolds = self.eigenvectors['right'])
        np.savez(os.path.join(dirname,'left_eigenvectors.npz'),all_manifolds = self.eigenvectors['left'])
        np.savez(os.path.join(dirname,'eigenvalues.npz'),all_manifolds = self.eigenvalues)

    def save_mu(self,dirname,*,mask=True):
        evl = self.eigenvectors['left']
        ev = self.eigenvectors['right']
        
        II = np.eye(self.mu.shape[0])
        mu_ket = np.kron(self.mu,II.T)
        mu_bra = np.kron(II,self.mu.T)

        mu_mask_tol = 10

        mu_ket_t = np.dot(np.dot(evl,mu_ket),ev)
        mu_ket_3d = np.zeros((mu_ket_t.shape[0],mu_ket_t.shape[0],3),dtype='complex')
        mu_ket_3d[:,:,0] = mu_ket_t

        mu_bra_t = np.dot(np.dot(evl,mu_bra),ev)
        mu_bra_3d = np.zeros((mu_bra_t.shape[0],mu_bra_t.shape[0],3),dtype='complex')
        mu_bra_3d[:,:,0] = mu_bra_t

        if mask:
            ket_mask = np.zeros(mu_ket_t.shape,dtype='bool')
            ket_mask[:,:] = np.round(mu_ket_t,mu_mask_tol)[:,:]
            mu_ket_t_masked = mu_ket_t * ket_mask
            mu_ket_3d_masked = np.zeros((mu_ket_t.shape[0],mu_ket_t.shape[0],3),dtype='complex')
            mu_ket_3d_masked[:,:,0] = mu_ket_t_masked

            bra_mask = np.zeros(mu_bra_t.shape,dtype='bool')
            bra_mask[:,:] = np.round(mu_bra_t,mu_mask_tol)[:,:]
            mu_bra_t_masked = mu_bra_t * bra_mask
            mu_bra_3d_masked = np.zeros((mu_ket_t.shape[0],mu_ket_t.shape[0],3),dtype='complex')
            mu_bra_3d_masked[:,:,0] = mu_bra_t_masked

            np.savez(os.path.join(dirname,'mu.npz'),ket=mu_ket_3d,bra=mu_bra_3d)
            np.savez(os.path.join(dirname,'eigenvalues.npz'),all_manifolds=self.eigenvalues)
            np.savez(os.path.join(dirname,'right_eigenvectors.npz'),all_manifolds=ev)
            np.savez(os.path.join(dirname,'left_eigenvectors.npz'),all_manifolds=evl)
            np.savez(os.path.join(dirname,'mu_boolean.npz'),ket=ket_mask,bra=bra_mask)
            np.savez(os.path.join(dirname,'mu_pruned.npz'),ket=mu_ket_3d_masked,bra=mu_bra_3d_masked)

        else:
            np.savez(os.path.join(dirname,'mu.npz'),ket=mu_ket_3d,bra=mu_bra_3d)
            np.savez(os.path.join(dirname,'eigenvalues.npz'),all_manifolds=self.eigenvalues)
            np.savez(os.path.join(dirname,'right_eigenvectors.npz'),all_manifolds=ev)
            np.savez(os.path.join(dirname,'left_eigenvectors.npz'),all_manifolds=evl)

    def save_RWA_mu(self,dirname,*,mask=True):
        evl = self.eigenvectors['left']
        ev = self.eigenvectors['right']
        
        II = np.eye(self.mu_ket_up.shape[0])

        mu_ket_up = np.kron(self.mu_ket_up,II.T)
        mu_ket_down = np.kron(self.mu_ket_up.T,II.T)
        mu_bra_up = np.kron(II,self.mu_ket_up)
        mu_bra_down = np.kron(II,self.mu_ket_up.T)

        mu_mask_tol = 10
        
        mu_ket_up_t = np.dot(np.dot(evl,mu_ket_up),ev)
        mu_ket_up_3d = np.zeros((mu_ket_up_t.shape[0],mu_ket_up_t.shape[0],3),dtype='complex')
        mu_ket_up_3d[:,:,0] = mu_ket_up_t

        mu_bra_up_t = np.dot(np.dot(evl,mu_bra_up),ev)
        mu_bra_up_3d = np.zeros((mu_bra_up_t.shape[0],mu_bra_up_t.shape[0],3),dtype='complex')
        mu_bra_up_3d[:,:,0] = mu_bra_up_t

        mu_ket_down_t = np.dot(np.dot(evl,mu_ket_down),ev)
        mu_ket_down_3d = np.zeros((mu_ket_down_t.shape[0],mu_ket_down_t.shape[0],3),dtype='complex')
        mu_ket_down_3d[:,:,0] = mu_ket_down_t

        mu_bra_down_t = np.dot(np.dot(evl,mu_bra_down),ev)
        mu_bra_down_3d = np.zeros((mu_bra_down_t.shape[0],mu_bra_down_t.shape[0],3),dtype='complex')
        mu_bra_down_3d[:,:,0] = mu_bra_down_t

        if mask:
            ket_up_mask = np.zeros(mu_ket_up_t.shape,dtype='bool')
            ket_up_mask[:,:] = np.round(mu_ket_up_t,mu_mask_tol)[:,:]
            mu_ket_up_t_masked = mu_ket_up_t * ket_up_mask
            mu_ket_up_3d_masked = np.zeros((mu_ket_up_t.shape[0],mu_ket_up_t.shape[0],3),dtype='complex')
            mu_ket_up_3d_masked[:,:,0] = mu_ket_up_t_masked

            bra_up_mask = np.zeros(mu_bra_up_t.shape,dtype='bool')
            bra_up_mask[:,:] = np.round(mu_bra_up_t,mu_mask_tol)[:,:]
            mu_bra_up_t_masked = mu_bra_up_t * bra_up_mask
            mu_bra_up_3d_masked = np.zeros((mu_ket_up_t.shape[0],mu_ket_up_t.shape[0],3),dtype='complex')
            mu_bra_up_3d_masked[:,:,0] = mu_bra_up_t_masked

            ket_down_mask = np.zeros(mu_ket_down_t.shape,dtype='bool')
            ket_down_mask[:,:] = np.round(mu_ket_down_t,mu_mask_tol)[:,:]
            mu_ket_down_t_masked = mu_ket_down_t * ket_down_mask
            mu_ket_down_3d_masked = np.zeros((mu_ket_down_t.shape[0],mu_ket_down_t.shape[0],3),dtype='complex')
            mu_ket_down_3d_masked[:,:,0] = mu_ket_down_t_masked

            bra_down_mask = np.zeros(mu_bra_down_t.shape,dtype='bool')
            bra_down_mask[:,:] = np.round(mu_bra_down_t,mu_mask_tol)[:,:]
            mu_bra_down_t_masked = mu_bra_down_t * bra_down_mask
            mu_bra_down_3d_masked = np.zeros((mu_ket_down_t.shape[0],mu_ket_down_t.shape[0],3),dtype='complex')
            mu_bra_down_3d_masked[:,:,0] = mu_bra_down_t_masked

            np.savez(os.path.join(dirname,'mu.npz'),ket_up=mu_ket_up_3d,bra_up=mu_bra_up_3d,
                     ket_down=mu_ket_down_3d,bra_down=mu_bra_down_3d)
            np.savez(os.path.join(dirname,'eigenvalues.npz'),all_manifolds=self.eigenvalues)
            np.savez(os.path.join(dirname,'right_eigenvectors.npz'),all_manifolds=ev)
            np.savez(os.path.join(dirname,'left_eigenvectors.npz'),all_manifolds=evl)
            np.savez(os.path.join(dirname,'mu_boolean.npz'),ket_up=ket_up_mask,bra_up=bra_up_mask,
                     ket_down=ket_down_mask,bra_down=bra_down_mask)
            np.savez(os.path.join(dirname,'mu_pruned.npz'),ket_up=mu_ket_up_3d_masked,
                     bra_up=mu_bra_up_3d_masked,ket_down=mu_ket_down_3d_masked,
                     bra_down=mu_bra_down_3d_masked)

        else:
            np.savez(os.path.join(dirname,'mu.npz'),ket_up=mu_ket_up_3d,bra_up=mu_bra_up_3d,
                     ket_down=mu_ket_down_3d,bra_down=mu_bra_down_3d)
            np.savez(os.path.join(dirname,'eigenvalues.npz'),all_manifolds=self.eigenvalues)
            np.savez(os.path.join(dirname,'right_eigenvectors.npz'),all_manifolds=ev)
            np.savez(os.path.join(dirname,'left_eigenvectors.npz'),all_manifolds=evl)

    def save_RWA_mu_site_basis(self,dirname):
        
        II = np.eye(self.mu_ket_up.shape[0])
        mu_ket_up = np.kron(self.mu_ket_up,II.T)
        mu_ket_down = np.kron(self.mu_ket_up.T,II.T)
        mu_bra_up = np.kron(II,self.mu_ket_up)
        mu_bra_down = np.kron(II,self.mu_ket_up.T)

        mu_mask_tol = 10
        
        mu_ket_up_3d = np.zeros((mu_ket_up.shape[0],mu_ket_up.shape[0],3),dtype='complex')
        mu_ket_up_3d[:,:,0] = mu_ket_up

        mu_bra_up_3d = np.zeros((mu_bra_up.shape[0],mu_bra_up.shape[0],3),dtype='complex')
        mu_bra_up_3d[:,:,0] = mu_bra_up

        mu_ket_down_3d = np.zeros((mu_ket_down.shape[0],mu_ket_down.shape[0],3),dtype='complex')
        mu_ket_down_3d[:,:,0] = mu_ket_down

        mu_bra_down_3d = np.zeros((mu_bra_down.shape[0],mu_bra_down.shape[0],3),dtype='complex')
        mu_bra_down_3d[:,:,0] = mu_bra_down

        np.savez(os.path.join(dirname,'mu_site_basis.npz'),ket_up=mu_ket_up_3d,bra_up=mu_bra_up_3d,
                     ket_down=mu_ket_down_3d,bra_down=mu_bra_down_3d)



class OpenPolymerVibrations(OpenPolymer):
    def __init__(self,yaml_file,*,mask_by_occupation_num=True,force_detailed_balance=False,for_RKE=False):
        """Initial set-up is the same as for the Polymer class, but I also need
to unpack the vibrational_frequencies, which must be passed as a nested list.
Each site may have N vibrational modes, and each has a frequency, a displacement
and a frequency shift for the excited state
for sites a, b, ...
"""
        with open(yaml_file) as yamlstream:
            params = yaml.load(yamlstream,Loader=yaml.SafeLoader)
        self.base_path = os.path.split(yaml_file)[0]
        self.save_path = os.path.join(self.base_path,'open')
        os.makedirs(self.save_path,exist_ok=True)
        super().__init__(params['site_energies'],params['site_couplings'],np.array(params['dipoles']))

        self.H_diagonalization_time = 0
        self.L_diagonalization_time = 0
        self.L_construction_time = 0
        
        self.truncation_size = params['initial truncation size']
        try:
            self.maximum_manifold = params['maximum_manifold']
        except:
            self.maximum_manifold = np.inf
        self.maximum_manifold = min(self.maximum_manifold,self.num_sites)
        self.params = params

        self.set_bath_coupling()

        if self.optical_relaxation_gamma != 0:
            self.manifolds_separable = False
        else:
            self.manifolds_separable = True
        
        self.set_electronic_dissipation_instructions()
        
        self.occupation_num_mask = mask_by_occupation_num
        self.set_vibrations()
        self.set_vibrational_ladder_operators()
        
        e_ham = self.extract_electronic_subspace(self.electronic_hamiltonian,0,self.maximum_manifold)
            
        self.total_hamiltonian = np.kron(e_ham,self.vibrational_identity)
        self.add_vibrations()

        t0 = time.time()
        self.set_H_eigsystem_by_manifold()
        self.H_diagonalization_time = time.time() - t0
        
        self.make_condon_mu()
        self.make_condon_mu_dict()

        if force_detailed_balance:
            H_eigentransform = True
            t0 = time.time()
            self.all_instructions = self.make_commutator_instructions(-1j*self.total_hamiltonian)
            self.set_L_by_manifold(H_eigentransform=H_eigentransform,add_eigenstate_relaxation_effects = False)
            self.add_eigenstate_relaxation_effects()
            self.add_eigenstate_optical_dephasing_effects()
            self.L_construction_time = time.time() - t0
        else:
            H_eigentransform = False
            t0 = time.time()
            self.all_instructions = self.convert_electronic_instructions_to_full_instructions(self.electronic_dissipation_instructions)
            self.all_instructions += self.make_commutator_instructions(-1j*self.total_hamiltonian)
            self.all_instructions += self.vibrational_dissipation_instructions()
            if self.manifolds_separable:
                self.set_L_by_manifold(H_eigentransform=H_eigentransform)
            else:
                self.set_L()
            self.L_construction_time = time.time() - t0

        if for_RKE:
            self.set_mu_by_manifold(H_eigentransform=H_eigentransform,L_eigentransform=False)
            self.save_mu_by_manifold(pruned=False)
            self.save_L_by_manifold()
            self.save_rho0(H_eigentransform=H_eigentransform)
            
        else:
            t0 = time.time()
            if self.manifolds_separable:
                self.set_eigensystem_by_manifold(force_detailed_balance = force_detailed_balance)
                self.set_mu_by_manifold(H_eigentransform=H_eigentransform)
                self.save_mu_by_manifold(pruned=True)
                self.save_eigensystem_by_manifold()
                self.L_diagonalization_time = time.time() - t0
                
            else:
                self.set_eigensystem()
                # self.set_mu()
                # self.save_mu(pruned=True)
                # self.save_eigensystem()
                # self.L_diagonalization_time = time.time() - t0
                

        self.save_timings()

    def save_timings(self):
        save_dict = {'H_diagonalization_time':self.H_diagonalization_time,
                     'L_diagonalization_time':self.L_diagonalization_time,
                     'L_construction_time':self.L_construction_time}
        np.savez(os.path.join(self.save_path,'Liouvillian_timings.npz'),**save_dict)

    def set_H_eigsystem_by_manifold(self):
        self.H_eigenvalues = []
        self.H_eigenvectors = []
        for i in range(self.maximum_manifold+1):
            e,v = np.linalg.eigh(self.extract_vibronic_manifold(self.total_hamiltonian,i))
            for i in range(e.size):
                max_ind = np.argmax(np.abs(v[:,i]))
                if v[max_ind,i] < 0:
                    v[:,i] = v[:,i] * -1
            self.H_eigenvalues.append(e)
            self.H_eigenvectors.append(v)

    def save_rho0(self,*,H_eigentransform=False):
        H_size = self.H_eigenvalues[0].size
        if H_size == 1:
            rho0 = np.array([[1]])
        elif self.kT == 0:
            rho0 = np.zeros((H_size,H_size))
            rho0[0,0] = 1
        else:
            Z = np.sum(np.exp(-self.H_eigenvalues[0]/self.kT))
            rho0_diag = np.exp(-self.H_eigenvalues[0]/self.kT)/Z
            rho0 = np.diag(rho0_diag)

        if H_eigentransform:
            # Already in the eigenbasis
            pass
        else:
            # Go back to original basis
            v = self.H_eigenvectors[0]
            rho0 = v.dot(rho0.dot(v.T))

        rho0 = rho0.flatten()
        np.save(os.path.join(self.base_path,'rho0.npy'),rho0)

    def save_L(self):
        save_npz(os.path.join(self.save_path,'L.npz'),csr_matrix(self.L))

    def save_L_by_manifold(self):
        np.savez(os.path.join(self.save_path,'L.npz'),**self.L_by_manifold)
            

    def eigfun2(self,ket_manifold_num,bra_manifold_num,*,check_eigenvectors = True):
        key = str(ket_manifold_num) + str(bra_manifold_num)
        L = self.L_by_manifold[key]
        E = L.diagonal().copy()
        V = np.eye(E.size,dtype='complex')
        VL = V.copy()
        
        if ket_manifold_num == bra_manifold_num:
            size = self.H_eigenvalues[ket_manifold_num].size
            pop_inds = np.arange(size)*(size+1)
            L_pop = L[pop_inds,:]
            L_pop = L_pop[:,pop_inds]
            e, v, vl = self.eigfun(L_pop,populations_only=True)
            E[pop_inds] = e[:]
            for i,j in zip(pop_inds,range(len(pop_inds))):
                
                V[pop_inds,i] = v[:,j]
                
                VL[pop_inds,i] = vl[:,j]

        if check_eigenvectors:
            LV = L.dot(V)
            D = VL.dot(LV)
            if np.allclose(D,np.diag(E),rtol=1E-10,atol=1E-10):
                pass
            else:
                warnings.warn('Using eigenvectors to diagonalize Liouvillian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(np.max(np.abs(D - np.diag(E)))))

        self.eigenvalues = E
        self.eigenvectors = {'left':VL,'right':V}

        return E,V,VL

    def vibrational_occupation_to_indices(self,vibration,occ_num,manifold_num):
        single_mode_occ = np.arange(self.truncation_size)
        vib_occ = self.vibrational_vector_of_ones_kron(vibration,single_mode_occ)
        masked_single_mode_occ = vib_occ[self.vibrational_mask]

        electronic_manifold_hamiltonian = self.get_electronic_hamiltonian(manifold_num = manifold_num)
        elec_size = electronic_manifold_hamiltonian.shape[0]
        
        masked_single_mode_occ = np.kron(np.ones(elec_size),masked_single_mode_occ)
        return np.where(masked_single_mode_occ == occ_num)[0]

    def electronic_occupation_to_indices(self,site_num,manifold_num):
        single_mode_occ = np.arange(2)
        elec_occ = self.electronic_vector_of_ones_kron(site_num,single_mode_occ)
        mask = self.electronic_manifold_mask(manifold_num)
        masked_elec_occ = elec_occ[mask]
        masked_elec_occ = np.kron(masked_elec_occ,np.ones(self.vibrational_mask[0].size))

        return np.where(masked_elec_occ == 1)[0]

    def get_vibrational_relaxation_rates(self,manifold_num):
        e = self.H_eigenvalues[manifold_num]
        rates = np.zeros((e.size,e.size))
        for i in range(e.size):
            for j in range(e.size):
                for n in range(self.num_vibrations):
                    if j > i:
                        rates[i,j] += self.single_vibrational_relaxation_rate(i,j,n,manifold_num)
        return rates
    
    def single_vibrational_relaxation_rate(self,i,j,vibration,manifold_num):
        vi = self.H_eigenvectors[manifold_num][:,i]
        vj = self.H_eigenvectors[manifold_num][:,j]
        rate = 0
        for k in range(self.truncation_size):
            k_inds = self.vibrational_occupation_to_indices(vibration,k,manifold_num)
            kp1_inds = self.vibrational_occupation_to_indices(vibration,k+1,manifold_num)
            for k_ind,kp1_ind in zip(k_inds,kp1_inds):
                rate = rate + np.abs(vi[k_ind])**2 * np.abs(vj[kp1_ind])**2*np.sqrt(k+1)
        return rate

    def get_electronic_relaxation_rates(self,a,b,manifold_num):
        e = self.H_eigenvalues[manifold_num]
        rates = np.zeros((e.size,e.size))
        for i in range(e.size):
            for j in range(e.size):
                if j > i:
                    rates[i,j] += self.single_electronic_relaxation_rate(i,j,a,b,manifold_num)
        return rates

    def get_all_electronic_relaxation_rates(self,manifold_num):
        """Treats all sites as having the same relaxation rates
"""
        e = self.H_eigenvalues[manifold_num]
        rates = np.zeros((e.size,e.size))
        for i in range(e.size):
            for j in range(e.size):
                if j > i:
                    for a in range(len(self.energies)):
                        Ea = self.energies[a]
                        for b in range(len(self.energies)):
                            Eb = self.energies[b]
                            if Eb > Ea:
                                rates[i,j] += self.single_electronic_relaxation_rate(i,j,a,b,manifold_num)
        return rates

    def get_all_relaxation_rates(self,manifold_num):
        rates = self.vibrational_gamma * self.get_vibrational_relaxation_rates(manifold_num)
        rates = rates + self.site_to_site_relaxation_gamma * self.get_all_electronic_relaxation_rates(manifold_num)
        return rates

    def all_eigenstate_relaxation_instructions_by_manifold(self,manifold_num):
        rates = self.get_all_relaxation_rates(manifold_num)
        E = self.H_eigenvalues[manifold_num]
        ins = []
        for i in range(rates.shape[0]):
            for j in range(rates.shape[1]):
                if j > i:
                    O = np.zeros(rates.shape)
                    O[i,j] = 1
                    down, up = self.boltzmann_factors(E[i],E[j])
                    down = down * rates[i,j]
                    up = up * rates[i,j]
                    ins += self.make_Lindblad_instructions(down,O)
                    if np.isclose(up,0):
                        pass
                    else:
                        ins += self.make_Lindblad_instructions(up,O.T)
        return ins

    def all_eigenstate_relaxation_instructions_by_coherence(self,ket_manifold_num,bra_manifold_num):
        if ket_manifold_num == bra_manifold_num:
            return self.all_eigenstate_relaxation_instructions_by_manifold(ket_manifold_num)
        ket_rates = self.get_all_relaxation_rates(ket_manifold_num)
        E_ket = self.H_eigenvalues[ket_manifold_num]
        bra_rates = self.get_all_relaxation_rates(bra_manifold_num)
        E_bra = self.H_eigenvalues[bra_manifold_num]
        ins = []
        Obra = np.zeros(bra_rates.shape)
        for i in range(ket_rates.shape[0]):
            for j in range(ket_rates.shape[1]):
                if j > i:
                    Oket = np.zeros(ket_rates.shape)
                    Oket[i,j] = 1
                    down,up = self.boltzmann_factors(E_ket[i],E_ket[j])
                    down = down * ket_rates[i,j]
                    up = up * ket_rates[i,j]
                        
                    ins += self.make_Lindblad_instructions2_Obra0(down,Oket,Obra)

                    if np.isclose(up,0):
                        pass
                    else:
                        ins += self.make_Lindblad_instructions2_Obra0(up,Oket.T,Obra)

        Oket = np.zeros(ket_rates.shape)
        for i in range(bra_rates.shape[0]):
            for j in range(bra_rates.shape[1]):
                if j > i:
                    Obra = np.zeros(bra_rates.shape)
                    Obra[i,j] = 1
                    down,up = self.boltzmann_factors(E_bra[i],E_bra[j])
                    down = down * bra_rates[i,j]
                    up = up * bra_rates[i,j]
                        
                    ins += self.make_Lindblad_instructions2_Oket0(down,Oket,Obra)

                    if np.isclose(up,0):
                        pass
                    else:
                        ins += self.make_Lindblad_instructions2_Oket0(up,Oket,Obra.T)
        return ins
                            
    def single_electronic_relaxation_rate(self,i,j,a,b,manifold_num):
        vi = self.H_eigenvectors[manifold_num][:,i]
        vj = self.H_eigenvectors[manifold_num][:,j]
        a_inds = self.electronic_occupation_to_indices(a,manifold_num)
        b_inds = self.electronic_occupation_to_indices(b,manifold_num)
        rate = np.sum(np.abs(vi[a_inds])**2) * np.sum(np.abs(vj[b_inds])**2)

        return rate

    def make_eigenstate_relaxation_Lindblad_all_rates(self,rates,manifold_num):
        """From j to i. Factor of 0.5 matches my previous definition of Lindblad formalism"""
        E = self.H_eigenvalues[manifold_num]
        size = E.size
        pop_inds = np.arange(size)*(size+1)
        pop_subspace = np.zeros((pop_inds.size,pop_inds.size))
        L_diagonal = np.zeros((size,size))
        
        for i in range(size):
            for j in range(size):
                if j > i:
                    down,up = self.boltzmann_factors(E[i],E[j])
                    down = down * rates[i,j]
                    up = up * rates[i,j]

                    pop_subspace[j,j] += -0.5*down
                    pop_subspace[i,j] += 0.5*down
                    pop_subspace[i,i] += -0.5*up
                    pop_subspace[j,i] += 0.5*up

                    L_diagonal[j,:] += -0.25*down
                    L_diagonal[:,j] += -0.25*down
                    L_diagonal[j,j] += -0.5*down

                    L_diagonal[i,:] += -0.25*up
                    L_diagonal[:,i] += -0.25*up
                    L_diagonal[i,i] += -0.5*up
                    
        L_total = np.diag(L_diagonal.ravel())
        for i,j in zip(pop_inds,np.arange(pop_inds.size)):
            L_total[i,pop_inds] = pop_subspace[j,:]

        return L_total

    def make_eigenstate_relaxation_Lindblad_all_rates_by_coherence(self,ket_rates,bra_rates,ket_manifold_num,bra_manifold_num):
        """From j to i. Factor of 0.5 matches my previous definition of Lindblad formalism"""
        if ket_manifold_num == bra_manifold_num:
            return self.make_eigenstate_relaxation_Lindblad_all_rates(ket_rates,ket_manifold_num)
        E_ket = self.H_eigenvalues[ket_manifold_num]
        E_bra = self.H_eigenvalues[bra_manifold_num]
        ket_size = E_ket.size
        bra_size = E_bra.size
        L_diagonal = np.zeros((ket_size,bra_size))
        
        for i in range(ket_size):
            for j in range(ket_size):
                if j > i:
                    down,up = self.boltzmann_factors(E_ket[i],E_ket[j])
                    down = down * ket_rates[i,j]
                    up = up * ket_rates[i,j]

                    L_diagonal[j,:] += -0.25*down
                    L_diagonal[i,:] += -0.25*up

        for i in range(bra_size):
            for j in range(bra_size):
                if j > i:
                    down,up = self.boltzmann_factors(E_bra[i],E_bra[j])
                    down = down * bra_rates[i,j]
                    down = down * bra_rates[i,j]

                    L_diagonal[:,j] += -0.25*down
                    L_diagonal[:,i] += -0.25*up
                    
        L_total = np.diag(L_diagonal.ravel())

        return L_total

    def add_eigenstate_relaxation_effects(self):
        for k in range(self.maximum_manifold+1):
            rates_k = self.get_all_relaxation_rates(k)
            for l in range(self.maximum_manifold+1):
                rates_l = self.get_all_relaxation_rates(l)
                key = str(k) + str(l)
                L = self.L_by_manifold[key]
                L += self.make_eigenstate_relaxation_Lindblad_all_rates_by_coherence(rates_k,rates_l,k,l)

    def add_eigenstate_optical_dephasing_effects(self):
        for k in range(self.maximum_manifold+1):
            for l in range(self.maximum_manifold+1):
                if k == l:
                    pass
                else:
                    key = str(k) + str(l)
                    L = self.L_by_manifold[key]
                    L += self.make_eigenstate_optical_dephasing_Lindblad(k,l)

    def make_eigenstate_relaxation_Lindblad(self,gamma,i,j,manifold_num):
        """From j to i. Factor of 0.5 matches my previous definition of Lindblad formalism"""
        size = self.H_eigenvalues[manifold_num].size
        pop_inds = np.arange(size)*(size+1)
        pop_subspace = np.zeros((pop_inds.size,pop_inds.size))
        pop_subspace[j,j] = -0.5
        pop_subspace[i,j] = 0.5

        L_diagonal = np.zeros((size,size))
        L_diagonal[j,:] = -0.25
        L_diagonal[:,j] = -0.25
        L_diagonal[j,j] = -0.5
        L_total = np.diag(L_diagonal.ravel())
        for i,j in zip(pop_inds,np.arange(pop_inds.size)):
            L_total[i,pop_inds] = pop_subspace[j,:]

        return gamma*L_total

    def make_eigenstate_relaxation_Lindblad_optical_coherence(self,gamma,i,j,ket_manifold_num,bra_manifold_num,*,
                                                              relaxation_in_ket = True):
        """From j to i. Factor of 0.25 matches my previous definition of Lindblad formalism"""
        ket_size = self.H_eigenvalues[ket_manifold_num].size
        bra_size = self.H_eigenvalues[bra_manifold_num].size

        L_diagonal = np.zeros((ket_size,bra_size))
        if relaxation_in_ket:
            L_diagonal[j,:] = -0.25
        else:
            L_diagonal[:,j] = -0.25
        L_total = np.diag(L_diagonal.ravel())

        return gamma*L_total

    def make_eigenstate_optical_dephasing_Lindblad(self,ket_manifold_num,bra_manifold_num):
        """Use a constant dephasing rate for all states: my best idea is to
createe the dephasing Lindblad for the electronic space only, and use it to 
fill in a single rate on the diagonal of the Liouvillian.  The trick is to get
dephasing between the nth and n+kth manifold right, when k > 1 (k = 1 is simply 
gamma)"""
        opt_deph = self.optical_dephasing_Liouvillian().diagonal().reshape(self.electronic_hamiltonian.shape)
        
        opt_deph = self.extract_coherence(opt_deph,ket_manifold_num,bra_manifold_num).ravel()

        if np.allclose(opt_deph[0],opt_deph):
            pass
        else:
            raise Exception('All optical dephasing rates are not the same, unknown error')

        ket_size = self.H_eigenvalues[ket_manifold_num].size
        bra_size = self.H_eigenvalues[bra_manifold_num].size

        opt_deph = np.ones((ket_size,bra_size),dtype='complex') * opt_deph[0]

        return np.diag(opt_deph.ravel())

    def set_bath_coupling(self):
        try:
            self.site_to_site_relaxation_gamma = self.params['bath']['site_to_site_relaxation_gamma']
        except KeyError:
            pass

        try:
            self.site_to_site_dephasing_gamma = self.params['bath']['site_to_site_dephasing_gamma']
        except KeyError:
            pass

        
        try:
            self.optical_dephasing_gamma = self.params['bath']['optical_dephasing_gamma']
        except KeyError:
            pass

        try:
            self.optical_relaxation_gamma = self.params['bath']['optical_relaxation_gamma']
        except KeyError:
            pass

        
        try:
            self.vibrational_gamma = self.params['bath']['vibrational_gamma']
        except KeyError:
            self.vibrational_gamma = 0.1

            
        try:
            self.kT = self.params['bath']['kT']
        except KeyError:
            pass

    def convert_electronic_instructions_to_full_instructions(self,inst_list):
        new_inst_list = []
        for ins in inst_list:
            left,right = ins
            if self.manifolds_separable == True:
                pass
            else:
                left = self.extract_electronic_subspace(left,0,self.maximum_manifold)
                right = self.extract_electronic_subspace(right,0,self.maximum_manifold)
            left = np.kron(left,self.vibrational_identity)
            right = np.kron(right,self.vibrational_identity)
            new_inst_list.append((left,right))
        return new_inst_list

    def vibronic_manifold_mask(self,manifold_num):
        """Gets the indices of the Hilbert space that occupy a particular electronic
            manifold, including all vibrational degrees of freedom from that manifold
"""
        try:
            vib_size = self.vibrational_mask[0].size
        except AttributeError:
            N = self.truncation_size
            nv = self.num_vibrations
            vib_size = N**nv
        vib_ones = np.ones(vib_size,dtype='int')
        vibronic_occupation_number = np.kron(self.electronic_total_occupation_number,vib_ones)
        manifold_inds = np.where(vibronic_occupation_number == manifold_num)[0]
        return manifold_inds

    def extract_vibronic_coherence(self,O,manifold1,manifold2):
        """Returns result of projecting the Operator O onto manifold1
            on the left and manifold2 on the right
"""
        manifold1_inds = self.vibronic_manifold_mask(manifold1)
        manifold2_inds = self.vibronic_manifold_mask(manifold2)
        O = O[manifold1_inds,:]
        O = O[:,manifold2_inds]
        return O
    
    def extract_vibronic_manifold(self,O,manifold_num):
        """Projects operator into the given electronic excitation manifold
"""
        return self.extract_vibronic_coherence(O,manifold_num,manifold_num)

    def set_L(self):
        self.L = self.make_Liouvillian(self.all_instructions)

    def set_eigensystem(self):
        self.eigfun(self.L)

    def set_L_by_manifold(self,*,H_eigentransform=False,add_eigenstate_relaxation_effects = False):
        all_inst = self.all_instructions
        
        self.L_by_manifold = dict()
        for i in range(self.maximum_manifold+1):
            for j in range(self.maximum_manifold+1):
                key = str(i) + str(j)
                inst = self.extract_coherence_instructions_from_full_instructions(all_inst,i,j,H_eigentransform=H_eigentransform)
                if add_eigenstate_relaxation_effects:
                    inst += self.all_eigenstate_relaxation_instructions_by_coherence(i,j)
                self.L_by_manifold[key] = self.make_Liouvillian(inst)

    def set_eigensystem_by_manifold(self,*,force_detailed_balance = False):
        self.right_eigenvectors_by_manifold = dict()
        self.left_eigenvectors_by_manifold = dict()
        self.eigenvalues_by_manifold = dict()
        for i in range(self.maximum_manifold+1):
            for j in range(self.maximum_manifold+1):
                key = str(i) + str(j)
                if force_detailed_balance:
                    e, r, l = self.eigfun2(i,j,check_eigenvectors = False)
                else:
                    e, r, l = self.eigfun(self.L_by_manifold[key])
                self.right_eigenvectors_by_manifold[key] = r
                self.left_eigenvectors_by_manifold[key] = l
                self.eigenvalues_by_manifold[key] = e

    def make_mu_by_manifold_ket(self,old_manifold,change,*,H_eigentransform=False,L_eigentransform=True):
        i,j = old_manifold
        i2 = i + change
        if i2 >= 0 and i2 <= self.maximum_manifold:
            pass
        else:
            return None, None
        if H_eigentransform:
            Vold = self.H_eigenvectors[i]
            Vnew = self.H_eigenvectors[i2]
        else:
            pass
        j2 = j
        bra_eye = np.eye(self.extract_vibronic_manifold(self.total_hamiltonian,j).shape[0])
        old_key = str(i) + str(j)
        new_key = str(i2) + str(j2)
        all_mus = []
        mu_dtype='float64'
        for pol in self.pols:
            full_mu = self.vibronic_mu_dict[pol]
            mu = self.extract_vibronic_coherence(full_mu,i2,i)
            if H_eigentransform:
                mu = Vnew.T.dot(mu.dot(Vold))
            mu = np.kron(mu,bra_eye)
            if L_eigentransform:
                l = self.left_eigenvectors_by_manifold[new_key]
                r = self.right_eigenvectors_by_manifold[old_key]
                mu = l.dot(mu.dot(r))
            if np.allclose(np.imag(mu),0):
                mu = np.real(mu)
            else:
                mu_dtype = 'complex128'
            all_mus.append(mu)
        mu_shape = all_mus[0].shape
        mu_3d = np.zeros((mu_shape[0],mu_shape[1],3),dtype=mu_dtype)
        for i in range(3):
            mu_3d[:,:,i] = all_mus[i]
        mu_key = old_key + '_to_' + new_key
        return mu_key, mu_3d

    def make_mu_by_manifold_bra(self,old_manifold,change,*,H_eigentransform=False,L_eigentransform=True):
        i,j = old_manifold
        j2 = j + change
        if j2 >= 0 and j2 <= self.maximum_manifold:
            pass
        else:
            return None, None
        if H_eigentransform:
            Vold = self.H_eigenvectors[j]
            Vnew = self.H_eigenvectors[j2]
        else:
            pass
        i2 = i
        ket_eye = np.eye(self.extract_vibronic_manifold(self.total_hamiltonian,i).shape[0])
        old_key = str(i) + str(j)
        new_key = str(i2) + str(j2)
        all_mus = []
        mu_dtype='float64'
        for pol in self.pols:
            full_mu = self.vibronic_mu_dict[pol]
            mu = self.extract_vibronic_coherence(full_mu,j,j2)
            if H_eigentransform:
                mu = Vold.T.dot(mu.dot(Vnew))
            mu = np.kron(ket_eye,mu.T)
            if L_eigentransform:
                l = self.left_eigenvectors_by_manifold[new_key]
                r = self.right_eigenvectors_by_manifold[old_key]
                mu = l.dot(mu.dot(r))
            if np.allclose(np.imag(mu),0):
                mu = np.real(mu)
            else:
                mu_dtype = 'complex128'
            all_mus.append(mu)
        mu_shape = all_mus[0].shape
        mu_3d = np.zeros((mu_shape[0],mu_shape[1],3),dtype=mu_dtype)
        for i in range(3):
            mu_3d[:,:,i] = all_mus[i]
        mu_key = old_key + '_to_' + new_key
        return mu_key, mu_3d

    def append_mu_by_manifold(self,old_manifold,change,ket_flag,H_eigentransform=False,
                              L_eigentransform=True):
        if ket_flag:
            f = self.make_mu_by_manifold_ket
        else:
            f = self.make_mu_by_manifold_bra
        key, mu = f(old_manifold,change,H_eigentransform=H_eigentransform,
                    L_eigentransform=L_eigentransform)
        if key == None:
            pass
        else:
            boolean_mu = np.zeros(mu.shape[:2],dtype='bool')
            boolean_mu[:,:] = np.round(np.sum(np.abs(mu)**2,axis=-1),12)
            mu = mu * boolean_mu[:,:,np.newaxis]
            self.boolean_mu_by_manifold[key] = boolean_mu
            self.mu_by_manifold[key] = mu

    def set_mu_by_manifold(self,H_eigentransform=False,L_eigentransform=True):
        self.mu_by_manifold = dict()
        self.boolean_mu_by_manifold = dict()
        changes = [-1,1]
        for i in range(self.maximum_manifold+1):
            for j in range(self.maximum_manifold+1):
                manifold = (i,j)
                self.append_mu_by_manifold(manifold,1,True,H_eigentransform=H_eigentransform,L_eigentransform=L_eigentransform)
                self.append_mu_by_manifold(manifold,-1,True,H_eigentransform=H_eigentransform,L_eigentransform=L_eigentransform)
                self.append_mu_by_manifold(manifold,1,False,H_eigentransform=H_eigentransform,L_eigentransform=L_eigentransform)
                self.append_mu_by_manifold(manifold,-1,False,H_eigentransform=H_eigentransform,L_eigentransform=L_eigentransform)
                
    def save_mu_by_manifold(self,*,pruned=True):
        if pruned:
            np.savez(os.path.join(self.save_path,'mu_pruned.npz'),**self.mu_by_manifold)
            np.savez(os.path.join(self.save_path,'mu_boolean.npz'),**self.boolean_mu_by_manifold)
        else:
            np.savez(os.path.join(self.save_path,'mu.npz'),**self.mu_by_manifold)

    def save_eigensystem_by_manifold(self):
        np.savez(os.path.join(self.save_path,'eigenvalues.npz'),**self.eigenvalues_by_manifold)
        np.savez(os.path.join(self.save_path,'right_eigenvectors.npz'),**self.right_eigenvectors_by_manifold)
        np.savez(os.path.join(self.save_path,'left_eigenvectors.npz'),**self.left_eigenvectors_by_manifold)
                
    def extract_coherence_instructions_from_full_instructions(self,inst_list,manifold1,manifold2,*,H_eigentransform=False,trim = None):
        new_inst_list = []
        H1 = self.extract_vibronic_manifold(self.total_hamiltonian,manifold1)
        H2 = self.extract_vibronic_manifold(self.total_hamiltonian,manifold2)
        if H_eigentransform:
            V1 = self.H_eigenvectors[manifold1]
            V2 = self.H_eigenvectors[manifold2]
        else:
            V1 = np.eye(H1.shape[0])
            V2 = np.eye(H2.shape[0])
        for (left,right) in inst_list:
            new_left = self.extract_vibronic_manifold(left,manifold1)
            new_left = V1.T.dot(new_left.dot(V1))
            new_right = self.extract_vibronic_manifold(right,manifold2)
            new_right = V2.T.dot(new_right.dot(V2))
            new_inst_list.append((new_left[:trim,:trim],new_right[:trim,:trim]))
        return new_inst_list

    def extract_manifold_instructions_from_full_instructions(self,inst_list,manifold):
        return self.extract_coherence_instructions_from_full_instructions(inst_list,manifold,manifold)
    
    def add_vibrations(self):
        v0 = self.empty_vibrations
        v1 = self.occupied_vibrations
        self.vibrational_hamiltonian = np.zeros(self.total_hamiltonian.shape)
        for i in range(len(v0)):
            self.vibrational_hamiltonian += v0[i]
            self.vibrational_hamiltonian += v1[i]

        self.total_hamiltonian = self.total_hamiltonian + self.vibrational_hamiltonian

    def set_vibrations(self):
        vibration_params = self.params['vibrations']
        # Vibrations in the ground manifold are assumed to be diagonal
        
        
        emp_vibs = [self.construct_vibrational_hamiltonian(mode_dict,0)
                    for mode_dict in vibration_params]
        self.num_vibrations = len(emp_vibs)
        occ_vibs = [self.construct_vibrational_hamiltonian(mode_dict,1)
                    for mode_dict in vibration_params]

        if self.occupation_num_mask:
            self.set_vibrational_total_occupation_number()
        else:
            N = self.truncation_size
            nv = self.num_vibrations
            self.vibrational_mask = (np.arange(N**nv),)
            self.vibrational_identity = np.eye(N**nv)
        empty_vibrations = self.kron_up_vibrations(emp_vibs)
        occupied_vibrations = self.kron_up_vibrations(occ_vibs)

        self.empty_vibrations = []
        self.occupied_vibrations = []
        
        for i in range(self.num_vibrations):
            site_index = vibration_params[i]['site_label']
            if self.manifolds_separable == True:
                empty = self.empty_list[site_index]
                occupied = self.occupied_list[site_index]
            else:
                empty = self.extract_electronic_subspace(self.empty_list[site_index],0,self.maximum_manifold)
                occupied = self.extract_electronic_subspace(self.occupied_list[site_index],0,self.maximum_manifold)
            self.empty_vibrations.append(np.kron(empty,empty_vibrations[i]))
            self.occupied_vibrations.append(np.kron(occupied,occupied_vibrations[i]))

    def kron_up_vibrations(self,vibrations_list):
        n = self.num_vibrations
        if n == 1:
            return vibrations_list
        new_vibrations_list = []
        for i in range(n):
            new_vibration = self.vibration_identity_kron(i,vibrations_list[i])
            if self.occupation_num_mask:
                new_vibration = self.mask_vibrational_space(new_vibration)
            new_vibrations_list.append(new_vibration)
        return new_vibrations_list
            
    def mask_vibrational_space(self,O):
        inds = self.vibrational_mask
        if type(O) is np.ndarray:
            O = O[inds[0],:].copy()
            O = O[:,inds[0]].copy()
            return O
        
        if type(O) is csr_matrix:
            pass
        else:
            O = O.tocsr()
        O = O[inds[0]]
        O = O.transpose()
        O = O[inds[0]]
        O = O.transpose()
        return O

    def vibration_identity_kron(self,position,item):
        """Takes in a single vibrational hamiltonians and krons it with the correct 
            number of vibrational identities, inserting it into its position as indexed by its mode
            position as specified in the input file"""
        identities = [np.eye(self.truncation_size) for n in
                      range(self.num_vibrations-1)]
        identities.insert(position,item)
        mat = identities.pop(0)
        for next_item in identities:
            mat = np.kron(mat,next_item)
        return mat

    def vibrational_vector_of_ones_kron(self,position,item):
        """Takes in a single vibrational hamiltonians and krons it with the correct 
            number of vibrational identities, inserting it into its position as indexed by its mode
            position as specified in the input file"""
        N = self.truncation_size
        nv = self.num_vibrations
        ones_list = [np.ones(N) for i in range(nv-1)]
        ones_list.insert(position,item)
        vec = ones_list.pop(0)
        for next_item in ones_list:
            vec = np.kron(vec,next_item)
        return vec

    def set_vibrational_total_occupation_number(self):
        N = self.truncation_size
        nv = self.num_vibrations
        single_mode_occ = np.arange(N)
        occ_num = self.vibrational_vector_of_ones_kron(0,single_mode_occ)
        for i in range(1,nv):
            occ_num += self.vibrational_vector_of_ones_kron(i,single_mode_occ)
        self.vibrational_total_occupation_number = occ_num
        self.vibrational_mask = np.where(occ_num < N)
        self.vibrational_identity = np.eye(self.vibrational_mask[0].size)

    def construct_vibrational_hamiltonian(self,single_mode,electronic_occupation):
        """For each vibrational mode, construct a list of sparse matrices defining the 
            vibrational hamiltonian for that mode in each excited state"""
        w = single_mode['omega_g']
        lam = single_mode['reorganization'][electronic_occupation]
        d  = single_mode['displacement'][electronic_occupation]
        kin = single_mode['kinetic'][electronic_occupation]
        pot = single_mode['potential'][electronic_occupation]
        aho = DisplacedAnharmonicOscillator(self.truncation_size)
        aho.set_ham(lam,d,kin,pot)
        return 0.5 * w * aho.ham

    def construct_vibrational_ladder_operator(self,single_mode,electronic_occupation):
        """Construct ladder operator given the electronic occupation for that site"""
        w = single_mode['omega_g']
        d  = single_mode['displacement'][electronic_occupation]
        lad = LadderOperators(self.truncation_size,disp=d,extra_size=0)
        up = lad.ad
        return up

    def set_vibrational_ladder_operators(self):
        vibration_params = self.params['vibrations']
        emp_ups = []
        occ_ups = []

        for i in range(len(vibration_params)):
            ad = self.construct_vibrational_ladder_operator(vibration_params[i],0)
            emp_ups.append(ad)

            ad = self.construct_vibrational_ladder_operator(vibration_params[i],1)
            occ_ups.append(ad)

        empty_ups = self.kron_up_vibrations(emp_ups)
        occupied_ups = self.kron_up_vibrations(occ_ups)
        self.empty_ups = []
        self.occupied_ups = []
        for i in range(self.num_vibrations):
            site_index = vibration_params[i]['site_label']
            if self.manifolds_separable == True:
                empty = self.empty_list[site_index]
                occupied = self.occupied_list[site_index]
            else:
                empty = self.extract_electronic_subspace(self.empty_list[site_index],0,self.maximum_manifold)
                occupied = self.extract_electronic_subspace(self.occupied_list[site_index],0,self.maximum_manifold)
            self.empty_ups.append(np.kron(empty,empty_ups[i]))
            self.occupied_ups.append(np.kron(occupied,occupied_ups[i]))

    def make_vibrational_dissipation_Liouvillian(self):
        ins_list = self.vibrational_dissipation_instructions()
        L = self.make_Liouvillian(ins_list)

        return L

    def vibrational_dissipation_instructions(self):
        gamma = self.vibrational_gamma
        instructions = []
        for k in range(self.num_vibrations):
            E = self.params['vibrations'][k]['omega_g']
            if self.params['vibrations'][k]['potential'][1][0] != 1:
                warnings.warn('The case of different excited and ground state frequencies is not properly handled by thermal dissipation')
            if self.kT == 0:
                N = 0
            else:
                N = 1/(np.exp(E/self.kT)-1)
            O = (self.occupied_ups[k]).T + (self.empty_ups[k]).T
            ins1 = self.make_Lindblad_instructions(gamma*(N+1),O)
            instructions += ins1
            if N == 0:
                pass
            else:
                ins2 = self.make_Lindblad_instructions(gamma*N,O.T)
                instructions += ins2
        return instructions

    def make_total_Liouvillian(self):
        ins = self.make_commutator_instructions(-1j*self.total_hamiltonian)
        self.L = self.make_Liouvillian(ins)
        self.L += self.make_vibrational_dissipation_Liouvillian()

    def make_condon_mu(self):
        try:
            vib_size = self.vibrational_mask[0].size
        except AttributeError:
            N = self.truncation_size
            nv = self.num_vibrations
            vib_size = N**nv
        self.mu = np.kron(self.mu,np.eye(vib_size))
        self.mu_ket_up = np.kron(self.mu_ket_up,np.eye(vib_size))

    def make_condon_mu_dict(self):
        try:
            vib_size = self.vibrational_mask[0].size
        except AttributeError:
            N = self.truncation_size
            nv = self.num_vibrations
            vib_size = N**nv
        self.vibronic_mu_dict = dict()
        for pol in self.pols:
            self.vibronic_mu_dict[pol] =  np.kron(self.mu_dict[pol],np.eye(vib_size))


class run_HLG:

    def __init__(self,folder,*,for_RKE = False,conserve_memory = False):
        self.for_RKE = for_RKE
        self.conserve_memory = conserve_memory
        self.folder = folder
        try:
            convert(folder)
        except FileNotFoundError:
            pass

        try:
            pv = PolymerVibrations(os.path.join(folder,'params.yaml'))
        except FileNotFoundError:
            raise Exception('Could not find params.yaml file in specified folder')

        if 'bath' in pv.params.keys():
            self.open = True
            if 'site_bath' in pv.params['bath'].keys():
                self.Redfield = True
            else:
                self.Redfield = False
                try:
                    self.fdb_flag = pv.params['bath']['force_detailed_balance']
                except KeyError:
                    self.fdb_flag = False
        else:
            self.open = False

        if self.open:
            self.run_open()
        else:
            self.run_closed()

    def run_closed(self):
        if self.for_RKE:
            pass
        else:
            dh = DiagonalizeHamiltonian(self.folder)

    def run_open(self):
        if self.Redfield:
            dh = DiagonalizeHamiltonian(self.folder)
            rf = RedfieldConstructor(self.folder,conserve_memory=self.conserve_memory)
            dl = DiagonalizeLiouvillian(self.folder,conserve_memory=self.conserve_memory)
        else:
            opv = OpenPolymerVibrations(os.path.join(self.folder,'params.yaml'),for_RKE=self.for_RKE,force_detailed_balance=self.fdb_flag)
