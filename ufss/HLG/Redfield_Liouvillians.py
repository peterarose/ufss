import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import csr_matrix, csc_matrix, identity, kron, save_npz, load_npz
from scipy.sparse.linalg import eigs, eigsh, spsolve
import itertools
from scipy.linalg import block_diag, eig, expm, eigh
import scipy.sparse as sp
from scipy.special import binom
from scipy.sparse.csgraph import connected_components
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

    def __init__(self,lam,gam,kT,*,cutoff='lorentz-drude',S = 0, Omega = 0, Gamma = 0, nu = 0):
        """
        Args:
            lam (float): bath coupling strength
            gam (float): bath decay rate or cutoff frequency
            kT (float): temperature
            cutoff (str): high-frequency cutoff function (options: 'lorentz-drude'
                'exponential' or 'brownian')
                N.B.: 'brownian' includes the SD and a cutoff
            S (float) : Huang-Rhys factor
            Omega (float) : oscillator frequency of the bath
            Gamma (float) : oscillator damping of the bath
            nu (float) : scaling factor of the bath
"""
        self.lam = lam
        self.gam = gam
        self.prefactor = 2 * self.lam/self.gam
        self.kT = kT
        self.S = S
        self.Omega = Omega
        self.Gamma = Gamma
        self.nu = nu


        if cutoff == 'lorentz-drude':
            self.J = self.lorentz_drude
        elif cutoff == 'exponential':
            self.J = self.exponential
        elif cutoff == 'brownian':
            self.J = self.brownian


    def lorentz_drude(self,w):
        return self.prefactor * self.gam**2 / (w**2 + self.gam**2)
    
    def exponential(self,w):
        return self.prefactor * np.exp( -np.abs(w) / self.gam)

    def brownian(self, w):
        return 2 * self.nu * self.prefactor*(self.gam**2/(w**2 + self.gam**2) + self.gam/self.lam * (self.S * self.Omega**3 * self.Gamma)/((self.Omega**2 - w**2)**2 + w**2 * self.Gamma**2))

    def temp_dist(self,w):
        """Includes the ohmic term w
"""
        return w * (1+self.coth(w))/2

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
            f[zi] *= self.kT
        elif zero_ind.size == 0:
            f = f * self.temp_dist(w)
        else:
            raise Exception("Can't handle more than one value of w = 0")
        return f

    def C_scal(self,w):
        f = self.J(w)
        if w == 0:
            f *= self.kT
        else:
            f *= self.temp_dist(w)
        return f

    def __call__(self,w):
        if type(w) is np.ndarray:
            return self.C_vec(w)
        else:
            return self.C_scal(w)

class WhiteNoiseSpectralDensity:
    """Creates a white noise (flat) spectral density function
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

        self.num_vibrations = len(self.params['vibrations'])
            
        self.PolyVib = PolymerVibrations(os.path.join(folder,'params.yaml'),
                                         separable_manifolds=separable_manifolds)

        self.N = self.PolyVib.Polymer.N

        # Verify whether we have more than one baths option for singly-excited sites
        if isinstance(self.params['bath']['site_bath'], list):
            self.multiple_site_bath = True
            if 'site_bath_option' in self.params['bath']:
                pass
            else:
                raise Exception(
                    'User must include site_bath_option list in order to treat different baths for each site')
        else:
            self.multiple_site_bath = False

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

        if self.multiple_site_bath:
            self.SD = {}
            self.SD['site_bath'] = []

            for n in range(self.PolyVib.Polymer.num_sites):
                bath_option = self.params['bath']['site_bath_option'][n]
                spec_density = self.make_spectral_density(self.params['bath']['site_bath'][bath_option])

                self.SD['site_bath'].append(spec_density)

            if 'site_bath2' in self.params['bath']:
                self.SD['site_bath2'] = []
                for n in range(self.PolyVib.Polymer.num_sites):
                    bath2_option = self.params['bath']['site_bath2_option'][n]
                    spec_density = self.make_spectral_density(self.params['bath']['site_bath2'][bath2_option])
                    self.SD['site_bath2'].append(spec_density)

            if 'site_bath3' in self.params['bath']:
                self.SD['site_bath3'] = []
                for n in range(self.PolyVib.Polymer.num_sites):
                    bath3_option = self.params['bath']['site_bath3_option'][n]
                    spec_density = self.make_spectral_density(self.params['bath']['site_bath3'][bath3_option])
                    self.SD['site_bath3'].append(spec_density)

            if self.num_vibrations > 0 :
                if 'vibration_bath' in self.params['bath'].keys():
                    self.SD['vibration_bath'] = self.make_spectral_density(self.params['bath']['vibration_bath'])
                else:
                    self.SD['vibration_bath'] = self.make_spectral_density(self.params['bath']['site_bath'])
                    warnings.warn('Using site_bath for vibration_bath')
            try:
                self.SD['site_internal_conversion_bath'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath'])
                if self.PolyVib.separable_manifolds == True:
                    raise Exception('Cannot model site internal conversion processes when input Hamiltonian has been divided into separate manifolds')
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath21'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath21'])
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath20'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath20'])
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath32'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath32'])
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath31'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath31'])
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath30'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath30'])
            except KeyError:
                pass

        else :
            self.SD = {}
            self.SD['site_bath'] = self.make_spectral_density(self.params['bath']['site_bath'])
            if self.num_vibrations > 0:
                if 'vibration_bath' in self.params['bath'].keys():
                    self.SD['vibration_bath'] = self.make_spectral_density(self.params['bath']['vibration_bath'])
                else:
                    self.SD['vibration_bath'] = self.make_spectral_density(self.params['bath']['site_bath'])
                    warnings.warn('Using site_bath for vibration_bath')
            try:
                self.SD['site_internal_conversion_bath'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath'])
                if self.PolyVib.separable_manifolds == True:
                    raise Exception('Cannot model site internal conversion processes when input Hamiltonian has been divided into separate manifolds')
            except KeyError:
                pass

            try:
                self.SD['site_bath2'] = self.make_spectral_density(self.params['bath']['site_bath2'])
            except KeyError:
                pass

            try:
                self.SD['site_bath3'] = self.make_spectral_density(self.params['bath']['site_bath3'])
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath21'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath21'])
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath20'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath20'])
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath32'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath32'])
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath31'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath31'])
            except KeyError:
                pass

            try:
                self.SD['site_internal_conversion_bath30'] = self.make_spectral_density(self.params['bath']['site_internal_conversion_bath30'])
            except KeyError:
                pass


    def make_spectral_density(self,bath_dict):
        kT = bath_dict['temperature']
        spectrum_type = bath_dict['spectrum_type']

        if spectrum_type == 'ohmic':
            lam = bath_dict['coupling']
            gam = bath_dict['cutoff_frequency']
            cutoff = bath_dict['cutoff_function']
            if cutoff == 'brownian':
                S  = bath_dict['S']
                Gamma = bath_dict['Gamma']
                Omega = bath_dict['Omega']
                nu = bath_dict['nu']
                SD = OhmicSpectralDensity(lam,gam,kT, cutoff = cutoff, S = S, Omega = Omega, Gamma = Gamma, nu = nu)
            else :
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
        size = self.eigenvalues[manifold_key].size
        Y1 = np.zeros((size,size,size,size))
        if self.num_vibrations == 0:
            return Y1
        spectral_density = self.SD['vibration_bath']
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

        if self.multiple_site_bath:
            size = self.eigenvalues[manifold_key].size
            Y2 = np.zeros((size, size, size, size))

            for n in range(self.PolyVib.Polymer.num_sites):
                spectral_density = self.SD['site_bath'][n]
                # site projector operators
                P_n = self.PolyVib.Polymer.occupied_list[n]
                P_n = self.extract_electronic_operator(P_n, manifold_key)

                P_n = np.kron(P_n, self.PolyVib.vibrational_identity)
                Y2 += self.make_dissipation_tensor(P_n, manifold_key, spectral_density)

            # create the electronic dissipation tensor for doubly-excited site
            if self.N >= 3:
                if 'site_bath2' in self.params['bath']:
                    for n in range(self.PolyVib.Polymer.num_sites):
                        spectral_density = self.SD['site_bath2'][n]
                        P_n2 = self.PolyVib.Polymer.doubly_occupied_list[n]
                        P_n2 = self.extract_electronic_operator(P_n2, manifold_key)

                        P_n2 = np.kron(P_n2, self.PolyVib.vibrational_identity)

                        Y2 += self.make_dissipation_tensor(P_n2, manifold_key, spectral_density)

            # create the electronic dissipation tensor for triply-excited site
            if self.N >= 4:
                if 'site_bath3' in self.params['bath']:
                    for n in range(self.PolyVib.Polymer.num_sites):
                        spectral_density = self.SD['site_bath3'][n]
                        P_n3 = self.PolyVib.Polymer.triply_occupied_list[n]
                        P_n3 = self.extract_electronic_operator(P_n3, manifold_key)

                        P_n3 = np.kron(P_n3, self.PolyVib.vibrational_identity)

                        Y2 += self.make_dissipation_tensor(P_n3, manifold_key, spectral_density)

        else :
            spectral_density = self.SD['site_bath']
            size = self.eigenvalues[manifold_key].size
            Y2 = np.zeros((size,size,size,size))

            for n in range(self.PolyVib.Polymer.num_sites):
                # site projector operators
                P_n = self.PolyVib.Polymer.occupied_list[n]
                P_n = self.extract_electronic_operator(P_n,manifold_key)

                P_n = np.kron(P_n,self.PolyVib.vibrational_identity)
                Y2 += self.make_dissipation_tensor(P_n,manifold_key,spectral_density)

            # create the electronic dissipation tensor for doubly-excited site
            if self.N >= 3:
                if 'site_bath2' in self.params['bath']:
                    spectral_density = self.SD['site_bath2']
                    for n in range(self.PolyVib.Polymer.num_sites):
                        P_n2 = self.PolyVib.Polymer.doubly_occupied_list[n]
                        P_n2 = self.extract_electronic_operator(P_n2,manifold_key)

                        P_n2 = np.kron(P_n2, self.PolyVib.vibrational_identity)

                        Y2 += self.make_dissipation_tensor(P_n2,manifold_key,spectral_density)

            # create the electronic dissipation tensor for triply-excited site
            if self.N >= 4:
                if 'site_bath3' in self.params['bath']:
                    spectral_density = self.SD['site_bath3']
                    for n in range(self.PolyVib.Polymer.num_sites):
                        P_n3 = self.PolyVib.Polymer.triply_occupied_list[n]
                        P_n3 = self.extract_electronic_operator(P_n3, manifold_key)

                        P_n3 = np.kron(P_n3, self.PolyVib.vibrational_identity)

                        Y2 += self.make_dissipation_tensor(P_n3, manifold_key, spectral_density)

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

        if self.N >= 3:
            if 'site_internal_conversion_bath21' in self.params['bath']:
                spectral_density = self.SD['site_internal_conversion_bath21']
                manifold_key = 'all_manifolds'

                for n in range(self.PolyVib.Polymer.num_sites):
                    up_n21 = self.PolyVib.Polymer.up21_list[n]
                    up_n21 = self.extract_electronic_operator(up_n21,manifold_key)
                    up_n21 = np.kron(up_n21,self.PolyVib.vibrational_identity)
                    sigma_x_n21 = up_n21 + up_n21.T

                    Y3 += self.make_dissipation_tensor(sigma_x_n21,manifold_key,spectral_density)

            if 'site_internal_conversion_bath20' in self.params['bath']:
                spectral_density = self.SD['site_internal_conversion_bath20']
                manifold_key = 'all_manifolds'

                for n in range(self.PolyVib.Polymer.num_sites):
                    up_n20 = self.PolyVib.Polymer.up20_list[n]
                    up_n20 = self.extract_electronic_operator(up_n20, manifold_key)
                    up_n20 = np.kron(up_n20, self.PolyVib.vibrational_identity)
                    sigma_x_n20 = up_n20 + up_n20.T

                    Y3 += self.make_dissipation_tensor(sigma_x_n20, manifold_key, spectral_density)

        if self.N >= 4:
            if 'site_internal_conversion_bath32' in self.params['bath']:
                spectral_density = self.SD['site_internal_conversion_bath32']
                manifold_key = 'all_manifolds'

                for n in range(self.PolyVib.Polymer.num_sites):
                    up_n32 = self.PolyVib.Polymer.up32_list[n]
                    up_n32 = self.extract_electronic_operator(up_n32, manifold_key)
                    up_n32 = np.kron(up_n32, self.PolyVib.vibrational_identity)
                    sigma_x_n32 = up_n32 + up_n32.T

                    Y3 += self.make_dissipation_tensor(sigma_x_n32, manifold_key, spectral_density)

            if 'site_internal_conversion_bath31' in self.params['bath']:
                spectral_density = self.SD['site_internal_conversion_bath31']
                manifold_key = 'all_manifolds'

                for n in range(self.PolyVib.Polymer.num_sites):
                    up_n31 = self.PolyVib.Polymer.up31_list[n]
                    up_n31 = self.extract_electronic_operator(up_n31, manifold_key)
                    up_n31 = np.kron(up_n31, self.PolyVib.vibrational_identity)
                    sigma_x_n31 = up_n31 + up_n31.T

                    Y3 += self.make_dissipation_tensor(sigma_x_n31, manifold_key, spectral_density)

            if 'site_internal_conversion_bath30' in self.params['bath']:
                spectral_density = self.SD['site_internal_conversion_bath30']
                manifold_key = 'all_manifolds'

                for n in range(self.PolyVib.Polymer.num_sites):
                    up_n30 = self.PolyVib.Polymer.up30_list[n]
                    up_n30 = self.extract_electronic_operator(up_n30, manifold_key)
                    up_n30 = np.kron(up_n30, self.PolyVib.vibrational_identity)
                    sigma_x_n30 = up_n30 + up_n30.T

                    Y3 += self.make_dissipation_tensor(sigma_x_n30, manifold_key, spectral_density)

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
                L_key = ket_man + ',' + bra_man
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
        old_key = str(i) + ',' + str(j)
        new_key = str(i2) + ',' + str(j2)
        
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
        old_key = str(i) + ',' + str(j)
        new_key = str(i2) + ',' + str(j2)
        
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



def spectral_density_matrix(SD,w):
    """Evaluates a spectral density on an array of frequencies, with the same
        treatment of w = 0 as the scalar call SD(w) that the Redfield
        constructors use element by element.
    Args:
        SD (callable): OhmicSpectralDensity, WhiteNoiseSpectralDensity or any
            callable taking a scalar frequency
        w (np.ndarray): frequencies (any shape)
    Returns:
        np.ndarray: SD(w) with the shape of w
"""
    w = np.asarray(w,dtype='float')
    flat = w.ravel()
    if isinstance(SD,OhmicSpectralDensity):
        out = np.empty(flat.size)
        zero = flat == 0
        out[zero] = SD.J(np.zeros(np.count_nonzero(zero))) * SD.kT
        nz = ~zero
        out[nz] = SD.J(flat[nz]) * SD.temp_dist(flat[nz])
    elif isinstance(SD,WhiteNoiseSpectralDensity):
        out = SD.C_vec(flat)
    else:
        out = np.array([SD(x) for x in flat],dtype='float')
    return out.reshape(w.shape)

class SecularRedfieldConstructor(RedfieldConstructor):
    """Secular Redfield theory that keeps every term coupling two density-matrix
        elements of matching Bohr frequency.

        A term R_pq, coupling rho_q = rho_kl to rho_p = rho_ij, is kept when
        the Bohr frequencies w_ij = e_i - e_j and w_kl = e_k - e_l
            (a) agree to within secular_tolerance (exact matches: populations,
                degenerate eigenstates, equally spaced ladders), or
            (b) differ by less than secular_cutoff times the size of the coupling
            between the two frequency classes (the Frobenius norm of the block
            of R connecting all elements of w_ij's class to all of w_kl's).
        The decision is taken per pair of frequency classes, so the trace and
        hermiticity of rho are preserved and the result does not depend on the
        choice of eigenvectors inside degenerate subspaces.

        Keeping all frequency-matched terms makes L independent of the choice
        of eigenvectors inside degenerate subspaces, and keeps the coherence-
        transfer terms of equally spaced ladders (e.g. rho_10 <- rho_21 for a
        harmonic mode).

        params.yaml, under 'bath':
            secular_cutoff (float): the factor in (b); default 1. Use 0 to keep
                only exact matches.
            secular_tolerance (float): the tolerance in (a), in energy units;
                default 1E-8 times the largest |eigenvalue| of H.

        For separable manifolds, couplings between optical coherences
        (ket and bra in different manifolds) keep the same terms as before
        (the bath acting on the ket or on the bra alone), now including the
        frequency-matched off-diagonal ones.
"""

    def set_secular_parameters(self):
        bath = self.params['bath']
        self.secular_cutoff = float(bath.get('secular_cutoff',1.0))
        e_max = max(np.max(np.abs(self.eigenvalues[key])) for key in self.manifolds)
        default_tol = 1E-8 * max(e_max,1.0)
        tol = bath.get('secular_tolerance',None)
        self.secular_tolerance = default_tol if tol is None else float(tol)

    def bath_coupling_terms(self,manifold_key):
        """List of (system-bath coupling operator, spectral density) for the
            given manifold, in the original (site) basis. Mirrors the
            operators used by RedfieldConstructor.make_Y1, make_Y2 and make_Y3.
"""
        terms = []

        # vibrations, bilinearly coupled to their bath
        if self.num_vibrations > 0:
            spectral_density = self.SD['vibration_bath']
            for p in range(self.PolyVib.num_vibrations):
                ad_p = self.PolyVib.total_ups[p]
                if manifold_key == 'all_manifolds':
                    x_p = (ad_p + ad_p.T)/np.sqrt(2)
                else:
                    ad_p_man = self.PolyVib.extract_vibronic_manifold(ad_p,int(manifold_key))
                    x_p = (ad_p_man + ad_p_man.T)/np.sqrt(2)
                terms.append((x_p,spectral_density))

        # electronic dephasing: projectors onto singly-, doubly-, triply-occupied sites
        def site_SD(key,n):
            if self.multiple_site_bath:
                return self.SD[key][n]
            return self.SD[key]

        def add_projectors(occupied_list,key):
            for n in range(self.PolyVib.Polymer.num_sites):
                P_n = occupied_list[n]
                P_n = self.extract_electronic_operator(P_n,manifold_key)
                P_n = np.kron(P_n,self.PolyVib.vibrational_identity)
                terms.append((P_n,site_SD(key,n)))

        add_projectors(self.PolyVib.Polymer.occupied_list,'site_bath')
        if self.N >= 3 and 'site_bath2' in self.params['bath']:
            add_projectors(self.PolyVib.Polymer.doubly_occupied_list,'site_bath2')
        if self.N >= 4 and 'site_bath3' in self.params['bath']:
            add_projectors(self.PolyVib.Polymer.triply_occupied_list,'site_bath3')

        # internal conversion (only possible when manifolds are not separated)
        if manifold_key == 'all_manifolds':
            def add_sigma_x(up_list,key):
                spectral_density = self.SD[key]
                for n in range(self.PolyVib.Polymer.num_sites):
                    up_n = up_list[n]
                    up_n = self.extract_electronic_operator(up_n,manifold_key)
                    up_n = np.kron(up_n,self.PolyVib.vibrational_identity)
                    terms.append((up_n + up_n.T,spectral_density))

            if 'site_internal_conversion_bath' in self.SD:
                add_sigma_x(self.PolyVib.Polymer.up_list,'site_internal_conversion_bath')
            if self.N >= 3:
                if 'site_internal_conversion_bath21' in self.params['bath']:
                    add_sigma_x(self.PolyVib.Polymer.up21_list,'site_internal_conversion_bath21')
                if 'site_internal_conversion_bath20' in self.params['bath']:
                    add_sigma_x(self.PolyVib.Polymer.up20_list,'site_internal_conversion_bath20')
            if self.N >= 4:
                if 'site_internal_conversion_bath32' in self.params['bath']:
                    add_sigma_x(self.PolyVib.Polymer.up32_list,'site_internal_conversion_bath32')
                if 'site_internal_conversion_bath31' in self.params['bath']:
                    add_sigma_x(self.PolyVib.Polymer.up31_list,'site_internal_conversion_bath31')
                if 'site_internal_conversion_bath30' in self.params['bath']:
                    add_sigma_x(self.PolyVib.Polymer.up30_list,'site_internal_conversion_bath30')

        return terms

    def set_Y(self):
        """For each manifold, stores the bath coupling operators in the
            Hamiltonian eigenbasis, A_b, with G_b = C_b * A_b (elementwise,
            C_b[i,k] = SD_b(e_k - e_i)) and T = sum_b A_b^dagger G_b. In terms of
            these the Redfield dissipation tensor of RedfieldConstructor is
            Y_ijkl = sum_b G_b[i,k] A_b^dagger[j,l] and its trace
            T_ik = sum_n Y_nikn; nothing of size N^4 is ever formed.
"""
        self.set_secular_parameters()
        self.Y = {}
        for manifold in self.manifolds:
            e = self.eigenvalues[manifold]
            v = self.eigenvectors[manifold]
            eki = e[np.newaxis,:] - e[:,np.newaxis]     # eki[i,k] = e_k - e_i
            G_list = []
            Adag_list = []
            T = np.zeros((e.size,e.size),dtype='complex')
            for CO, SD in self.bath_coupling_terms(manifold):
                A = np.conjugate(v.T).dot(CO).dot(v)
                Adag = np.conjugate(A.T)
                G = spectral_density_matrix(SD,eki) * A
                G_list.append(G)
                Adag_list.append(Adag)
                T += Adag.dot(G)
            if np.allclose(T.imag,0):
                T = T.real
            self.Y[manifold] = {'G':G_list,'Adag':Adag_list,'T':T}

    def _R_values(self,ket_manifold,bra_manifold,I,J,K,L):
        """Redfield tensor elements R_ijkl (rho_kl -> rho_ij) for arrays of
            indices, using the same formula as RedfieldConstructor.make_R_from_Y:
            R_ijkl = -(Y_ijkl + Y*_jilk) + delta_jl T_ik + delta_ik T*_jl.
            Between different manifolds (separable case) only the delta terms
            are present, as in RedfieldConstructor.make_R_from_Y_coherence.
"""
        Yk = self.Y[ket_manifold]
        Yb = self.Y[bra_manifold]
        R = np.zeros(I.size,dtype='complex')
        jl = J == L
        ik = I == K
        R[jl] += Yk['T'][I[jl],K[jl]]
        R[ik] += np.conjugate(Yb['T'][J[ik],L[ik]])
        if ket_manifold == bra_manifold:
            for G, Adag in zip(Yk['G'],Yk['Adag']):
                R -= G[I,K] * Adag[J,L]
                R -= np.conjugate(G[J,L] * Adag[I,K])
        return R

    def make_flat_R(self,ket_manifold,bra_manifold,*,chunk_size=int(2E6)):
        """Sparse Redfield tensor on the flattened (row-major, index i*n_bra+j)
            density matrix, keeping the frequency-matched terms described in the
            class docstring.
"""
        e_ket = self.eigenvalues[ket_manifold]
        e_bra = self.eigenvalues[bra_manifold]
        n_ket, n_bra = e_ket.size, e_bra.size
        size = n_ket * n_bra
        w = (e_ket[:,np.newaxis] - e_bra[np.newaxis,:]).ravel()
        tol = self.secular_tolerance
        kappa = self.secular_cutoff

        # frequency classes: elements whose Bohr frequencies agree within tol.
        # Elements of class c are order[c_start[c] : c_start[c] + c_size[c]]
        order = np.argsort(w,kind='stable')
        w_sorted = w[order]
        new_class = np.concatenate(([True],np.diff(w_sorted) > tol))
        cls_sorted = np.cumsum(new_class) - 1
        n_cls = cls_sorted[-1] + 1
        cls = np.empty(size,dtype='int64')
        cls[order] = cls_sorted
        c_size = np.bincount(cls_sorted,minlength=n_cls)
        c_start = np.concatenate(([0],np.cumsum(c_size)[:-1]))
        w_cls = np.bincount(cls_sorted,weights=w_sorted,minlength=n_cls) / c_size

        # largest possible |R_pq|; the Frobenius norm of the block between
        # classes of sizes m_p, m_q is at most R_bound * sqrt(m_p * m_q)
        Yk = self.Y[ket_manifold]
        Yb = self.Y[bra_manifold]
        R_bound = np.max(np.abs(Yk['T']),initial=0) + np.max(np.abs(Yb['T']),initial=0)
        if ket_manifold == bra_manifold:
            R_bound += 2 * sum(np.max(np.abs(G),initial=0) * np.max(np.abs(Ad),initial=0)
                               for G, Ad in zip(Yk['G'],Yk['Adag']))

        # candidate class pairs (cp <- cq): searched separately for each class
        # size, so that large classes (e.g. the populations) get a wide enough
        # window without widening it for everything else
        cps, cqs = [], []
        all_cls = np.arange(n_cls)
        for m in np.unique(c_size):
            bucket = np.where(c_size == m)[0]          # sorted by frequency
            window = kappa * R_bound * np.sqrt(m * c_size) + 2 * tol
            lo = np.searchsorted(w_cls[bucket],w_cls - window,side='left')
            hi = np.searchsorted(w_cls[bucket],w_cls + window,side='right')
            cnt = hi - lo
            cq = np.repeat(all_cls,cnt)
            offsets = np.arange(cq.size) - np.repeat(np.cumsum(cnt) - cnt,cnt)
            cp = bucket[np.repeat(lo,cnt) + offsets]
            cps.append(cp)
            cqs.append(cq)
        cp = np.concatenate(cps)
        cq = np.concatenate(cqs)
        dw = np.abs(w_cls[cp] - w_cls[cq])
        possible = (cp == cq) | (dw < kappa * R_bound * np.sqrt(c_size[cp] * c_size[cq]))
        cp, cq = cp[possible], cq[possible]

        # expand class pairs into element pairs, in chunks, and evaluate R
        n_pairs = c_size[cp] * c_size[cq]
        rows, cols, vals, grp = [], [], [], []
        start = 0
        while start < cp.size:
            cum = np.cumsum(n_pairs[start:])
            stop = start + max(1,int(np.searchsorted(cum,chunk_size,side='right')))
            stop = min(stop,cp.size)
            g = np.arange(start,stop)
            npair = n_pairs[g]
            G = np.repeat(g,npair)
            k = np.arange(G.size) - np.repeat(np.cumsum(npair) - npair,npair)
            mq = c_size[cq[G]]
            P = order[c_start[cp[G]] + k // mq]
            Q = order[c_start[cq[G]] + k % mq]
            I, J = np.divmod(P,n_bra)
            K, L = np.divmod(Q,n_bra)
            R = self._R_values(ket_manifold,bra_manifold,I,J,K,L)
            # drop exact zeros and rounding-level remainders (e.g. -2Y + 2T
            # for a single state), which the old secular construction gave as 0
            nz = np.abs(R) > 1E-14 * R_bound
            rows.append(P[nz])
            cols.append(Q[nz])
            vals.append(R[nz])
            grp.append(G[nz])
            start = stop
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        grp = np.concatenate(grp)

        # one decision per pair of frequency classes: keep the whole block when
        # the classes match, or when their frequency difference is below
        # secular_cutoff times the block's Frobenius norm. The norm is unchanged
        # by rotations inside degenerate subspaces, and keeping whole classes
        # preserves the trace and hermiticity of rho.
        g_norm = np.sqrt(np.bincount(grp,weights=np.abs(vals)**2,minlength=cp.size))
        g_keep = (cp == cq) | (np.abs(w_cls[cp] - w_cls[cq]) < kappa * g_norm)
        keep = g_keep[grp]
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        if np.allclose(vals.imag,0):
            vals = vals.real
        R = sp.csr_matrix((vals,(rows,cols)),shape=(size,size))
        return R

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

    def check_parallel(self,v1,v2):
        v1n = np.sqrt(np.dot(v1,v1))
        v2n = np.sqrt(np.dot(v2,v2))
        cos_theta = np.abs(np.dot(v1,v2)/v1n/v2n)
        if np.isclose(cos_theta-1,0,atol=np.spacing(1)):
            par = True
        else:
            par = False
        return par

    def orthonormalize(self,eigenvalues,left,right):
        """Orthogonalize the left and right eigenvectors with respect to 
            each other
"""
        ev = eigenvalues
        norms = np.einsum('ij,ji->i',left,right)
        if np.any(np.abs(norms) < 1E-12):
            raise Exception("""Overlap between right and left eigenvectors is
                                too small, L is not diagonalizable""")
        left = np.dot(np.diag(1/norms),left)
        lr = np.dot(left,right)
        lr_diagonal = np.diag(lr.diagonal())
        diff = lr - lr_diagonal
        tol = 1E-4
        if np.max(np.abs(diff)) > tol:
            num_digits = int(-np.log10(tol))
            diff = np.round(diff,num_digits)
            row_inds,col_inds = np.where(np.abs(diff) > tol)
            for ri,ci in zip(row_inds,col_inds):
                if np.isclose(ev[ri],ev[ci],atol=tol,rtol=tol):
                    if self.check_parallel(left[ri,:],left[ci,:]):
                        raise Exception("""Two left eigenvectors are 
                                            parallel; L is not
                                            diagonalizable""")
                else:
                    pass
                    # raise Exception("""Left and right eigenvectors are not 
                    #                     orthogonal, L is not 
                    #                     diagonalizable""")
            lr_sparse = csc_matrix(np.round(lr,num_digits))
            coefs = spsolve(lr_sparse,identity(lr.shape[0],format='csc'))
            left = coefs.dot(left)
            lr = np.dot(left,right)
            lr_diagonal = np.diag(lr.diagonal())
            diff = lr - lr_diagonal
            if np.max(np.abs(diff)) > tol:
                raise Exception("""Failed to orthogonalize left and right 
                                    eigenvectors""")

        return left, right

    def eig(self,L,*,check_eigenvectors = True,populations_only = False):
        
        eigvals, eigvecs_left, eigvecs = eig(L,left=True,right=True)

        eigvecs_left = np.conjugate(eigvecs_left.T)

        eigvecs_left, eigvecs = self.orthonormalize(eigvals,eigvecs_left,eigvecs)
        
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
            # print(np.max(np.abs(D-np.diag(eigvals))))
            if np.allclose(D,np.diag(eigvals),rtol=1E-10,atol=1E-10):
                pass
            else:
                warnings.warn('Using eigenvectors to diagonalize Liouvillian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(np.max(np.abs(D - np.diag(eigvals)))))

        return eigvals, eigvecs, eigvecs_left

    def eig2(self,L_key,*,check_eigenvectors = True):
        """Diagonalizes a secular Liouvillian block by block. L is split into
            the connected components of its off-diagonal couplings (the
            population block, and any groups of frequency-matched coherences);
            each component is diagonalized densely and the rest is already
            diagonal. Zero-eigenvalue (steady-state) eigenvectors are normalized
            to unit trace.
"""
        if L_key == 'all_manifolds':
            ket_key = 'all_manifolds'
            bra_key = 'all_manifolds'
        else:
            ket_key,bra_key = L_key.split(',')
        L = self.L[L_key]
        if L.dtype == np.dtype('O'):
            L = L[()]
        L = sp.csr_matrix(L)
        n = L.shape[0]
        E = np.array(L.diagonal(),dtype='complex')
        V = sp.eye(n,format='lil',dtype='complex')
        VL = sp.eye(n,format='lil',dtype='complex')

        pop_mask = np.zeros(n,dtype='bool')
        if ket_key == bra_key:
            H_size = int(np.sqrt(n))
            pop_mask[np.arange(H_size)*(H_size+1)] = True

        off = (L - sp.diags(L.diagonal())).tocsr()
        off.eliminate_zeros()
        num_blocks, labels = connected_components(off,directed=False)
        block_sizes = np.bincount(labels,minlength=num_blocks)
        for b in np.where(block_sizes > 1)[0]:
            inds = np.where(labels == b)[0]
            L_block = L[inds,:][:,inds].toarray()
            e, v, vl = self.eig(L_block,check_eigenvectors=check_eigenvectors,
                                populations_only=True)
            # eig normalizes steady states by the sum of all their entries;
            # renormalize by the sum over populations only (the trace)
            block_pops = pop_mask[inds]
            for m in np.where(np.isclose(e,0,atol=1E-12))[0]:
                trace = v[block_pops,m].sum()
                if block_pops.any() and np.abs(trace) > 0:
                    v[:,m] = v[:,m] / trace
                    vl[m,:] = vl[m,:] * trace
            E[inds] = e
            V[np.ix_(inds,inds)] = v
            VL[np.ix_(inds,inds)] = vl

        V = V.tocsr()
        VL = VL.tocsr()
        if check_eigenvectors:
            LV = L.dot(V)
            D = VL.dot(LV)
            check = D - sp.diags(E)
            if check.data.size > 0:
                max_check_value = np.max(np.abs(check.data))
                if max_check_value > 1E-10:
                    warnings.warn('Using eigenvectors to diagonalize Liouvillian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(max_check_value))

        return E,V,VL

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
                boolean_mu[:,:] = np.round(np.sqrt(np.sum(np.abs(new_mu)**2,axis=-1)),12)
                mu = mu * boolean_mu[:,:,np.newaxis]
                self.boolean_mu[key] = boolean_mu

    def save_mu(self):
        if self.prune:
            np.savez(os.path.join(self.save_path,'mu_pruned.npz'),**self.transformed_mu)
            np.savez(os.path.join(self.save_path,'mu_boolean.npz'),**self.boolean_mu)
        else:
            np.savez(os.path.join(self.save_path,'mu.npz'),**self.transformed_mu)
