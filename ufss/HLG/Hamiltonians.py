import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import eigs, eigsh
import itertools
from scipy.linalg import block_diag, eig, expm, eigh
import yaml
import os
import copy
from scipy.special import factorial, binom
from numpy.polynomial.hermite import hermval
import warnings
import time

class LadderOperators:
    """Constructs creation and annihilation operators for the SHO.
    
    This class is used to construct harmonic and anharmonic Hamiltonians.  
    Powers of x and p are created using ladder operators that are larger than
    the desired size of the truncated matrices, in order to avoid errors

    Attributes:
        size (int): size of returned operators
        calculation_size (int): size of internal operators
        a (np.ndarray): annihilation operator
        ad (np.ndarray): creation operator
        x (np.ndarray): position operator
        p (np.ndarray); momentum operator
        
"""
    def __init__(self,size,*,disp = 0, extra_size=10):
        """
        Args:
            size (int): desired size of returned operators
            disp (complex): complex phase-space displacement of the bottom
                of the potential energy surface
            extra_size (int): additional size used for calculating 
                matrix powers default value 10
"""
        self.size = size
        self.disp = disp
        self.calculation_size = size + extra_size
        self.set_a_and_ad()
        self.set_x_and_p()

    def destroy(self):
        """Constructs annihilation operator for dispalced SHO

        Returns:
            np.ndarray: annihilation operator
"""
        def offdiag1(n):
            return np.sqrt(n+1)

        n = np.arange(0,self.calculation_size)
        off1 = offdiag1(n[0:-1])
        a = - np.eye(self.calculation_size) * self.disp/np.sqrt(2)
        a += np.diag(off1,k=1)
        return a

    def create(self):
        """Constructs creation operator for displaced SHO

        Returns:
            np.ndarray: creation operator
"""
        a = self.destroy()
        ad = np.conjugate(a.transpose())
        return ad

    def set_a_and_ad(self):
        """Sets raising (a^dagger) and lowering (a) operators
"""
        self._a = self.destroy()
        self.a = self._a[:self.size,:self.size]

        self._ad = self.create()
        self.ad = self._ad[:self.size,:self.size]

    def set_x_and_p(self):
        """Sets the position (x) and momentum (p) operators using the 
            creation and annihilation operator methods.
"""
        self._x = (self._ad + self._a)/np.sqrt(2)
        self.x = self._x[:self.size,:self.size]
        
        self._p = 1j* (self._ad - self._a)/np.sqrt(2)
        self.p = self._p[:self.size,:self.size]

    def x_power_n(self,n):
        """Calculates x^n 

        Args:
            n (int): power

        Returns:
            np.ndarray: position operator x to the n power
"""
        xn = np.linalg.matrix_power(self._x,n)
        return xn[:self.size,:self.size]

    def p_power_n(self,n):
        """Calculates p^n 
        
        Args:
            n (int): power

        Returns:
            np.ndarray: momentum operator p to the n power
"""
        pn = np.linalg.matrix_power(self._p,n)
        return pn[:self.size,:self.size]
    
class DisplacedAnharmonicOscillator:
    """Hamiltonian for a displaced anharmonic oscillator
        
    This class can model all the same things that ArbitraryHamiltonian can.  
    However, it is much easier to work with when you need to displace an 
    anharmonic oscillator in x

    Attributes:
        constant (float): constant energy offset
        displacement (complex): complex phase-space displacement of the 
            bottom of the potential energy surface
        pcoef (list): coefficients for 2nd order and higher momentum terms
        xcoef (list): coefficients for 2nd order and higher position terms
        ham (np.ndarray): Hamiltonian
"""
    def __init__(self,size):
        """
        Args:
            size (int) : truncation size for Hamiltonian
"""
        self.size = size

    def set_ham(self,constant,displacement,pcoef,xcoef):
        """Creates anharmonic Hamiltonian for 1 vibrational mode

        Args:
            constant (complex) : complex reorganization energy
            displacement (complex) : phase-space displacement
            pcoef (list): coefficients for 2nd order + higher momentum terms
            xcoef (list): coefficients for 2nd order + higher position terms
"""
        extra_size = max(len(pcoef),len(xcoef))
        L = LadderOperators(self.size,disp=displacement,extra_size=extra_size)
        self.constant = constant
        self.displacement = displacement
        self.pcoef = pcoef
        self.xcoef = xcoef
        L.set_x_and_p()
        self.ham = np.eye(self.size,dtype=complex)*constant
        for n in range(0,len(pcoef)):
            self.ham += pcoef[n] * L.p_power_n(n+2)
        for m in range(0,len(xcoef)):
            self.ham += xcoef[m] * L.x_power_n(m+2)
            
        if np.allclose(np.imag(self.ham),0):
            self.ham = np.real(self.ham)

    def potential(self,x):
        """Potential energy surface for this Hamiltonian
        
        Args:
            x (np.ndarray): 1d-array of position values

        Returns:
            np.ndarray: 1d-array of potential energy as a function of x
"""
        y = np.ones(x.shape,x.dtype)*self.constant
        x0 = np.real(self.displacement)
        for i in range(len(self.xcoef)):
            y += np.power(x-x0,i+2) * self.xcoef[i]
        return y

    def plot(self,x):
        """Displays plot of eigenfunctions as a function of x

        Args:
            x (np.ndarray): 1d-array of position values
"""
        fig, ax = plt.subplots()
        ax.plot(x,self.potential(x))
        for n in range(len(self.eigvals)):
            ev = self.eigvals[n]
            vec = self.eigvecs[:,n]
            ax.axhline(y=ev,linestyle='--')
            f = oscillator_1d(vec,x)
            ax.plot(x,f**2+ev)

    def set_eigs(self,num,*,hermitian=True):
        """Use sparse eigensolver to find the bottom specified eigenvalues

        Args:
            num (int): number of eigenvalues to find
"""
        if hermitian:
            eigvals, eigvecs = eigsh(self.ham,k=num,which='SM')
        else:
            eigvals, eigvecs = eigs(self.ham,k=num,which='SM')
        sort_indices = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:,sort_indices]
        # I choose to pick the phase of my eigenvectors such that the state
        # which has the largest overlap has a positive overlap. For
        # sufficiently small d, and alpha close to 1, this will be the
        # overlap between the same excited and ground states.
        for i in range(num):
            max_index = np.argmax(np.abs(eigvecs[:,i]))
            if eigvecs[max_index,i] < 0:
                eigvecs[:,i] *= -1
        self.eigvecs = eigvecs
        if np.max(np.imag(eigvals)) > 1E-12:
            warnings.warn('Eigenvalues are not fully real')
            self.eigvals = eigvals
        else:
            self.eigvals = np.real(eigvals)

    def displaced_vacuum(self,*,disp = 1):
        """Create a displaced vacuum state relative to an unshifted potential
"""
        L = LadderOperators(self.size,disp=0)
        vac = np.zeros(L.calculation_size)
        vac[0] = 1
        disp_oper = expm(disp * L._ad + np.conjugate(disp) * L._a)
        disp_vac = disp_oper.dot(vac)
        disp_vac = disp_vac[:L.size]
        return disp_vac

    def time_evolution(self,state,time):
        """Evolves a state forward in time using thee eigensystem
"""
        basis_change = self.eigvecs
        state = np.conjugate(basis_change.T).dot(state)
        time_oper = np.exp(1j * self.eigvals * time)
        state = state * time_oper
        state = basis_change.dot(state)
        return state

    def animate_wavepacket(self,*,disp = 1,x=np.arange(-3,3,0.01),t=np.arange(0,2*np.pi,np.pi/30)):
        disp_vac = self.displaced_vacuum(disp = disp)
        fig, ax = plt.subplots()
        ax.plot(x,self.potential(x))
        fig.subplots_adjust(bottom=0.4)
        t_ax = fig.add_axes([0.1,0.05,0.65,0.02])
        t_slider = Slider(t_ax, 'time', 0, t[-1],valinit=0)

        psi = oscillator_1d(disp_vac,x)

        line_r, line_i, line_abs, = ax.plot(x,np.real(psi)*2+2,
                                            x,np.imag(psi)*2+2,
                                            x,np.abs(psi)*2+2)
        
        def update(val):
            t = t_slider.val
            psi = oscillator_1d(self.time_evolution(disp_vac,t),x)
            line_r.set_ydata(np.real(psi)*2+2)
            line_i.set_ydata(np.imag(psi)*2+2)
            line_abs.set_ydata(np.abs(psi)*2+2)
        t_slider.on_changed(update)

class Polymer:
    """Creates the Hamiltonian and other useful operators describing a 
        Polymer of two-level systems (2LS)
    
    Attributes:
        num_sites (int): number of two level systems
        energies (list): excitation energy of each 2LS if it were isolated
        couplings (list): energetic couplings between singly-
            excited electronic states in the site basis, ordered as
            [J12,J13,...,J1N,J23,...,J2N,...]
        pols (list): set to value ['x','y','z']
        N (int): size of each 2LS (set to value 2)
        up (np.ndarray): raising operator for a 2LS
        down (np.ndarray): lowering operator for a 2LS
        ii (np.ndarray): identity operator for a 2LS
        occupied (np.ndarray): occupation operator for a 2LS
        empty (np.ndarray): ii - occupied
"""

    def __init__(self,site_energies,site_couplings,dipoles):
        """This initializes an object with an arbitrary number of
        coupled two-level systems

        Args:
            site_energies (list): excitation energies of individual sites
            site_couplings (list): energetic couplings between singly-
                excited electronic states in the site basis, ordered as
                [J12,J13,...,J1N,J23,...,J2N,...]
"""
        self.num_sites = len(site_energies)
        self.energies = site_energies
        self.couplings = site_couplings
        self.dipoles = dipoles
        self.pols = ['x','y','z']

        ### Operators that define a single two-level system (2LS)
        self.N = 2 # one of the key ingedients that makes this a 2LS

        self.up = self.basis(1,0)

        self.down = self.basis(0,1)

        self.ii = np.eye(self.N)

        self.occupied = self.basis(1,1)

        self.empty = self.basis(0,0)

        ### Kron up the single 2LS operators to act on the full Hilbert space of the polymer

        self.set_up_list()
        self.set_down_list()
        self.set_occupied_list()
        self.set_empty_list()
        self.set_exchange_list()

        ### Make Hamiltonian (total and by manifold)
        
        self.set_electronic_hamiltonian()
        self.set_electronic_total_occupation_number()

        self.make_mu_dict_site_basis()
        self.make_mu_site_basis('x')
        self.make_mu_up_dict_site_basis()
        self.make_mu_up_site_basis('x')

        self.set_manifold_eigensystems()
        self.set_electronic_eigensystem()
        

    def basis(self,row,col):
        """Matrix basis for a single two-level system
"""
        b = np.zeros((self.N,self.N))
        b[row,col] = 1
        return b

    ### Tools for making the basic operators in the polymer space

    def electronic_identity_kron(self,element_list):
        """Takes in a list of tuples (element, position) and krons
            them up into the full space (element is assumed to be
            a 2x2 array acting on a single 2LS.  Any positions that
            are unspecified are assumed to be identities
        Args:
            element_list (list): list of tuples

        Returns:
            2^n x 2^n array acting on the full space of the polymer
"""
        # all unspecified positions will be taken to be identities
        num_identities = self.num_sites - len(element_list)
        if num_identities < 0:
            # cannot have more operators than 2LS's
            raise ValueError('Too many elements for Hilbert space')
        # create a list of all identities
        matrix_list = [self.ii for j in range(self.num_sites)]

        for el, pos in element_list:
            # for each position specified, replace identity in
            # matrix_list with element
            matrix_list[pos] = el
        return self.recursive_kron(matrix_list)

    def recursive_kron(self,list_of_matrices):
        """Krons together a list of matrices
        Args:
            list_of_matrices (list): list of np.ndarrays
        Returns:
            the kronecker product of all the input matrices
"""
        mat = list_of_matrices.pop(0)
        n = len(list_of_matrices)
        for next_item in list_of_matrices:
            mat = np.kron(mat,next_item)
        return mat

    def make_single_operator_list(self,O):
        """Make a list of full-space operators for a given 2x2 operator,
            by taking tensor product with identities on the other 
            excitations
"""
        O_list = []
        for i in range(self.num_sites):
            Oi = self.electronic_identity_kron([(O,i)])
            O_list.append(Oi)
        return O_list

    def make_multi_operator_list(self,o_list):
        """Make a list of full-space operators for a given set of 2x2 
            operators by inserting the necessary identities
"""
        O_list = []
        positions = itertools.combinations(range(self.num_sites),len(o_list))
        for pos_tuple in positions:
            Oi = self.electronic_identity_kron(list(zip(o_list,pos_tuple)))
            O_list.append(Oi)
        return O_list

    def set_occupied_list(self):
        self.occupied_list = self.make_single_operator_list(self.occupied)

    def set_up_list(self):
        self.up_list = self.make_single_operator_list(self.up)

    def set_down_list(self):
        self.down_list = self.make_single_operator_list(self.down)

    def set_empty_list(self):
        self.empty_list = self.make_single_operator_list(self.empty)
    
    def set_exchange_list(self):
        self.exchange_list = self.make_multi_operator_list([self.up,self.down])

    ### Tools for moving back and forth between full Hamiltonian and manifold(s)

    def electronic_vector_of_ones_kron(self,position,item):
        n = self.num_sites
        ones_list = [np.ones(self.N) for i in range(n-1)]
        ones_list.insert(position,item)
        vec = ones_list.pop(0)
        for next_item in ones_list:
            vec = np.kron(vec,next_item)
        return vec

    def set_electronic_total_occupation_number(self):
        n = self.num_sites
        single_mode_occ = np.arange(self.N)
        occ_num = self.electronic_vector_of_ones_kron(0,single_mode_occ)
        for i in range(1,n):
            occ_num += self.electronic_vector_of_ones_kron(i,single_mode_occ)
        self.electronic_total_occupation_number = occ_num

    def electronic_manifold_mask(self,manifold_num):
        """Creates a boolean mask to describe which states obey the truncation
           size collectively
"""
        manifold_inds = np.where(self.electronic_total_occupation_number == manifold_num)[0]
        return manifold_inds

    def electronic_subspace_mask(self,min_occ_num,max_occ_num):
        """Creates a boolean mask to describe which states obey the range of 
            electronic occupation collectively
"""
        manifold_inds = np.where((self.electronic_total_occupation_number >= min_occ_num) &
                                 (self.electronic_total_occupation_number <= max_occ_num))[0]
        return manifold_inds

    def extract_coherence(self,O,manifold1,manifold2):
        """Returns result of projecting the Operator O onto manifold1
            on the left and manifold2 on the right
"""
        manifold1_inds = self.electronic_manifold_mask(manifold1)
        manifold2_inds = self.electronic_manifold_mask(manifold2)
        O = O[manifold1_inds,:]
        O = O[:,manifold2_inds]
        return O
    
    def extract_manifold(self,O,manifold_num):
        """Projects operator into the given electronic excitation manifold
"""
        return self.extract_coherence(O,manifold_num,manifold_num)

    def coherence_to_full(self,O,manifold1,manifold2):
        """Creates an array of zeros of the size of the full Hilbert space,
            and fills the correct entries with the operator O existing in
            a particular optical coherence between manifolds
"""
        Ofull = np.zeros(self.electronic_hamiltonian.shape,dtype=O.dtype)
        manifold1_inds = self.electronic_manifold_mask(manifold1)
        manifold2_inds = self.electronic_manifold_mask(manifold2)
        for i in range(manifold2_inds.size):
            ind = manifold2_inds[i]
            Ofull[manifold1_inds,ind] = O[:,i]
        return Ofull
    
    def manifold_to_full(self,O,manifold_num):
        """Creates an array of zeros of the size of the full Hilbert space,
            and fills the correct entries with the operator O existing in
            a single optical manifold
"""
        return self.coherence_to_full(O,manifold_num,manifold_num)

    def extract_electronic_subspace(self,O,min_occ_num,max_occ_num):
        """Projects operator into the given electronic excitation manifold(s)
"""
        manifold_inds = self.electronic_subspace_mask(min_occ_num,max_occ_num)
        O = O[manifold_inds,:]
        O = O[:,manifold_inds]
        return O

    
    ### Tools for making the Hamiltonian

    def make_electronic_hamiltonian(self):
        ham = self.energies[0] * self.occupied_list[0]
        for i in range(1,self.num_sites):
            ham += self.energies[i] * self.occupied_list[i]

        for i in range(len(self.exchange_list)):
            ham += self.couplings[i] * self.exchange_list[i]
            ham += np.conjugate(self.couplings[i]) * self.exchange_list[i].T

        return ham

    def set_electronic_hamiltonian(self):
        self.electronic_hamiltonian = self.make_electronic_hamiltonian()

    def get_electronic_hamiltonian(self,*,manifold_num = 'all'):
        if manifold_num == 'all':
            return self.electronic_hamiltonian
        else:
            return self.extract_manifold(self.electronic_hamiltonian,manifold_num)
    

    ### Tools for diagonalizing the Hamiltonian

    def make_manifold_eigensystem(self,manifold_num):
        h = self.get_electronic_hamiltonian(manifold_num = manifold_num)
        e, v = np.linalg.eigh(h)
        sort_inds = e.argsort()
        e = e[sort_inds]
        v = v[:,sort_inds]
        return e,v

    def set_manifold_eigensystems(self):
        self.electronic_eigenvalues_by_manifold = []
        self.electronic_eigenvectors_by_manifold = []
        for i in range(self.num_sites+1):
            e,v = self.make_manifold_eigensystem(i)
            self.electronic_eigenvalues_by_manifold.append(e)
            self.electronic_eigenvectors_by_manifold.append(v)

    def get_eigensystem_by_manifold(self,manifold_num):
        e = self.electronic_eigenvalues_by_manifold[manifold_num]
        v = self.electronic_eigenvectors_by_manifold[manifold_num]
        return e,v

    def set_electronic_eigensystem(self):
        H = self.electronic_hamiltonian
        eigvecs = np.zeros(H.shape)
        d = np.zeros(H.shape)
        for i in range(self.num_sites+1):
            e,v = self.get_eigensystem_by_manifold(i)
            if i ==1:
                self.exciton_energies = e
            eigvecs += self.manifold_to_full(v,i)
            d += self.manifold_to_full(np.diag(e),i)
        Hd = eigvecs.T.dot(H.dot(eigvecs))
        if np.allclose(Hd,d):
            pass
        else:
            raise Exception('Diagonalization by manifold failed')

        self.electronic_eigenvectors = eigvecs
        self.electronic_eigenvalues = d.diagonal()
    

    ### Tools for making the dipole operator
        
    def make_mu_site_basis(self,pol):
        pol_dict = {'x':0,'y':1,'z':2}
        d = self.dipoles[:,pol_dict[pol]]
        self.mu = d[0]*(self.up_list[0] + self.down_list[0])
        for i in range(1,len(self.up_list)):
            self.mu += d[i]*(self.up_list[i] + self.down_list[i])

    def make_mu_dict_site_basis(self):
        self.mu_dict = dict()
        for pol in self.pols:
            self.make_mu_site_basis(pol)
            self.mu_dict[pol] = self.mu.copy()

    def make_mu_up_site_basis(self,pol):
        pol_dict = {'x':0,'y':1,'z':2}
        d = self.dipoles[:,pol_dict[pol]]
        self.mu_ket_up = self.up_list[0].copy()*d[0]
        for i in range(1,len(self.up_list)):
            self.mu_ket_up += self.up_list[i]*d[i]

    def make_mu_up_dict_site_basis(self):
        self.mu_up_dict = dict()
        for pol in self.pols:
            self.make_mu_up_site_basis(pol)
            self.mu_up_dict[pol] = self.mu_ket_up.copy()
            

class PolymerVibrations():
    def __init__(self,yaml_file,*,mask_by_occupation_num=True,
                 separable_manifolds = 'auto'):
        """Initial set-up involves unpacking the vibrational_frequencies, 
            which must be passed as a nested list. Each site may have N 
            vibrational modes, and each has a frequency, a displacement
            and a frequency shift for the excited state for sites a, b, ...
"""
        with open(yaml_file) as yamlstream:
            params = yaml.load(yamlstream,Loader=yaml.SafeLoader)
        self.base_path = os.path.split(yaml_file)[0]

        self.save_path = os.path.join(self.base_path,'closed')
        os.makedirs(self.save_path,exist_ok=True)

        self.Polymer = Polymer(params['site_energies'],params['site_couplings'],np.array(params['dipoles']))
        
        self.truncation_size = params['initial truncation size']
        try:
            self.maximum_manifold = params['maximum_manifold']
        except:
            self.maximum_manifold = np.inf
        
        self.maximum_manifold = min(self.maximum_manifold,
                                    self.Polymer.num_sites)

        if separable_manifolds == 'auto':
            try:
                int_conv_bath = params['bath']['site_internal_conversion_bath']
                self.separable_manifolds = False
            except KeyError:
                try:
                    opt_rel = params['bath']['optical_relaxation_gamma']
                    self.separable_manifolds = False
                except KeyError:
                    self.separable_manifolds = True

        else:
            self.separable_manifolds = separable_manifolds
        
        self.params = params
        
        self.occupation_num_mask = mask_by_occupation_num
        self.set_vibrations()
        self.set_vibrational_ladder_operators()
        
        e_ham = self.Polymer.extract_electronic_subspace(self.Polymer.electronic_hamiltonian,0,self.maximum_manifold)
            
        self.total_hamiltonian = np.kron(e_ham,self.vibrational_identity)
        self.add_vibrations()

        self.set_H()
        self.save_H()
        
        self.make_condon_mu()

        self.save_mu()

    def set_H(self):
        self.H = {}
        if self.separable_manifolds:
            for i in range(self.maximum_manifold+1):
                H = self.extract_vibronic_manifold(self.total_hamiltonian,i)
                self.H[str(i)] = H
        else:
            self.H['all_manifolds'] = self.total_hamiltonian

    def save_H(self):
        np.savez(os.path.join(self.save_path,'H.npz'),**self.H)

    def vibrational_occupation_to_indices(self,vibration,occ_num,
                                          manifold_num):
        single_mode_occ = np.arange(self.truncation_size)
        vib_occ = self.vibrational_vector_of_ones_kron(vibration,
                                                       single_mode_occ)
        masked_single_mode_occ = vib_occ[self.vibrational_mask]

        electronic_manifold_hamiltonian = self.Polymer.get_electronic_hamiltonian(manifold_num = manifold_num)
        elec_size = electronic_manifold_hamiltonian.shape[0]
        
        masked_single_mode_occ = np.kron(np.ones(elec_size),
                                         masked_single_mode_occ)
        return np.where(masked_single_mode_occ == occ_num)[0]

    def electronic_occupation_to_indices(self,site_num,manifold_num):
        single_mode_occ = np.arange(2)
        elec_occ = self.Polymer.electronic_vector_of_ones_kron(site_num,
                                                       single_mode_occ)
        mask = self.Polymer.electronic_manifold_mask(manifold_num)
        masked_elec_occ = elec_occ[mask]
        masked_elec_occ = np.kron(masked_elec_occ,
                                  np.ones(self.vibrational_size))

        return np.where(masked_elec_occ == 1)[0]

    def vibronic_manifold_mask(self,manifold_num):
        """Gets the indices of the Hilbert space that occupy a particular 
            electronic manifold, including all vibrational degrees of 
            freedom from that manifold
"""
        vib_size = self.vibrational_size
        vib_ones = np.ones(vib_size,dtype='int')
        elec_occ = self.Polymer.electronic_total_occupation_number
        elec_inds = np.where(elec_occ <= self.maximum_manifold)[0]
        elec_occ = elec_occ[elec_inds]
        vibronic_occupation_number = np.kron(elec_occ,vib_ones)
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
    
    def add_vibrations(self):
        v0 = self.empty_vibrations
        v1 = self.occupied_vibrations
        self.vibrational_hamiltonian = np.zeros(self.total_hamiltonian.shape)
        for i in range(len(v0)):
            self.vibrational_hamiltonian += v0[i]
            self.vibrational_hamiltonian += v1[i]

        self.total_hamiltonian = self.total_hamiltonian + self.vibrational_hamiltonian

    def add_linear_couplings(self):
        self.linear_coupling_hamiltonian = np.zeros(self.total_hamiltonian.shape)

        for i in range(self.num_vibrations):
            try:
                coupling_list = self.params['vibrations'][i]['linear_couplings']
                self.linear_coupling_hamiltonian += self.linear_site_vibrational_couplings(i,manifold_num)
            except KeyError:
                pass
        self.total_hamiltonian = self.total_hamiltonian + self.linear_coupling_hamiltonian

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
            self.vibrational_size = N**nv
            self.vibrational_mask = (np.arange(N**nv),)
            self.vibrational_identity = np.eye(N**nv)
        empty_vibrations = self.kron_up_vibrations(emp_vibs)
        occupied_vibrations = self.kron_up_vibrations(occ_vibs)

        self.empty_vibrations = []
        self.occupied_vibrations = []
        
        for i in range(self.num_vibrations):
            site_index = vibration_params[i]['site_label']
            empty = self.Polymer.empty_list[site_index]
            empty = self.Polymer.extract_electronic_subspace(empty,0,self.maximum_manifold)
            occupied = self.Polymer.occupied_list[site_index]
            occupied = self.Polymer.extract_electronic_subspace(occupied,0,self.maximum_manifold)
            
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

    def linear_site_vibrational_couplings(self,mode_num):
        mode_dict = self.params['vibrations'][mode_num]
        coupling_list = mode_dict['linear_couplings']
        coupling_strength, site_index_pair = coupling_list[0]
        vib_coupling = self.single_linear_site_vibration_coupling(site_index_pair,mode_num)
        coupling_ham =  vib_coupling * coupling_strength
        
        for i in range(1,len(coupling_list)):
            coupling_strength, site_index_pair = coupling_list[i]
            vib_coupling = self.single_linear_site_vibration_coupling(site_index_pair,mode_num,manifold_num)
            coupling_ham +=  vib_coupling * coupling_strength
        return coupling_ham
    
    def single_linear_site_vibration_coupling(self,site_index_pair,mode_num,manifold_num):
        electronic_matrix = self.site_basis(0,manifold_num)
        electronic_matrix[:,:] = 0
        i,j = site_index_pair
        electronic_matrix[i,j] = 1
        electronic_matrix[j,i] = 1

        x = LadderOperators(self.truncation_size).x
        x = x[:self.truncation_size,:self.truncation_size]
        vib_coupling = self.vibration_identity_kron(mode_num,x,manifold_num)
        return kron(electronic_matrix,vib_coupling)

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
        if nv == 0:
            single_mode_occ = np.array([0])
        else:
            single_mode_occ = np.arange(N)
        occ_num = self.vibrational_vector_of_ones_kron(0,single_mode_occ)
        for i in range(1,nv):
            occ_num += self.vibrational_vector_of_ones_kron(i,single_mode_occ)
        self.vibrational_total_occupation_number = occ_num
        self.vibrational_mask = np.where(occ_num < N)
        self.vibrational_size = self.vibrational_mask[0].size
        self.vibrational_identity = np.eye(self.vibrational_size)

    def construct_vibrational_hamiltonian(self,single_mode,electronic_occupation):
        """For each vibrational mode, construct a list of sparse matrices 
            defining the vibrational hamiltonian for that mode in each 
            excited state"""
        w = single_mode['omega_g']
        lam = single_mode['reorganization'][electronic_occupation]
        d  = single_mode['displacement'][electronic_occupation]
        kin = single_mode['kinetic'][electronic_occupation]
        pot = single_mode['potential'][electronic_occupation]
        aho = DisplacedAnharmonicOscillator(self.truncation_size)
        aho.set_ham(lam,d,kin,pot)
        return 0.5 * w * aho.ham

    def construct_vibrational_ladder_operator(self,single_mode,electronic_occupation):
        """Construct ladder operator given the electronic occupation for 
            that site"""
        w = single_mode['omega_g']
        d  = single_mode['displacement'][electronic_occupation]
        L = LadderOperators(self.truncation_size,disp=d,extra_size=1)
        return L.ad

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
        self.total_ups = []
        
        for i in range(self.num_vibrations):
            site_index = vibration_params[i]['site_label']
            empty = self.Polymer.empty_list[site_index]
            empty = self.Polymer.extract_electronic_subspace(empty,0,self.maximum_manifold)
            occupied = self.Polymer.occupied_list[site_index]
            occupied = self.Polymer.extract_electronic_subspace(occupied,0,self.maximum_manifold)
            
            self.empty_ups.append(np.kron(empty,empty_ups[i]))
            self.occupied_ups.append(np.kron(occupied,occupied_ups[i]))

            self.total_ups.append(self.empty_ups[i] + self.occupied_ups[i])

    def make_condon_mu(self):
        H_shape = self.total_hamiltonian.shape
        mu_shape = (*H_shape,3)
        self.mu = np.zeros(mu_shape)
        self.mu_ket_up = np.zeros(mu_shape)
        for i in range(3):
            pol = self.Polymer.pols[i]
            elec_mu = self.Polymer.mu_dict[pol]
            mu = self.Polymer.extract_electronic_subspace(elec_mu,0,self.maximum_manifold)
            self.mu[:,:,i] = np.kron(mu,self.vibrational_identity)

            elec_mu_up = self.Polymer.mu_up_dict[pol]
            mu_up = self.Polymer.extract_electronic_subspace(elec_mu_up,0,self.maximum_manifold)
            
            self.mu_ket_up[:,:,i] = np.kron(mu_up,self.vibrational_identity)

    def save_mu_by_manifold(self):
        mu_dict = {}
        for i in range(self.maximum_manifold):
            j = i+1
            mu = self.extract_vibronic_coherence(self.mu[:,:,0],j,i)
            mu_shape = (*mu.shape,3)
            mu3d = np.zeros(mu_shape)
            mu3d[:,:,0] = mu[:,:]
            mu3d[:,:,1] = self.extract_vibronic_coherence(self.mu[:,:,1],j,i)
            mu3d[:,:,2] = self.extract_vibronic_coherence(self.mu[:,:,2],j,i)
            key = str(i) + '_to_' + str(j)
            mu_dict[key] = mu3d

        np.savez(os.path.join(self.save_path,'mu_original_H_basis.npz'),**mu_dict)

    def save_mu_all_manifolds(self):
        mu_up = self.mu_ket_up
        mu_dict = {'up':mu_up}

        np.savez(os.path.join(self.save_path,'mu_original_H_basis.npz'),**mu_dict)

    def save_mu(self):
        if self.separable_manifolds:
            self.save_mu_by_manifold()
        else:
            self.save_mu_all_manifolds()
        

class DiagonalizeHamiltonian:

    def __init__(self,file_path,*,qr_flag=False,check_eigenvectors = True):
        self.base_path = file_path
        self.load_path = os.path.join(file_path,'closed')
        self.save_path = self.load_path
        self.qr_flag = qr_flag
        self.r_mats = {}
        self.check_eigenvectors = check_eigenvectors

        self.load_H()
        self.load_mu()

        self.eigenvalues = {}
        self.eigenvectors = {}
        self.H_diagonalization_time = {}

        self.mu_eigen_basis = {}

        self.diagonalize()
        self.transform_mu()
        self.save_eigsystem()
        self.save_mu()

    def load_H(self):
        H_save_name = os.path.join(self.load_path,'H.npz')
        with np.load(H_save_name) as H_archive:
            self.manifolds = list(H_archive.keys())
            self.H = {key:H_archive[key] for key in self.manifolds}

    def load_mu(self):
        mu_save_name = os.path.join(self.load_path,'mu_original_H_basis.npz')
        with np.load(mu_save_name) as mu_archive:
            self.mu_keys = list(mu_archive.keys())
            self.mu = {key:mu_archive[key] for key in self.mu_keys}

    def save_timing(self):
        np.savez(os.path.join(self.save_path,'diagonalization_timings.npz')
                 ,**self.H_diagonalization_time)

    def eig(self,manifold_key):
        """Diagonalzie the requested manifold

        Args:
            manifold_key (str): options: 'all_manifolds' or '0', '1', ...

        Returns:
            tuple: (e, v) where e are the eigenvalues and v is a square array
                of eigenvectors arranged in columns
"""
        ham = self.H[manifold_key]
        eigvals, eigvecs = eigh(ham)
        
        if self.qr_flag:
            # Force degenerate eigenvectors to be orthogonal
            eigvecs, r = np.linalg.qr(eigvecs,mode='reduced')
            
        if self.check_eigenvectors:
            HV = ham.dot(eigvecs)
            D = eigvecs.T.dot(HV)
            if np.allclose(D,np.diag(eigvals),rtol=1E-11,atol=1E-11):
                pass
            else:
                warnings.warn('Using eigenvectors to diagonalize hamiltonian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(np.max(np.abs(D - np.diag(eigvals)))))
        
        sort_indices = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:,sort_indices]
        if self.qr_flag:
            r = r[:,sort_indices]
            self.r_mats[manifold_key] = r
        # I choose to pick the phase of my eigenvectors such that the state
        # which has the largest magnitude overlap is positive. For
        # sufficiently small d, and alpha close to 1, this will be the
        # overlap between the same excited and ground states.
        for i in range(eigvals.size):
            max_index = np.argmax(np.abs(eigvecs[:,i]))
            if eigvecs[max_index,i] < 0:
                eigvecs[:,i] *= -1

        return eigvals, eigvecs

    def diagonalize(self):
        for manifold in self.manifolds:
            t0 = time.time()
            e, v = self.eig(manifold)
            t1 = time.time()
            self.H_diagonalization_time[manifold] = t1-t0
            self.eigenvalues[manifold] = e
            self.eigenvectors[manifold] = v
        return None

    def mu_key_to_manifold_keys(self,key):
        if self.manifolds[0] == 'all_manifolds':
            starting_key = 'all_manifolds'
            ending_key = 'all_manifolds'
        else:
            starting_key, ending_key = key.split('_to_')
        return starting_key, ending_key

    def transform_mu(self):
        for key in self.mu_keys:
            old_key, new_key = self.mu_key_to_manifold_keys(key)
            
            v_old = self.eigenvectors[old_key]
            v_new = self.eigenvectors[new_key]
            mu_site_basis = self.mu[key]
            mu_t = np.zeros(mu_site_basis.shape)
            for i in range(3):
                mu_t[...,i] = v_new.T.dot(mu_site_basis[...,i]).dot(v_old)
            self.mu_eigen_basis[key] = mu_t
        return None

    def save_eigsystem(self):
        e_save_name = os.path.join(self.save_path,'eigenvalues.npz')
        v_save_name = os.path.join(self.save_path,'eigenvectors.npz')
        np.savez(e_save_name,**self.eigenvalues)
        np.savez(v_save_name,**self.eigenvectors)
        return None

    def save_mu(self):
        mu_save_name = os.path.join(self.save_path,'mu.npz')
        np.savez(mu_save_name,**self.mu_eigen_basis)
        return None


def SHO_basis(n,xvalues):
    coef = [0 for i in range(n)]
    coef.append(1)
    norm = 1/(np.pi**(1/4)*2**(n/2) * np.sqrt(factorial(n)))
    return  norm * np.exp(-xvalues**2/2) * hermval(xvalues,coef)

def oscillator_1d(array_1d,x):
    ans = np.zeros(x.shape,dtype=array_1d.dtype)
    for i in range(len(array_1d)):
        ans += SHO_basis(i,x) * array_1d[i]
    return ans
