import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import eigs, eigsh
import itertools
from scipy.linalg import block_diag, eig, expm, eigh
import yaml
import copy
from scipy.special import factorial, binom
from numpy.polynomial.hermite import hermval
import warnings

class LadderOperators:
    def __init__(self,size):
        self.size = size
        self.calculation_size = size + 20
        self.set_x_and_p()

    def set_x_and_p(self):
        self.x = (self.create() + self.destroy())/np.sqrt(2)
        self.p = 1j* (-self.create() + self.destroy())/np.sqrt(2)

    def create(self):
        def offdiag1(n):
            return np.sqrt(n+1)

        n = np.arange(0,self.calculation_size)
        off1 = offdiag1(n[0:-1])
        ham = np.zeros((self.calculation_size,self.calculation_size))
        ham += np.diag(off1,k=-1)
        return ham

    def destroy(self):
        def offdiag1(n):
            return np.sqrt(n+1)

        n = np.arange(0,self.calculation_size)
        off1 = offdiag1(n[0:-1])
        ham = np.zeros((self.calculation_size,self.calculation_size))
        ham += np.diag(off1,k=1)
        return ham

    def x_power_n(self,n):
        xn = np.linalg.matrix_power(self.x,n)
        return xn[:self.size,:self.size]

    def p_power_n(self,n):
        pn = np.linalg.matrix_power(self.p,n)
        return pn[:self.size,:self.size]
    
class ArbitraryHamiltonian(LadderOperators):
    def set_ham(self,constant,pcoef,xcoef,*,real=True):
        self.constant = constant
        self.pcoef = pcoef
        self.xcoef = xcoef
        ham = np.diag(np.ones(self.calculation_size,dtype=complex)*constant)
        for n in range(len(pcoef)):
            ham += pcoef[n] * np.linalg.matrix_power(self.p,n+1)
        for m in range(len(xcoef)):
            ham += xcoef[m] * np.linalg.matrix_power(self.x,m+1)
        self.ham = ham[:self.size,:self.size]
        if real:
            self.ham = np.real(self.ham)

    def set_eigs(self,num,*,hermitian=True):
        if hermitian:
            eigvals, eigvecs = eigsh(self.ham,k=num,which='SM')
        else:
            eigvals, eigvecs = eigs(self.ham,k=num,which='SM')
        sort_indices = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:,sort_indices]
        # I choose to pick the phase of my eigenvectors such that the state which has the
        # largest overlap has a positive overlap. For sufficiently small d, and alpha close
        # to 1, this will be the overlap between the same excited and ground states.
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

    def potential(self,x):
        y = np.ones(x.shape,x.dtype)*self.constant
        for i in range(len(self.xcoef)):
            y += np.power(x,i+1) * self.xcoef[i]
        return y

    def plot(self,*,x=np.arange(-3,3,.01)):
        fig, ax = plt.subplots()
        ax.plot(x,self.potential(x))
        for n in range(len(self.eigvals)):
            ev = self.eigvals[n]
            vec = self.eigvecs[:,n]
            ax.axhline(y=ev,linestyle='--')
            f = oscillator_1d(vec,x)
            ax.plot(x,f**2+ev)
            # ax.plot(x,np.real(f)+ev,x,np.imag(f)+ev)

    def displaced_vacuum(self,*,disp = 1):
        vac = np.zeros(self.calculation_size)
        vac[0] = 1
        disp_oper = expm(disp * self.create() + np.conjugate(disp) * self.destroy())
        disp_vac = disp_oper.dot(vac)
        disp_vac = disp_vac[:len(self.eigvals)]
        return disp_vac

    def time_evolution(self,state,time):
        basis_change = self.eigvecs[:len(self.eigvals),:len(self.eigvals)]
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

class AnharmonicDisplaced(ArbitraryHamiltonian):
    """Implements displacement in x for all powers. Does not explicitly treat a linear coupling.
        This class can model all the same things that ArbitraryHamiltonian can.  However, it is
        much easier to work with when you need to displace an anharmonic oscillator in x
"""
    def __init__(self,size):
        """
        Args:
            size (int) : truncation size for Hamiltonian
"""
        self.size = size

    def set_ham(self,constant,displacement,pcoef,xcoef,*,real=True):
        """Creates anharmonic Hamiltonian for 1 vibrational mode
        Args:
            constant (complex) : complex reorganization energy
            displacement (complex) : phase-space displacement
            pcoef (list) : coefficients for 2nd order and higher momentum terms
            xcoef (list) : coefficients for 2nd order and higher position terms
"""
        self.constant = constant
        self.p0 = np.imag(displacement)
        self.x0 = np.real(displacement)
        self.pcoef = pcoef
        self.xcoef = xcoef
        extra_size = max(len(pcoef),len(xcoef))
        # Calculates powers of x and p via matrix powers. Must use an increased
        # base truncation size in order to correctly resolve x^n and p^n
        self.calculation_size = self.size + extra_size
        self.set_x_and_p()
        ham = np.diag(np.ones(self.calculation_size,dtype=complex)*constant)
        for n in range(0,len(pcoef)):
            p = self.p - np.diag(np.ones(self.calculation_size))*self.p0
            ham += pcoef[n] * np.linalg.matrix_power(p,n+2)
        for m in range(0,len(xcoef)):
            x = self.x - np.diag(np.ones(self.calculation_size))*self.x0
            ham += xcoef[m] * np.linalg.matrix_power(x,m+2)
        self.ham = ham[:self.size,:self.size]
        if real:
            self.ham = np.real(self.ham)

    def potential(self,x):
        y = np.ones(x.shape,x.dtype)*self.constant
        for i in range(len(self.xcoef)):
            y += np.power(x-self.x0,i+2) * self.xcoef[i]
        return y

    def plot(self,*,x=np.arange(-3,3,.01)):
        fig, ax = plt.subplots()
        ax.plot(x,self.potential(x))
        for n in range(len(self.eigvals)):
            ev = self.eigvals[n]
            vec = self.eigvecs[:,n]
            ax.axhline(y=ev,linestyle='--')
            f = oscillator_1d(vec,x)
            ax.plot(x,f**2+ev)
    
class Polymer:
    """This class creates a matrix for an electronic polymer in site basis.
It is hard-coded to always set the ground-state energy to 0.  This version 
requires more inputs, and does not do additional calculations, as compared
to PolymerOld"""
    def __init__(self,site_energies,site_couplings,*,auto_DEM=True):
        """This initializes an object with an arbitrary number of site
energies and couplings

site_energies - list of excitation energies of individual sites

site_couplings - list of energetic couplings between singly-excited electronic 
states in the site basis, for example [J12,J13,...,J1N,J23,...,J2N,...]
"""
        try:
            self.num_sites = len(site_energies[0])
            self.energies = site_energies
            self.couplings = site_couplings
        except TypeError:    
            self.num_sites = len(site_energies)
            self.energies = [[0.0],site_energies]
            self.couplings = [[],site_couplings]
        if self.num_sites > 1 and auto_DEM:
            self.couplings.append([])
            DEM_ham = self.electronic_manifold_2()
            self.energies.append(DEM_ham.diagonal())
        

    def electronic_manifold_0(self):
        return np.array([[0.0]])
    
    def electronic_manifold_1(self):
        """n-excited manifold"""
        J_iter = iter(self.couplings[1])
        ham = np.diag(np.array(self.energies[1],dtype=float))
        N = len(self.energies[1])
        for i in range(0,N-1):
            for j in range(i+1,N):
                ham[i,j] = next(J_iter)
                ham[j,i] = ham[i,j]
        return ham

    def electronic_manifold_2(self):
        SEM_ham = self.electronic_manifold_1()
        DEM_indices = list(itertools.combinations(range(len(self.energies[1])),2))
        DEM_energies = np.array([self.energies[1][i] + self.energies[1][j] for (i,j) in DEM_indices],dtype=float)
        DEM_ham = np.diag(DEM_energies)
        couplings_list = []
        for (i,j) in itertools.combinations(range(len(DEM_energies)),2):
            inds_i = set(DEM_indices[i])
            inds_j = set(DEM_indices[j])
            diff = tuple(inds_i ^ inds_j)
            if len(diff) == 2:
                DEM_ham[i,j] = SEM_ham[diff]
                DEM_ham[j,i] = DEM_ham[i,j]
                couplings_list.append(DEM_ham[i,j])
        self.couplings[2] = couplings_list
        return DEM_ham

    def electronic_manifold(self,n):
        if n == 0:
            return self.electronic_manifold_0()
        elif n == 1:
            return self.electronic_manifold_1()
        elif n == 2:
            return self.electronic_manifold_2()
        else:
            print('Cannot construct higher excited manifolds')
    
    def electronic_manifold_eig(self,n):
        ham = self.electronic_manifold(n)
        eigvals, eigvecs = eig(ham)
        eigvals = np.real(eigvals)
        sort_ind = eigvals.argsort()
        eigvals = eigvals[sort_ind]
        eigvecs = eigvecs[:,sort_ind]
        return eigvals, eigvecs
        
    def total_electronic_hamiltonian(self):
        """Only returns the GSM, SEM, and DEM"""
        return block_diag(*[self.electronic_manifold(n) for n in range(3)])

class PolymerVibrations(Polymer):
    def __init__(self,yaml_file,*,qr_flag=False,mask_by_occupation_num=False):
        """Initial set-up is the same as for the Polymer class, but I also need
to unpack the vibrational_frequencies, which must be passed as a nested list.
Each site may have N vibrational modes, and each has a frequency, a displacement
and a frequency shift for the excited state
for sites a, b, ...

qr_flag - defaults to False. If True, applies a qr factorization to each set of 
eigenvalues. This forces the eigenvalues to be orthogonal, in the case of
degenerate subspaces. If the hamiltonian is not hermitian, this will break
the eigenvalues (since they are not expected to be orthogonal).
"""
        with open(yaml_file) as yamlstream:
            params = yaml.load(yamlstream,Loader=yaml.SafeLoader)
        try:
            auto_DEM = params['auto_DEM']
        except KeyError:
            auto_DEM = True
        super().__init__(params['site_energies'],params['site_couplings'],auto_DEM=auto_DEM)
        self.truncation_size = params['initial truncation size']
        self.params = params
        self.occupation_num_mask = mask_by_occupation_num
        self.set_vibrations()
        self.qr_flag = False
        self.r_mats = []
        self.check_eigenvectors = False

    def vector_of_ones_kron(self,position,item):
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
        occ_num = self.vector_of_ones_kron(0,single_mode_occ)
        for i in range(1,nv):
            occ_num += self.vector_of_ones_kron(i,single_mode_occ)
        self.vibrational_total_occupation_number = occ_num

    def set_truncation_mask(self):
        """Creates a boolean mask to describe which states obey the truncation
           size collectively
"""
        N = self.truncation_size
        self.manifold_indices = []
        for i in range(len(self.energies)):
            num_excitations = len(self.energies[i])
            total_occ_num = np.kron(np.ones(num_excitations),self.vibrational_total_occupation_number)
            inds_to_keep = np.where(total_occ_num < N)
            self.manifold_indices.append(inds_to_keep)

    def set_vibrations(self):
        vibration_params = self.params['vibrations']
        # Vibrations in the ground manifold are assumed to be diagonal
        GSM_vib = [self.construct_vibrational_hamiltonians(mode_dict,0)
                           for mode_dict in vibration_params]
        
        self.num_vibrations = len(GSM_vib)
        
        # Vibrations in the SEM
        SEM_vib= [self.construct_vibrational_hamiltonians(mode_dict,1)
                           for mode_dict in vibration_params]

        self.vibrations = [GSM_vib,SEM_vib]
        
        # Vibrations in the DEM
        try:
            DEM_vib = [self.construct_vibrational_hamiltonians(mode_dict,2)
                           for mode_dict in vibration_params]
            self.vibrations.append(DEM_vib)
        except KeyError:
            pass

        if self.occupation_num_mask:
            self.set_vibrational_total_occupation_number()
            self.set_truncation_mask()
        
    def construct_vibrational_hamiltonians(self,single_mode,manifold_num):
        """For each vibrational mode, construct a list of sparse matrices defining the 
            vibrational hamiltonian for that mode in each excited state"""
        w = single_mode['omega_g']
        lams = single_mode['reorganization'+str(manifold_num)]
        ds  = single_mode['displacement'+str(manifold_num)]
        kins = single_mode['kinetic'+str(manifold_num)]
        pots = single_mode['potential'+str(manifold_num)]
        hams = []
        for l, d, kin, pot in zip(lams,ds,kins,pots):
            aho = AnharmonicDisplaced(self.truncation_size)
            aho.set_ham(l,d,kin,pot,real=True)
            hams.append(0.5 * w * aho.ham)
        return hams
    
    def site_basis(self,n,manifold_num):
        """This defines the site basis for the electronic space"""
        basis_matrix = np.zeros((len(self.energies[manifold_num]),len(self.energies[manifold_num])))
        basis_matrix[n,n] = 1
        return basis_matrix

    def single_site_vibrations(self,n,manifold_num):
        """This defines the hamiltonian on a single excited state given all vibrational modes"""
        vibrations_on_site = []
        for vibrational_mode in self.vibrations[manifold_num]:
            site_ham = vibrational_mode[n]
            vibrations_on_site.append(site_ham)
        total_ham = self.vibration_identity_kron(0,vibrations_on_site[0],manifold_num)
        for i in range(1,len(self.vibrations[manifold_num])):
            total_ham += self.vibration_identity_kron(i,vibrations_on_site[i],manifold_num)
        return kron(self.site_basis(n,manifold_num), total_ham)

    def linear_site_vibrational_couplings(self,mode_num,manifold_num):
        mode_dict = self.params['vibrations'][mode_num]
        coupling_list = mode_dict['linear_couplings'+str(manifold_num)]
        coupling_strength, site_index_pair = coupling_list[0]
        vib_coupling = self.single_linear_site_vibration_coupling(site_index_pair,mode_num,manifold_num)
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
    
    def vibration_identity_kron(self,position,item,manifold_num):
        """Takes in a single vibrational hamiltonians and krons it with the correct 
            number of vibrational identities, inserting it into its position as indexed by its mode
            position as specified in the input file"""
        identities = [identity(self.truncation_size) for n in
                      range(len(self.vibrations[manifold_num])-1)]
        identities.insert(position,item)
        mat = identities.pop(0)
        for next_item in identities:
            mat = kron(mat,next_item)
        return mat
    
    def manifold_hamiltonian(self,manifold_num):
        """Creates requested hamiltonian
        Args:
            manifold_num (int): manifold - 0,1,2 correspond to GSM, SEM, DEM respectively
        Returns:
            (csr_matrix): hamiltonian of requested manifold as a sparse matrix
"""
        elec_manifold = self.electronic_manifold(manifold_num)
        vibrational_identity = identity(self.truncation_size)
        total_vibrational_identity = self.vibration_identity_kron(0,vibrational_identity,manifold_num)
        total_ham = kron(elec_manifold,total_vibrational_identity)
        for i in range(len(self.energies[manifold_num])):
            total_ham += self.single_site_vibrations(i,manifold_num)

        for i in range(self.num_vibrations):
            try:
                coupling_list = self.params['vibrations'][i]['linear_couplings'+str(manifold_num)]
                total_ham += self.linear_site_vibrational_couplings(i,manifold_num)
            except KeyError:
                pass
        if self.occupation_num_mask:
            inds = self.manifold_indices[manifold_num]
            total_ham = total_ham[inds[0]]
            total_ham = total_ham.transpose()
            total_ham = total_ham[inds[0]]
            total_ham = total_ham.transpose()
        
        return total_ham

    def eigs(self,num_eigvals,manifold_num):
        """Returns the requested number of eigvals, num_eigvals, sorted from smallest to largest,
            along with the corresponding eigenvectors reshaped into matrices with indices as follows:
            eigmat[electronic site, vibration 1, vibration 2, ..., eigen number], where eigen number is the
            index of the corresponding eigenvalue, starting from 0."""
        num_sites = len(self.energies[manifold_num])
        ham = self.manifold_hamiltonian(manifold_num)
        eigvals, eigvecs = eigsh(ham,k=num_eigvals*num_sites,which='SM')
        # Force degenerate eigenvectors to be orthogonal
        if self.qr_flag:
            eigvecs, r = np.linalg.qr(eigvecs,mode='reduced')
        if self.check_eigenvectors:
            HV = ham.dot(eigvecs)
            D = eigvecs.T.dot(HV)
            if np.allclose(D,np.diag(eigvals),rtol=1E-11,atol=1E-11):
                pass
            else:
                # warnings.warn('Eigenvalues altered by QR factorization, max absolute change in diagonal matrix of {}'.format(np.max(D-np.diag(eigvals))))
                warnings.warn('Using eigenvectors to diagonalize hamiltonian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(np.max(np.abs(D - np.diag(eigvals)))))
        
        sort_indices = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:,sort_indices]
        if self.qr_flag:
            r = r[:,sort_indices]
            self.r_mats.append(r)
        # I choose to pick the phase of my eigenvectors such that the state which has the
        # largest overlap has a positive overlap. For sufficiently small d, and alpha close
        # to 1, this will be the overlap between the same excited and ground states.
        for i in range(num_eigvals):
            max_index = np.argmax(np.abs(eigvecs[:,i]))
            if eigvecs[max_index,i] < 0:
                eigvecs[:,i] *= -1

        return eigvals, eigvecs

    def eig(self,manifold_num):
        """Returns the requested number of eigvals, num_eigvals, sorted from smallest to largest,
            along with the corresponding eigenvectors reshaped into matrices with indices as follows:
            eigmat[electronic site, vibration 1, vibration 2, ..., eigen number], where eigen number is the
            index of the corresponding eigenvalue, starting from 0."""
        num_sites = len(self.energies[manifold_num])
        ham = self.manifold_hamiltonian(manifold_num).toarray()
        eigvals, eigvecs = eigh(ham)
        # Force degenerate eigenvectors to be orthogonal
        if self.qr_flag:
            eigvecs, r = np.linalg.qr(eigvecs,mode='reduced')
        if self.check_eigenvectors:
            HV = ham.dot(eigvecs)
            D = eigvecs.T.dot(HV)
            if np.allclose(D,np.diag(eigvals),rtol=1E-11,atol=1E-11):
                pass
            else:
                # warnings.warn('Eigenvalues altered by QR factorization, max absolute change in diagonal matrix of {}'.format(np.max(D-np.diag(eigvals))))
                warnings.warn('Using eigenvectors to diagonalize hamiltonian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(np.max(np.abs(D - np.diag(eigvals)))))
        
        sort_indices = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:,sort_indices]
        if self.qr_flag:
            r = r[:,sort_indices]
            self.r_mats.append(r)
        # I choose to pick the phase of my eigenvectors such that the state which has the
        # largest overlap has a positive overlap. For sufficiently small d, and alpha close
        # to 1, this will be the overlap between the same excited and ground states.
        for i in range(eigvals.size):
            max_index = np.argmax(np.abs(eigvecs[:,i]))
            if eigvecs[max_index,i] < 0:
                eigvecs[:,i] *= -1

        return eigvals, eigvecs

    def serial_eigs(self,num_eigvals,manifold_num):
        vib_params = self.params['vibrations']
        N = len(vib_params) # number of vibrations
        omega_gs = np.array([vib_params[i]['omega_g'] for i in range(N)])
        diag_site_energies, electronic_eigvecs = self.electronic_manifold_eig(manifold_num)
        num_sites = diag_site_energies.size
        
        # if manifold_num == 0:
        #     alpha_aves = np.array([1 for i in range(N)])
        # elif manifold_num == 1:
        #     alpha_aves = np.array([np.mean(vib_params[i]['alphas1']) for i in range(N)])
        # elif manifold_num ==2:
        #     alpha_aves = np.array([np.mean(vib_params[i]['alphas2']) for i in range(N)])
        # w_ave = np.average(omega_gs,weights=alpha_aves)
        w_ave = np.average(omega_gs)

        num_eigs_found = 0
        quantum_number = 0
        energies_and_densities = []
        while num_eigs_found < num_eigvals:
            approx_vib_energy = w_ave * (quantum_number + N*0.5)
            approx_energies = diag_site_energies + approx_vib_energy
            number_of_nearby_vib_states = int(binom(quantum_number + N - 1,quantum_number)) + 2
            for en in approx_energies:
                energies_and_densities.append((en,number_of_nearby_vib_states))

            num_eigs_found += number_of_nearby_vib_states * num_sites
            quantum_number += 1

        ham = self.manifold_hamiltonian(manifold_num)

        eigenvalues = []
        eigenvectors = []
            
        for en, num_eigs in energies_and_densities:
            eigvals, eigvecs = eigsh(ham,k=num_eigs,which='LM',sigma=en + 1E-8)
            eigenvalues.append(eigvals)
            eigenvectors.append(eigvecs)
        print(energies_and_densities)
        print(eigenvalues)
        eigenvalues, eigenvectors = self.remove_redundant_eigenvectors(eigenvalues,eigenvectors)
        if len(eigenvalues) == 1:
            eigenvalues = eigenvalues[0][:num_eigvals]
            eigenvectors = eigenvectors[0][:,:num_eigvals]
        else:
            warnings.warn('Failure')

        for i in range(num_eigvals):
            max_index = np.argmax(np.abs(eigenvectors[:,i]))
            if eigenvectors[max_index,i] < 0:
                eigenvectors[:,i] *= -1
                
        return eigenvalues, eigenvectors
                

    def remove_redundant_eigenvectors(self,eigenvalues,eigenvectors):
        all_eigenvalues = [eigenvalues[0]]
        all_eigenvectors = [eigenvectors[0]]
        last_eigenvalues = eigenvalues[0]
        overlap_status = np.zeros(len(eigenvalues)-1,dtype=bool)
        for i in range(1,len(eigenvalues)):
            next_eigenvalues = eigenvalues[i]
            next_eigenvectors = eigenvectors[i]
            mask = np.ones(next_eigenvalues.size,dtype=bool)
            for n in range(last_eigenvalues.size):
                for m in range(next_eigenvalues.size):
                    if np.abs(last_eigenvalues[n] - next_eigenvalues[m]) < 1E-10:
                        mask[m] = False
                        overlap_status[i-1] = True
            next_eigenvalues = next_eigenvalues[mask]
            next_eigenvectors = next_eigenvectors[:,mask]
            if overlap_status[i-1]:
                all_eigenvalues[-1] = np.hstack((all_eigenvalues[-1],next_eigenvalues))
                all_eigenvectors[-1] = np.hstack((all_eigenvectors[-1],next_eigenvectors))
            else:
                all_eigenvalues.append(next_eigenvalues)
                all_eigenvectors.append(next_eigenvectors)

            last_eigenvalues = next_eigenvalues
            last_eigenvectors = next_eigenvectors

        return all_eigenvalues, all_eigenvectors
            

    def vibrationless_basis(self,manifold_num):
        elec_manifold = self.electronic_manifold(manifold_num)
        eigvals, eigvecs = eig(elec_manifold)
        sort_ind = eigvals.argsort()
        eigvals = eigvals[sort_ind]
        eigvecs = eigvecs[:,sort_ind]
        if eigvecs[0,0] < 0:
            eigvecs[:,0] *= -1
        if eigvecs[0,1] < 0:
            eigvecs[:,1] *= -1
        return eigvals, eigvecs

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
    
def oscillator_2d(array_2d,x1,x2):
    ans = np.zeros(x1.shape)
    l1, l2 = array_2d.shape
    for i in range(l1):
        for j in range(l2):
            ans += SHO_basis(i,x1) * SHO_basis(j,x2) * array_2d[i,j]
    return ans

def oscillator_2d_old(array_2d,x1,x2,*,rel_tol=1E-4):
    ans = np.zeros(x1.shape)
    flat_sum_ind = np.abs(array_2d.flatten()).argsort()[::-1]
    inds1, inds2 = np.unravel_index(flat_sum_ind,array_2d.shape)
    for i,j in zip(inds1, inds2):
        old_ans = ans
        ans += SHO_basis(i,x1) * SHO_basis(j,x2) * array_2d[i,j]
        if np.max(np.abs((old_ans - ans)/ans)) < rel_tol:
            break
    return ans

def plot_dimer_2vib(eigval, eigmat):
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_axes([0.17,0.1,0.3,0.9])
    ax2 = fig.add_axes([0.55,0.1,0.3,0.9])
    cax = fig.add_axes([0.05,0.25,0.02,0.4])
    cax2 = fig.add_axes([0.9,0.25,0.02,0.4])
    ax1.set_title('Site a')
    ax2.set_title('Site b')
    fig.suptitle('E = {:.4f}'.format(eigval),fontsize=16)
    mat_a = eigmat[0,...]
    mat_b = eigmat[1,...]
    x1_1d = np.arange(-3,3,.05)
    x1, x2 = np.meshgrid(x1_1d, x1_1d)
    site_a = oscillator_2d(mat_a,x1,x2)
    site_b = oscillator_2d(mat_b,x1,x2)
    im1 = ax1.contourf(x1,x2,site_a)
    im2 = ax2.contourf(x1,x2,site_b)
    fig.colorbar(im1,cax=cax)
    fig.colorbar(im2,cax=cax2)
    ax1.set_xlabel('$q_1$',fontsize=20)
    ax1.set_ylabel('$q_2$',fontsize=20)
    ax2.set_xlabel('$q_1$',fontsize=20)
    ax2.set_ylabel('$q_2$',fontsize=20)
    ax1.set(adjustable='box-forced', aspect='equal')
    ax2.set(adjustable='box-forced', aspect='equal')
    x2a_max, x1a_max = np.unravel_index(np.argmax(np.abs(site_a)),x1.shape)
    print('Excitation on site a is centered at (q1,q2) = ({0:.2f},{1:.2f})'.format(x1_1d[x1a_max],
                                                                           x1_1d[x2a_max]))
    x2b_max, x1b_max = np.unravel_index(np.argmax(np.abs(site_b)),x1.shape)
    print('Excitation on site b is centered at (q1,q2) = ({0:.2f},{1:.2f})'.format(x1_1d[x1b_max],
                                                                           x1_1d[x2b_max]))

def plot_dimer_2vib_b(dimer_obj,eigen_index,*,fig = None):
    if fig == None:
        fig = plt.figure(figsize=(12,5))
    eigval = dimer_obj.eigenvalues[1][eigen_index]
    eigmat = dimer_obj.eigenmatrices[1][...,eigen_index]
    ax1 = fig.add_axes([0.17,0.1,0.3,0.9])
    ax2 = fig.add_axes([0.55,0.1,0.3,0.9])
    cax = fig.add_axes([0.05,0.25,0.02,0.4])
    cax2 = fig.add_axes([0.9,0.25,0.02,0.4])
    ax1.set_title('Site a')
    ax2.set_title('Site b')
    fig.suptitle('E = {:.2f}'.format(eigval),fontsize=16)
    mat_a = eigmat[0,...]
    mat_b = eigmat[1,...]
    x1_1d = np.arange(-3,3,.05)
    x1, x2 = np.meshgrid(x1_1d, x1_1d)
    site_a = oscillator_2d(mat_a,x1,x2)
    site_b = oscillator_2d(mat_b,x1,x2)
    max_val = np.max((site_a,site_b))
    min_val = np.min((site_a,site_b))
    im1 = ax1.pcolor(x1,x2,site_a,vmin=min_val,vmax=max_val)
    # ax1.contour(x1,x2,site_a)
    im2 = ax2.pcolor(x1,x2,site_b,vmin=min_val,vmax=max_val)
    # ax2.contour(x1,x2,site_b)
    d1a = dimer_obj.params['vibrations'][0]['displacements1'][0]
    d1b = dimer_obj.params['vibrations'][0]['displacements1'][1]
    wg1 = dimer_obj.params['vibrations'][0]['omega_g']
    we1a = wg1 * dimer_obj.params['vibrations'][0]['alphas1'][0]
    we1b = wg1 * dimer_obj.params['vibrations'][0]['alphas1'][1]
    wg2 = dimer_obj.params['vibrations'][1]['omega_g']
    d2a = dimer_obj.params['vibrations'][1]['displacements1'][0]
    d2b = dimer_obj.params['vibrations'][1]['displacements1'][1]
    wg2 = dimer_obj.params['vibrations'][1]['omega_g']
    we2a = wg2 * dimer_obj.params['vibrations'][1]['alphas1'][0]
    we2b = wg2 * dimer_obj.params['vibrations'][1]['alphas1'][1]
    ax1.axvline(x=d1a,c='k',ls='--')
    ax1.axhline(y=d2a,c='k',ls='--')
    ax2.axvline(x=d1b,c='k',ls='--')
    ax2.axhline(y=d2b,c='k',ls='--')
    
    fig.colorbar(im1,cax=cax)
    fig.colorbar(im2,cax=cax2)
    cax.clear()
    cax.axis('off')
    # ax1.set_xlabel('$q_1, \omega={:.2f}$'.format(we1a),fontsize=20)
    # ax1.set_ylabel('$q_2, \omega={:.2f}$'.format(we2a),fontsize=20)
    # ax2.set_xlabel('$q_1, \omega={:.2f}$'.format(we1b),fontsize=20)
    # ax2.set_ylabel('$q_2, \omega={:.2f}$'.format(we2b),fontsize=20)
    ax1.set_xlabel('$R_1$')
    ax1.set_ylabel('$R_2$')
    ax2.set_xlabel('$R_1$')
    ax2.set_ylabel('$R_2$')
    
    ax1.set(adjustable='box-forced', aspect='equal')
    ax2.set(adjustable='box-forced', aspect='equal')
    x2a_max, x1a_max = np.unravel_index(np.argmax(np.abs(site_a)),x1.shape)
    print('Excitation on site a is centered at (q1,q2) = ({0:.2f},{1:.2f})'.format(x1_1d[x1a_max],
                                                                           x1_1d[x2a_max]))
    x2b_max, x1b_max = np.unravel_index(np.argmax(np.abs(site_b)),x1.shape)
    print('Excitation on site b is centered at (q1,q2) = ({0:.2f},{1:.2f})'.format(x1_1d[x1b_max],
                                                                           x1_1d[x2b_max]))


def plot_dimer_2vib_alpha_beta(dimer_obj,eigen_index,*,fig = None):
    if fig == None:
        fig = plt.figure(figsize=(12,5))
    eigval = dimer_obj.eigenvalues[1][eigen_index]
    eigmat = dimer_obj.eigenmatrices[1][...,eigen_index]
    eigvals, eigvecs = dimer_obj.vibrationless_basis(1)
    alpha = eigvecs[:,0]
    beta = eigvecs[:,1]
    
    ax1 = fig.add_axes([0.17,0.1,0.3,0.9])
    ax2 = fig.add_axes([0.55,0.1,0.3,0.9])
    cax = fig.add_axes([0.05,0.25,0.02,0.4])
    cax2 = fig.add_axes([0.9,0.25,0.02,0.4])
    ax1.set_title(r'State $\alpha$')
    ax2.set_title(r'State $\beta$')
    fig.suptitle('E = {:.4f}'.format(eigval),fontsize=16)
    mat_alpha = eigmat[0,...]*alpha[0] + eigmat[1,...]*alpha[1]
    mat_beta = eigmat[0,...]*beta[0] + eigmat[1,...]*beta[1]
    x1_1d = np.arange(-3,3,.05)
    x1, x2 = np.meshgrid(x1_1d, x1_1d)
    site_a = oscillator_2d(mat_alpha,x1,x2)
    site_b = oscillator_2d(mat_beta,x1,x2)
    max_val = np.max((site_a,site_b))
    min_val = np.min((site_a,site_b))
    im1 = ax1.pcolor(x1,x2,site_a,vmin=min_val,vmax=max_val)
    # ax1.contour(x1,x2,site_a)
    im2 = ax2.pcolor(x1,x2,site_b,vmin=min_val,vmax=max_val)
    # ax2.contour(x1,x2,site_b)
    d1a = dimer_obj.params['vibrations'][0]['displacements1'][0]
    d1b = dimer_obj.params['vibrations'][0]['displacements1'][1]
    wg1 = dimer_obj.params['vibrations'][0]['omega_g']
    we1a = wg1 * dimer_obj.params['vibrations'][0]['alphas1'][0]
    we1b = wg1 * dimer_obj.params['vibrations'][0]['alphas1'][1]
    wg2 = dimer_obj.params['vibrations'][1]['omega_g']
    d2a = dimer_obj.params['vibrations'][1]['displacements1'][0]
    d2b = dimer_obj.params['vibrations'][1]['displacements1'][1]
    wg2 = dimer_obj.params['vibrations'][1]['omega_g']
    we2a = wg2 * dimer_obj.params['vibrations'][1]['alphas1'][0]
    we2b = wg2 * dimer_obj.params['vibrations'][1]['alphas1'][1]
    ax1.axvline(x=d1a,c='k',ls='--')
    ax1.axhline(y=d2a,c='k',ls='--')
    ax2.axvline(x=d1b,c='k',ls='--')
    ax2.axhline(y=d2b,c='k',ls='--')
    
    fig.colorbar(im1,cax=cax)
    fig.colorbar(im2,cax=cax2)
    ax1.set_xlabel('$q_1, \omega={}$'.format(we1a),fontsize=20)
    ax1.set_ylabel('$q_2, \omega={}$'.format(we2a),fontsize=20)
    ax2.set_xlabel('$q_1, \omega={}$'.format(we1b),fontsize=20)
    ax2.set_ylabel('$q_2, \omega={}$'.format(we2b),fontsize=20)
    ax1.set(adjustable='box-forced', aspect='equal')
    ax2.set(adjustable='box-forced', aspect='equal')
    x2a_max, x1a_max = np.unravel_index(np.argmax(np.abs(site_a)),x1.shape)
    # print('Excitation on site a is centered at (q1,q2) = ({0:.2f},{1:.2f})'.format(x1_1d[x1a_max],
                                                                           # x1_1d[x2a_max]))
    x2b_max, x1b_max = np.unravel_index(np.argmax(np.abs(site_b)),x1.shape)
    # print('Excitation on site b is centered at (q1,q2) = ({0:.2f},{1:.2f})'.format(x1_1d[x1b_max],
                                                                           # x1_1d[x2b_max]))
    # print('Site alpha contribution: {}'.format(np.sum(mat_alpha.flatten()**2)))
    # print('Site beta contribution: {}'.format(np.sum(mat_beta.flatten()**2)))


def plot_dimer_2vib_single_plot(eigval, eigmat):
    fig, ax = plt.subplots()
    ax.set_title('Probability density \n electronic basis traced out, E = {:.4f}'.format(eigval),fontsize=16)
    mat_a = eigmat[0,...]
    mat_b = eigmat[1,...]
    x1_1d = np.arange(-3,3,.05)
    x1, x2 = np.meshgrid(x1_1d, x1_1d)
    site_a = oscillator_2d(mat_a,x1,x2)
    site_b = oscillator_2d(mat_b,x1,x2)
    total = site_a**2 + site_b**2
    im = ax.contourf(x1,x2,total)
    fig.colorbar(im)
    ax.set_xlabel('$q_1$',fontsize=20)
    ax.set_ylabel('$q_2$',fontsize=20)
    ax.set(adjustable='box-forced', aspect='equal')
    x2a_max, x1a_max = np.unravel_index(np.argmax(total),x1.shape)
    print('Excitation is centered at (q1,q2) = ({0:.2f},{1:.2f})'.format(x1_1d[x1a_max],
                                                                           x1_1d[x2a_max]))
    print('Site a contribution: {}'.format(np.sum(mat_a.flatten()**2)))
    print('Site b contribution: {}'.format(np.sum(mat_b.flatten()**2)))
    
    
def plot_dimer_2vib_g_or_f(eigval, eigmat):
    fig, ax1 = plt.subplots()
    ax1.set_title('E = {:.4f}'.format(eigval),fontsize=16)
    mat_a = eigmat[0,...]
    x1_1d = np.arange(-3,3,.05)
    x1, x2 = np.meshgrid(x1_1d, x1_1d)
    site_a = oscillator_2d(mat_a,x1,x2)
    im1 = ax1.contourf(x1,x2,site_a)
    fig.colorbar(im1)
    ax1.set_xlabel('$q_1$',fontsize=20)
    ax1.set_ylabel('$q_2$',fontsize=20)
    ax1.set(adjustable='box-forced', aspect='equal')
    x2a_max, x1a_max = np.unravel_index(np.argmax(np.abs(site_a)),x1.shape)
    print('Excitation on site a is centered at (q1,q2) = ({0:.2f},{1:.2f})'.format(x1_1d[x1a_max],
                                                                           x1_1d[x2a_max]))

def plot_dimer_singly_excited_states(obj,*,plot_range=[-0.1,4.1],fig=None):
    if fig == None:
        fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    for ev in obj.eigenvalues[1]:
        ax.axvline(x=ev)
    ax.set_xlim(plot_range)
    ax.set_xlabel('Energy ($\hbar\omega$)')
    d = (obj.params['vibrations'][0]['displacements2'][0] +
         obj.params['vibrations'][1]['displacements2'][0] )/2
    ax.set_title('Eigenenergies for $J =$ {:.2f}, $d =$ {:.2f}'.format(obj.couplings[1][0],d))
        
