#Standard python libraries
import numpy as np
import os
import itertools
from scipy.sparse import csr_matrix, kron, identity

from .eigen_generator import EigenGenerator
from .eigenstates import LadderOperators

class CalculateCartesianDipoleOperatorLowMemory(EigenGenerator):
    """This class calculates the dipole operator in the eigenbasis of
        the system hamiltonian directly in the cartesian basis"""

    def __init__(self,parameter_file_path,*,mask_by_occupation_num=True):
        super().__init__(parameter_file_path,mask_by_occupation_num=mask_by_occupation_num)
        self.base_path = parameter_file_path
        self.load_params()
        self.set_vibrations()
        self.set_H()
        self.set_molecular_dipoles()

        self.calculate_mu()
        self.save_mu()

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

    def set_H(self,*,truncation_size = None):
        if truncation_size:
            self.truncation_size = truncation_size
            self.set_vibrations()
        self.H0 = self.manifold_hamiltonian(0)
        self.H1 = self.manifold_hamiltonian(1)
        
        if 'DEM' in self.manifolds:
            self.H2 = self.manifold_hamiltonian(2)

    def dipole_matrix(self,starting_manifold_num,next_manifold_num,pol):
        """Calculates the dipole matrix that connects from one 
            manifold to the next, using the known dipole moments and the efield 
            polarization, determined by the pulse number.
"""
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

    def calculate_mu_x(self):
        x = np.array([1,0,0])
        e0 = self.eigenvectors[0]
        e1 = self.eigenvectors[1]

        mu10_x = self.dipole_matrix(0,1,x)

        mu10_x = mu10_x.dot(e0)
        mu10_x = e1.T.dot(mu10_x)

        if 'DEM' in self.manifolds:
            mu21_x = self.dipole_matrix(1,2,x)

            e2 = self.eigenvectors[2]

            mu21_x = mu21_x.dot(e1)
            mu21_x = e2.T.dot(mu21_x)

    def calculate_mu_y(self):
        y = np.array([0,1,0])
        e0 = self.eigenvectors[0]
        e1 = self.eigenvectors[1]

        mu10_y = self.dipole_matrix(0,1,y)

        mu10_y = mu10_y.dot(e0)
        mu10_y = e1.T.dot(mu10_y)

        if 'DEM' in self.manifolds:
            mu21_y = self.dipole_matrix(1,2,y)

            e2 = self.eigenvectors[2]

            mu21_y = mu21_y.dot(e1)
            mu21_y = e2.T.dot(mu21_y)


    def calculate_mu_z(self):
        z = np.array([0,0,1])
        e0 = self.eigenvectors[0]
        e1 = self.eigenvectors[1]

        mu10_z = self.dipole_matrix(0,1,z)

        mu10_z = mu10_z.dot(e0)
        mu10_z = e1.T.dot(mu10_z)

        if 'DEM' in self.manifolds:
            mu21_z = self.dipole_matrix(1,2,z)

            e2 = self.eigenvectors[2]

            mu21_z = mu21_z.dot(e1)
            mu21_z = e2.T.dot(mu21_z)

    def combine_mu(self):
        mu10 = np.zeros((mu10_x.shape[0],mu10_x.shape[1],3))
        mu10[:,:,0] = mu10_x
        mu10[:,:,1] = mu10_y
        mu10[:,:,2] = mu10_z

        self.mu = {'GSM_to_SEM':mu10}

        if 'DEM' in self.manifolds:
            mu21 = np.zeros((mu21_x.shape[0],mu21_x.shape[1],3))
            mu21[:,:,0] = mu21_x
            mu21[:,:,1] = mu21_y
            mu21[:,:,2] = mu21_z

            self.mu['SEM_to_DEM'] = mu21

    def save_mu(self):
        np.savez(os.path.join(self.base_path,'mu.npz'),**self.mu)

class CalculateCartesianDipoleOperator(EigenGenerator):
    """This class calculates the dipole operator in the eigenbasis of
        the system hamiltonian directly in the cartesian basis"""

    def __init__(self,parameter_file_path,*,mask_by_occupation_num=True):
        super().__init__(parameter_file_path,mask_by_occupation_num=mask_by_occupation_num)
        self.base_path = parameter_file_path
        self.load_params()
        self.Ladder = LadderOperators(self.truncation_size)
        self.set_vibrations()
        self.set_H()
        self.set_molecular_dipoles()

        self.calculate_mu()
        self.save_mu()

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

    def set_H(self,*,truncation_size = None):
        if truncation_size:
            self.truncation_size = truncation_size
            self.set_vibrations()
        self.H0 = self.manifold_hamiltonian(0)
        self.H1 = self.manifold_hamiltonian(1)
        
        if 'DEM' in self.manifolds:
            self.H2 = self.manifold_hamiltonian(2)

    def dipole_matrix(self,starting_manifold_num,next_manifold_num,pol):
        """Calculates the dipole matrix that connects from one 
            manifold to the next, using the known dipole moments and the efield 
            polarization, determined by the pulse number.
"""
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

    def dipole_matrix_nonCondon(self,starting_manifold_num,next_manifold_num,nonCondonList,*,order=1):
        """Calculates the dipole matrix that connects from one 
            manifold to the next, using the known dipole moments and the efield 
            polarization, determined by the pulse number.
        Args:
            nonCondonList (list): list of floats describing condon violation of each vibrational mode
        Kwargs:
            order (int): order of non-condon violation
"""
        if len(self.energies[1]) > 1:
            warnings.warn('This only works correctly for 1 excited state')
        upper_manifold_num = max(starting_manifold_num,next_manifold_num)
        
        if abs(starting_manifold_num - next_manifold_num) != 1:
            warnings.warn('Can only move from manifolds 0 to 1 or 1 to 2')
            return None
        #This is broken except for 1st order
        if order > 1:
            warnings.warn('Basis correction not implemented for order > 1')
        xn = self.Ladder.x_power_n(order).copy()
        basis_correction = -self.params['vibrations'][0]['displacement0'][0] * np.eye(xn.shape[0])
        xn += basis_correction
        
        # 1st order Condon violation
        XN = nonCondonList[0] * self.vibration_identity_kron(0,xn,0)
        for i in range(1,len(nonCondonList)):
            XN += nonCondonList[i] * self.vibration_identity_kron(i,xn,0)
        
        # if upper_manifold_num == 1:
        #     d_vec = self.molecular_dipoles.dot(pol)
        #     d_mat = d_vec[:,np.newaxis]
        #     overlap_matrix = kron(d_mat,XN)
            
        # elif upper_manifold_num == 2:
        #     d_mat = self.molecular_dipoles_SEM_to_DEM.dot(pol)
        #     overlap_matrix = kron(d_mat.T,XN)

        if starting_manifold_num > next_manifold_num:
            # Take transpose if transition is down rather than up
            XN = np.conjugate(XN.T)

        starting_manifold_indices = self.manifold_indices[starting_manifold_num][0]
        next_manifold_indices = self.manifold_indices[next_manifold_num][0]
        XN = XN[next_manifold_indices,:]
        XN = XN[:,starting_manifold_indices]

        return csr_matrix(XN)

    def calculate_mu(self):
        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        e0 = self.eigenvectors[0]
        e1 = self.eigenvectors[1]

        mu10_x = self.dipole_matrix(0,1,x)
        mu10_y = self.dipole_matrix(0,1,y)
        mu10_z = self.dipole_matrix(0,1,z)
        try:
            linear_non_condon_list_x = [self.params['vibrations'][i]['condon_violation'][0][0] for i in range(self.num_vibrations)]
            linear_non_condon_list_y = [self.params['vibrations'][i]['condon_violation'][0][1] for i in range(self.num_vibrations)]
            linear_non_condon_list_z = [self.params['vibrations'][i]['condon_violation'][0][2] for i in range(self.num_vibrations)]
            linear_violation_x = self.dipole_matrix_nonCondon(0,1,linear_non_condon_list_x)
            mu10_x += linear_violation_x
            mu10_y += self.dipole_matrix_nonCondon(0,1,linear_non_condon_list_y)
            mu10_z += self.dipole_matrix_nonCondon(0,1,linear_non_condon_list_z)
        except KeyError:
            pass
            

        mu10_x = mu10_x.dot(e0)
        mu10_x = e1.T.dot(mu10_x)

        mu10_y = mu10_y.dot(e0)
        mu10_y = e1.T.dot(mu10_y)

        mu10_z = mu10_z.dot(e0)
        mu10_z = e1.T.dot(mu10_z)

        mu10 = np.zeros((mu10_x.shape[0],mu10_x.shape[1],3))
        mu10[:,:,0] = mu10_x
        mu10[:,:,1] = mu10_y
        mu10[:,:,2] = mu10_z

        self.mu = {'GSM_to_SEM':mu10}

        if 'DEM' in self.manifolds:
            mu21_x = self.dipole_matrix(1,2,x)
            mu21_y = self.dipole_matrix(1,2,y)
            mu21_z = self.dipole_matrix(1,2,z)

            e2 = self.eigenvectors[2]

            mu21_x = mu21_x.dot(e1)
            mu21_x = e2.T.dot(mu21_x)

            mu21_y = mu21_y.dot(e1)
            mu21_y = e2.T.dot(mu21_y)

            mu21_z = mu21_z.dot(e1)
            mu21_z = e2.T.dot(mu21_z)

            mu21 = np.zeros((mu21_x.shape[0],mu21_x.shape[1],3))
            mu21[:,:,0] = mu21_x
            mu21[:,:,1] = mu21_y
            mu21[:,:,2] = mu21_z

            self.mu['SEM_to_DEM'] = mu21

    def save_mu(self):
        np.savez(os.path.join(self.base_path,'mu.npz'),**self.mu)

class CalculateDipoleOperator(EigenGenerator):
    """This class calculates the dipole operator in the eigenbasis of
        the system Hamiltonian using the eigenvectors"""
    def __init__(self,parameter_file_path,*,mask_by_occupation_num=True):
        super().__init__(parameter_file_path,mask_by_occupation_num=mask_by_occupation_num)
        self.base_path = parameter_file_path
        self.load_params()

        self.set_mu()

    def x0(self,size):
        """Defines the identity operator in the vibrational space"""
        ham = np.diag(np.ones(size))
        return csr_matrix(ham)

    def x1(self,size):
        """Defines the position operator in the vibrational space"""
        def offdiag1(n):
            return np.sqrt((n+1)/2)

        n = np.arange(0,size)
        off1 = offdiag1(n[0:-1])
        ham = np.zeros((size,size))
        ham += np.diag(off1,k=1) + np.diag(off1,k=-1)
        return csr_matrix(ham)

    def new_vibration_identity_kron(self,position,item):
        """Takes in an operator on a single vibrational and krons it with the
            correct number of vibrational identities, inserting it into its 
            position as indexed by its mode position as specified in the 
            input file
"""
        identities = [np.identity(self.truncation_size) for n in
                      range(self.num_vibrations-1)]
        identities.insert(position,item)
        mat = identities.pop(0)
        for next_item in identities:
            mat = kron(mat,next_item)
        return mat

    def mu_vibrational_space(self):
        """Untested for condon violations """
        ident = self.x0(self.truncation_size)
        mu = self.new_vibration_identity_kron(0,ident) # Condon Approximation
        try:
            kappas = np.array(self.params['kappa'])
        except KeyError:
            # If parameters file does not specify a condon violation,
            # Assume condon approximation holds
            kappas = np.zeros(self.num_vibrations)
        if np.all(kappas == 0):
            # Assumes condon approximation
            pass
        else:
            # Linear condon violation supported so far
            x = self.x1(self.truncation_size)
            for i in range(kappas.size):
                mu += kappas[i] * self.new_vibration_identity_kron(i,x)
        return mu
            
    def mu_inner_product(self,eigmats1,eigmats2):
        """Example of how to write a potentially more complicated
            dipole operator on the vibrational space.
        Args:
            eigmats1 (np.ndarray): 3d numpy array with indices [n,m,o] 
                where n is the site index, m is the vibrational-space index
                and o is the eigen index
            eigmast2 (np.ndarray): 3d numpy array with indices [n,m,o] (same
                as eigmats1)
"""
        sites1, vib, num_eig1 = eigmats1.shape
        sites2, vib, num_eig2 = eigmats2.shape
        in_prod = np.zeros((num_eig1,num_eig2,sites1,sites2))
        vib_mu = self.mu_vibrational_space()
        # iterate over all sites
        for i in range(sites1):
            eigvecs1 = eigmats1[i,...]
            # Take matrix product with vibrational space mu
            eigvecs1 = vib_mu.dot(eigvecs1)
            for j in range(sites2):
                eigvecs2 = eigmats2[j,...]
                in_prod[...,i,j] = np.dot(eigvecs1.T,eigvecs2)

        return in_prod

    def simple_inner_product(self,eigmats1,eigmats2):
        return np.einsum('mji,njk',eigmats1,eigmats2)

    def make_overlap_matrix(self,manifold1,manifold2):
        eigvecs1 = self.eigenvectors[manifold1]
        eigvecs2 = self.eigenvectors[manifold2]

        num_eigvals1 = self.eigenvalues[manifold1].size
        num_eigvals2 = self.eigenvalues[manifold2].size

        num_sites1 = len(self.energies[manifold1])
        num_sites2 = len(self.energies[manifold2])

        vibration_space_size = len(self.eigenvectors[0][:,0])

        eigmats1 = eigvecs1.reshape((num_sites1,vibration_space_size,num_eigvals1))
        eigmats2 = eigvecs2.reshape((num_sites2,vibration_space_size,num_eigvals2))

        overlap_matrix = self.simple_inner_product(eigmats1,eigmats2)

        return overlap_matrix
            
    def calculate_mu(self):
        self.mu_GSM_to_SEM_site = self.make_overlap_matrix(1,0)
        if 'DEM' in self.manifolds:
            self.mu_SEM_to_DEM_site = self.make_overlap_matrix(2,1)

    def set_mu(self):
        file_name = os.path.join(self.base_path,'mu_site_basis.npz')
        try:
            mu_archive = np.load(file_name)
            self.mu_GSM_to_SEM_site = mu_archive['GSM_to_SEM']
            if 'DEM' in self.manifolds:
                self.mu_SEM_to_DEM_site = mu_archive['SEM_to_DEM']
        except (FileNotFoundError, KeyError):
            self.calculate_mu()
            self.save_mu()

    def save_mu(self):
        file_name = os.path.join(self.base_path,'mu_site_basis.npz')
        mu_site_dict = {'GSM_to_SEM':self.mu_GSM_to_SEM_site}
        if 'DEM' in self.manifolds:
            mu_site_dict['SEM_to_DEM'] = self.mu_SEM_to_DEM_site
        np.savez(file_name,**mu_site_dict)

class DipoleConverter(CalculateDipoleOperator):
    """Converts mu represented in the site basis into mu represented
in cartesian coordinates
"""
    def __init__(self,parameter_file_path):
        super().__init__(parameter_file_path)
        self.set_molecular_dipoles()
        self.save_cartesian_mu()

    ### Setting the molecular dipole

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

    def conv_mu_site_basis_to_cartesian(self,overlap_matrix,manifold1,manifold2):
        if np.abs(manifold1 - manifold2) != 1:
            raise ValueError('Dipole only moves between adjacent manifolds')
        if manifold1 == 0 or manifold2 == 0:
            d = self.molecular_dipoles
            # 4th index of overlap_matrix can only be 0
            # There is only 1 "site" in the GSM
            overlap_matrix = np.dot(overlap_matrix[...,0],d)
            
        elif manifold1 == 2 or manifold2 == 2:
            d = self.molecular_dipoles_SEM_to_DEM
            overlap_matrix = np.einsum('abij,jic',overlap_matrix,d)
            
            # I have to swap the indices of d to match the convention of overlap_matrix

        return overlap_matrix

    def save_cartesian_mu(self):
        file_name = os.path.join(self.base_path,'mu.npz')
        mu_GSM_to_SEM_cartesian = self.conv_mu_site_basis_to_cartesian(self.mu_GSM_to_SEM_site,1,0)
        mu_dict = {'GSM_to_SEM':mu_GSM_to_SEM_cartesian}
        if 'DEM' in self.manifolds:
            mu_SEM_to_DEM_cartesian = self.conv_mu_site_basis_to_cartesian(self.mu_SEM_to_DEM_site,2,1)
            mu_dict['SEM_to_DEM'] = mu_SEM_to_DEM_cartesian
        np.savez(file_name,**mu_dict)
