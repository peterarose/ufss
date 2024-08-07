#Standard python libraries
import os
import warnings
import time

#Dependencies - numpy, scipy, matplotlib, pyfftw
import numpy as np
from scipy.sparse import csr_matrix

from ufss.perturbative_calculations import UF2BaseClass
from ufss.perturbative_calculations import perturbative_container
from ufss.perturbative_calculations import OpenBaseClass

class UF2OpenEngine(OpenBaseClass,UF2BaseClass):
    """This class is designed to calculate perturbative wavepackets in the
        light-matter interaction given the eigenvalues of the unperturbed 
        hamiltonian and the material dipole operator evaluated in the
        eigenbasis of the unperturbed hamiltonian.

    Args:
        file_path (string): path to folder containing eigenvalues and the
            dipole operator for the system Hamiltonian
        num_conv_points (int): number of desired points for linear 
            convolution. Also number of points used to resolve all optical
            pulse shapes
        dt (float): time spacing used to resolve the shape of all optical
            pulses
        initial_state (int): index of initial state for psi^0

"""
    def __init__(self,file_path,*,detection_type = 'polarization',
                 conserve_memory=False):
        UF2BaseClass.__init__(self,file_path,detection_type=detection_type)
        OpenBaseClass.__init__(self,detection_type = detection_type)
        self.slicing_time = 0
        self.interpolation_time = 0
        self.expectation_time = 0
        self.next_order_expectation_time = 0
        self.convolution_time = 0
        self.extend_time = 0
        self.mask_time = 0
        self.dipole_time = 0
        self.automation_time = 0
        self.diagram_to_signal_time = 0
        self.diagram_generation_counter = 0
        self.number_of_diagrams_calculated = 0
        self.efield_mask_time = 0
        self.dipole_down_dot_product_time = 0
        self.reshape_and_sum_time = 0
        self.rho_to_polarization_time = 0
        self.next_order_time = 0

        self.next_order_counter = 0

        self.check_for_zero_calculation = False
        self.interaction_picture_shift = True
        self.interaction_picture_calculations = True

        self.sparsity_threshold = .1

        self.conserve_memory = conserve_memory

        self.load_eigensystem()

        self.set_rho_shapes()

        self.load_H_eigensystem()

        try:
            self.load_H_mu()
            self.H_mu_flag = True
        except FileNotFoundError:
            self.load_mu()
            self.H_mu_flag = False
            warnings.warn('Could not find H_mu, defaulting to L_mu for calculating expectation values, which is slower')

        if not self.conserve_memory:
            self.load_mu()

        if self.detection_type == 'fluorescence':
            self.load_fl_yield()

        self.set_maximum_manifold()
        

    ### Methods for calculating composite diagrams

    def add_impulsive_rhos(self,ra,rb):
        """Add two density matrix objects together
"""
        manifold_key = ra.manifold_key
        pdc = ra.pdc
        sim_a = ra.simultaneous
        sim_b = rb.simultaneous
        if sim_a != sim_b:
            raise Exception('Cannot add impulsive rhos if they do not have the same number of simultaneous interactions')
        else:
            simultaneous = sim_a
        if np.allclose(ra.t0,rb.t0):
            t = ra.t[1:2]
            rho_a = ra.f[:,1]
            rho_b = rb.f[:,1]
            pulse_number = ra.pulse_number
            t0 = ra.t0
        else:
            raise Exception('Cannot add two impulsive rhos unless they have the same value of t0')

        rho = np.zeros((ra.bool_mask.size,t.size),dtype='complex')
        rho[ra.bool_mask,:] = rho_a[:,np.newaxis]
        rho[rb.bool_mask,:] += rho_b[:,np.newaxis]
        rho *= 2 * simultaneous

        bool_mask = np.logical_or(ra.bool_mask,rb.bool_mask)
        rho = rho[bool_mask,:]
        
        rab=perturbative_container(t,rho,bool_mask,pulse_number,manifold_key,pdc,t0,
                          simultaneous = simultaneous)
        return rab
        

    def add_rhos(self,ra,rb):
        """Add two density matrix objects together
"""
        if not np.allclose(ra.pdc,rb.pdc):
            raise Exception('Cannot add density matrices with different phase-discrimination conditions')
        if 'all_manifolds' in self.manifolds:
            pass
        else:
            if ra.manifold_key != rb.manifold_key:
                raise Exception('Cannot add perturbative_container objects that exist in different manifolds')
            
        if ra.impulsive:
            return self.add_impulsive_rhos(ra,rb)
        
        manifold_key = ra.manifold_key
        
        pdc = ra.pdc
        
        tmin = min(ra.t[0],rb.t[0])
        tmax = max(ra.t[-1],rb.t[-1])
        dt = min(ra.t[1]-ra.t[0],rb.t[1]-rb.t[0])
        t = np.arange(tmin,tmax+dt*0.9,dt)
        rho = np.zeros((ra.bool_mask.size,t.size),dtype='complex')
        rho_a = ra(t)
        rho_b = rb(t)

        if ra.t0 > rb.t0:
            t0 = ra.t0
            pulse_number = ra.pulse_number
        else:
            t0 = rb.t0
            pulse_number = rb.pulse_number

        try:
            eva = self.eigenvalues['all_manifolds'][ra.bool_mask]
            evb = self.eigenvalues['all_manifolds'][rb.bool_mask]
        except KeyError:
            eva = self.eigenvalues[ra.manifold_key][ra.bool_mask]
            evb = self.eigenvalues[rb.manifold_key][rb.bool_mask]

        if self.interaction_picture_shift:
            t_exp_a = t0 - ra.t0
            t_exp_b = t0 - rb.t0
            rho_a *= np.exp(eva[:,np.newaxis]*t_exp_a)
            rho_b *= np.exp(evb[:,np.newaxis]*t_exp_b)
        else:
            pass

        rho[ra.bool_mask,:] = rho_a
        rho[rb.bool_mask,:] += rho_b

        bool_mask = np.logical_or(ra.bool_mask,rb.bool_mask)
        rho = rho[bool_mask,:]
        
        rab=perturbative_container(t,rho,bool_mask,pulse_number,manifold_key,pdc,t0)
        return rab

    def set_efields(self,times_list,efields_list,centers_list,
                    phase_discrimination,*,reset_calculations = True,
                    plot_fields = False):

        max_calculation_times = [np.max(np.abs(t)) for t in times_list]
        max_calc_time = np.max(max_calculation_times)
        largest_decay_rates = []
        for key in self.eigenvalues.keys():
            gamma_max = -np.min(np.real(self.eigenvalues[key]))
            largest_decay_rates.append(gamma_max)
        largest_decay_rate = np.max(largest_decay_rates)
        if largest_decay_rate * max_calc_time > 40:
            self.interaction_picture_calculations = False
        else:
            self.interaction_picture_calculations = True
        
        UF2BaseClass.set_efields(self,times_list,efields_list,centers_list,
                                 phase_discrimination,
                                 reset_calculations = reset_calculations,
                                 plot_fields = plot_fields)
        
        # Initialize unperturbed density matrix
        self.set_rho0()

    def convert_manifold_keys(self):
        old_manifolds = self.manifolds
        new_manifolds = []
        for k,b in self.manifolds:
            new_key = ''.join([k,',',b])
            new_manifolds.append(new_key)
        for old_key,new_key in zip(old_manifolds,new_manifolds):
            self.eigenvalues[new_key] = self.eigenvalues.pop(old_key)
            self.eigenvectors[new_key] = self.eigenvectors.pop(old_key)
            if self.conserve_memory:
                self.left_eigenvectors[new_key] = self.left_eigenvectors.pop(old_key)

        self.manifolds = new_manifolds

    ########## specific to UF2

    def save_timing(self):
        save_dict = {'UF2_calculation_time':self.calculation_time}
        np.savez(os.path.join(self.base_path,'UF2_calculation_time.npz'),**save_dict)

    def get_rho(self,t,rho_obj,*,reshape=True,original_L_basis=True):
        manifold_key = rho_obj.manifold_key
        mask = rho_obj.bool_mask
        if self.interaction_picture_shift:
            t_exp = t - rho_obj.t0
        else:
            t_exp = t
        
        try:
            e = self.eigenvalues['all_manifolds'][mask]
            ev = self.eigenvectors['all_manifolds'][:,mask]
            ket_size, bra_size = self.rho_shapes['all_manifolds']
        except KeyError:
            e = self.eigenvalues[manifold_key][mask]
            ev = self.eigenvectors[manifold_key][:,mask]
            ket_size, bra_size = self.rho_shapes[manifold_key]

        rho = rho_obj(t)*np.exp(e[:,np.newaxis]*t_exp[np.newaxis,:])
        if original_L_basis:
            new_rho = ev.dot(rho)
        else:
            new_rho = rho
        if reshape:
            if original_L_basis:
                new_rho = new_rho.reshape(ket_size,bra_size,rho.shape[-1])
            else:
                raise Exception('Cannot reshape without moving to original L basis')
        return new_rho

    def set_rho0(self):
        """Creates the unperturbed density matarix by finding the 0 
            eigenvalue of the ground-state manifold, which should correspond
            to a thermal distribution
"""
        try:
            ev = self.eigenvalues['all_manifolds']
        except KeyError:
            ev = self.eigenvalues['0,0']

        t = self.efield_times[0] # can be anything of the correct length
        
        initial_state = np.where(np.isclose(ev,0,atol=1E-12))[0]
        rho0 = np.ones((1,t.size),dtype=complex)
        bool_mask = np.zeros(ev.size,dtype='bool')
        bool_mask[initial_state] = True

        if bool_mask.sum() != 1:
            warnings.warn('Could not automatically determine the initial thermal state. User must specify the initial condition, rho^(0), manually')
            return None

        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)

        t0 = 0

        self.rho0 = perturbative_container(t,rho0,bool_mask,None,'0,0',pdc,t0,
                                  interp_kind='zero',interp_left_fill=1)

    def set_rho0_manual_L_eigenbasis(self,manifold_key,bool_mask,weights):
        """
"""
        ev = self.eigenvalues[manifold_key][bool_mask]

        t = self.efield_times[0] # can be anything of the correct length
        
        rho0 = np.ones((bool_mask.sum(),t.size),dtype=complex) * weights[:,np.newaxis]

        if manifold_key == 'all_manifolds':
            manifold_key = '0,0'

        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)

        t0 = 0

        self.rho0 = perturbative_container(t,rho0,bool_mask,None,manifold_key,pdc,t0,
                                  interp_kind='zero',interp_left_fill=1)

    def set_rho0_manual(self,rho0,*,manifold_key = 'all_manifolds'):
        """Set the initial condition.  Must be done after setting the pulse shapes
        Args:
            rho0 (2D np.array) : the initial density matrix, in the basis that
                the Liouvillian was defined in
            manfifold_key (str) : manifold in which initial density matrix is 
                defined (usually 'all_manifolds' (default) or '00')
"""
        try:
            evl = self.left_eigenvectors[manifold_key]
        except AttributeError:
            self.load_left_eigenvectors()
            evl = self.left_eigenvectors[manifold_key]

        rho0_flat = rho0.flatten() # ufss always works with vectors
        

        rho0_eig = evl.dot(rho0_flat) # transform into Liouvillian eigenbasis
        
        nonzero_inds = np.where(np.abs(rho0_eig) > 1E-12)[0]

        bool_mask = np.zeros(rho0_eig.size,dtype='bool')
        bool_mask[nonzero_inds] = True

        rho0_trimmed = rho0_eig[nonzero_inds]

        t = self.efield_times[0]
        time_dependence = np.ones(t.size)
        rho0 = rho0_trimmed[:,np.newaxis] * time_dependence[np.newaxis,:]
        
        if manifold_key == 'all_manifolds':
            manifold_key = '0,0'

        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)

        t0 = 0

        self.rho0 = perturbative_container(t,rho0,bool_mask,None,manifold_key,pdc,t0,
                                  interp_kind='zero',interp_left_fill=1)

    def load_eigensystem(self):
        """Load in known eigenvalues and eigenvectors. Must be stored as a 
            numpy archive file
"""
        eigval_save_name = os.path.join(self.base_path,'eigenvalues.npz')
        eigvec_save_name = os.path.join(self.base_path,'right_eigenvectors.npz')
        with np.load(eigval_save_name) as eigval_archive:
            self.manifolds = list(eigval_archive.keys())
            self.eigenvalues = {key:eigval_archive[key] for key in self.manifolds}
        with np.load(eigvec_save_name,allow_pickle=True) as eigvec_archive:
            self.eigenvectors = dict()
            for key in self.manifolds:
                ev = eigvec_archive[key]
                if ev.dtype == np.dtype('O'):
                    self.eigenvectors[key] = ev[()]
                elif self.check_sparsity(ev):
                    self.eigenvectors[key] = csr_matrix(ev)
                else:
                    self.eigenvectors[key] = ev

        if self.conserve_memory:
            self.load_left_eigenvectors()

        if len(self.manifolds[0]) == 2:
            self.convert_manifold_keys()

    def load_left_eigenvectors(self):
        left_eigvec_save_name = os.path.join(self.base_path,'left_eigenvectors.npz')
        with np.load(left_eigvec_save_name,allow_pickle=True) as left_eigvec_archive:
            self.left_eigenvectors = dict()
            for key in self.manifolds:
                evl = left_eigvec_archive[key]
                if evl.dtype == np.dtype('O'):
                    self.left_eigenvectors[key] = evl[()]
                elif self.check_sparsity(evl):
                    self.left_eigenvectors[key] = csr_matrix(evl)
                else:
                    self.left_eigenvectors[key] = evl
                        
    def set_rho_shapes(self):
        self.rho_shapes = dict()
        if 'all_manifolds' in self.manifolds:
            L_size = self.eigenvalues['all_manifolds'].size
            H_size = int(np.sqrt(L_size))
            self.rho_shapes['all_manifolds'] = (H_size,H_size)
        else:
            H_sizes = dict()
            for key in self.manifolds:
                ket_key, bra_key = key.split(',')
                if ket_key == bra_key:
                    L_size = self.eigenvalues[key].size
                    H_size = int(np.sqrt(L_size))
                    H_sizes[ket_key] = H_size
            for key in self.manifolds:
                ket_key, bra_key = key.split(',')
                ket_size = H_sizes[ket_key]
                bra_size = H_sizes[bra_key]
                self.rho_shapes[key] = (ket_size,bra_size)
            

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
            be stored as a .npz file, and must contain at least one array, each with three 
            indices: (new manifold eigenfunction, old manifold eigenfunction, 
            cartesian coordinate)."""
        file_name = os.path.join(self.base_path,'mu.npz')
        file_name_pruned = os.path.join(self.base_path,'mu_pruned.npz')
        file_name_bool = os.path.join(self.base_path,'mu_boolean.npz')
        try:
            mu_boolean_archive = np.load(file_name_bool)
            with np.load(file_name_bool) as mu_boolean_archive:
                self.mu_boolean = {key:mu_boolean_archive[key] for key in mu_boolean_archive.keys()}
            file_name = file_name_pruned
            pruned = True
        except FileNotFoundError:
            pruned = False
        with np.load(file_name) as mu_archive:
            self.mu = {key:mu_archive[key] for key in mu_archive.keys()}
        if pruned == False:
            self.mu_boolean = dict()
            for key in self.mu.keys():
                self.mu_boolean[key] = np.ones(self.mu[key].shape[:2],dtype='bool')
        sparse_flags = []
        for key in self.mu.keys():
            mu_2D = np.sum(np.abs(self.mu[key])**2,axis=-1)
            sparse_flags.append(self.check_sparsity(mu_2D))
        sparse_flags = np.array(sparse_flags)
        if np.allclose(sparse_flags,True):
            self.sparse_mu_flag = True
        else:
            self.sparse_mu_flag = False

        for key in self.mu.keys():
            mu_x = self.mu[key][...,0]
            mu_y = self.mu[key][...,1]
            mu_z = self.mu[key][...,2]

            if self.sparse_mu_flag:
                self.mu[key] = [csr_matrix(mu_x),csr_matrix(mu_y),csr_matrix(mu_z)]
            else:
                self.mu[key] = [mu_x,mu_y,mu_z]

        if 'all_manifolds' in self.manifolds:
            pass
        elif ',' in key:
            pass
        else:
            self.convert_mu_keys()

    ### Tools for recursively calculating perturbed wavepackets using TDPT

    def dipole_matrix(self,pulse_number,key,ket_flag=True,up_flag=True):
        """Calculates the dipole matrix given the electric field polarization vector,
            if ket_flag = False then uses the bra-interaction"""
        overlap_matrix = OpenBaseClass.dipole_matrix(self,pulse_number,key,
                                                     ket_flag=ket_flag,
                                                     up_flag=up_flag)
        try:
            boolean_matrix = self.mu_boolean[key]
        except KeyError:
            if ket_flag:
                key = 'ket'
            else:
                key = 'bra'
            if up_flag:
                key += '_up'
            else:
                key += '_down'
            boolean_matrix = self.mu_boolean[key]

        return boolean_matrix, overlap_matrix

    def electric_field_mask(self,pulse_number,key,conjugate_flag=False):
        """This method determines which molecular transitions will be 
supported by the electric field.  We assume that the electric field has
0 amplitude outside the minimum and maximum frequency immplied by the 
choice of dt and num_conv_points.  Otherwise we will inadvertently 
alias transitions onto nonzero electric field amplitudes.
"""
        ta = time.time()

        try:
            mask = self.efield_masks[pulse_number][key]
        except KeyError:
            starting_key, ending_key = key.split('_to_')
            efield_t = self.efield_times[pulse_number]
            efield_w = self.efield_frequencies[pulse_number]
            if conjugate_flag:
                center = -self.centers[pulse_number]
            else:
                center = self.centers[pulse_number]
            try:
                eig_starting = self.eigenvalues['all_manifolds']
                eig_ending = self.eigenvalues['all_manifolds']
            except KeyError:
                eig_starting = self.eigenvalues[starting_key]
                eig_ending = self.eigenvalues[ending_key]
            # imag part corresponds to the energetic transitions
            diff = np.imag(eig_ending[:,np.newaxis] - eig_starting[np.newaxis,:])



            if efield_t.size == 1:
                mask = np.ones(diff.shape,dtype='bool')
            else:
                # The only transitions allowed by the electric field shape are
                inds_allowed = np.where((diff + center > efield_w[0]) & (diff + center < efield_w[-1]))
                mask = np.zeros(diff.shape,dtype='bool')
                mask[inds_allowed] = 1
            
            self.efield_masks[pulse_number][key] = mask

            tb = time.time()
            self.efield_mask_time += tb - ta

        return mask

    def mask_dipole_matrix(self,boolean_matrix,overlap_matrix,
                           starting_manifold_mask,*,next_manifold_mask = None):
        """Takes as input the boolean_matrix and the overlap matrix that it 
            corresponds to. Also requires the starting manifold mask, which specifies
            which states have non-zero amplitude, given the signal tolerance requested.
            Trims off unnecessary starting elements, and ending elements. If 
            next_manifold_mask is None, then the masking is done automatically
            based upon which overlap elements are nonzero. If next_manifold_mask is
            a 1D numpy boolean array, it is used as the mask for next manifold."""
        t0 = time.time()
        if np.all(starting_manifold_mask == True):
            pass
        else:
            boolean_matrix = boolean_matrix[:,starting_manifold_mask]
            overlap_matrix = overlap_matrix[:,starting_manifold_mask]

        #Determine the nonzero elements of the new psi, in the
        #eigenenergy basis, n_nonzero
        if type(next_manifold_mask) is np.ndarray:
            n_nonzero = next_manifold_mask
        else:
            n_nonzero = np.any(boolean_matrix,axis=1)
        if np.all(n_nonzero == True):
            pass
        else:
            overlap_matrix = overlap_matrix[n_nonzero,:]

        t1 = time.time()
        self.mask_time += t1-t0

        return overlap_matrix, n_nonzero

    def get_H_mu(self,pulse_number,key,rotating_flag=True):
        """Calculates the dipole matrix given the electric field polarization vector"""
        overlap_matrix = OpenBaseClass.get_H_mu(self,pulse_number,key,
                                                 rotating_flag=rotating_flag)

        boolean_matrix = np.ones(overlap_matrix.shape,dtype='bool')

        return boolean_matrix, overlap_matrix

    def rho_matrix_to_L_vector(self,rho,manifold_key,*,
                               diagonal_L_basis=True):
        t_size = rho.shape[-1]
        rho_vec_size = rho.shape[0]*rho.shape[1]
        rho = rho.reshape(rho_vec_size,t_size)
        if diagonal_L_basis:
            if 'all_manifolds' in self.manifolds:
                evl = self.left_eigenvectors['all_manifolds']
            else:
                evl = self.left_eigenvectors[manifold_key]
                
            rho = evl.dot(rho)

        return rho

    def rho_L_vector_to_matrix(self,rho,manifold_key,bool_mask):
        try:
            ev = self.eigenvectors['all_manifolds'][:,bool_mask]
            ket_size,bra_size = self.rho_shapes['all_manifolds']
        except KeyError:
            ev = self.eigenvectors[manifold_key][:,bool_mask]
            ket_size,bra_size = self.rho_shapes[manifold_key]

        rho = ev.dot(rho)
        tsize = rho.shape[-1]
        
        rho = rho.reshape(ket_size,bra_size,tsize)

        return rho

    def mu_dot_rho(self,t,rho_in,pulse_number,ket_flag=True,up_flag=True,
                   next_manifold_mask = None,mu_type='L'):
        old_manifold_key = rho_in.manifold_key
        new_manifold_key = self.get_new_manifold_key(old_manifold_key,
                                                     ket_flag,up_flag)

        boolean_mu, mu = self.get_mu(pulse_number,old_manifold_key,
                                     new_manifold_key,ket_flag=ket_flag,
                                     up_flag=up_flag,mu_type=mu_type)
            
        if mu_type=='H':
            rho =self.get_rho(t,rho_in,reshape=True,original_L_basis=True)

            if ket_flag:
                rho = np.einsum('ij,jkl',mu,rho)
            else:
                rho = np.einsum('ijl,jk',rho,mu)
            
            rho = self.rho_matrix_to_L_vector(rho,new_manifold_key)
            
            n_nonzero = np.ones(rho.shape[0],dtype='bool')
            zero_inds = np.where(np.isclose(rho[:,-1],0,atol=1E-160))
            n_nonzero[zero_inds] = False
            rho = rho[n_nonzero,:]
            
        elif mu_type=='L':
            rho =self.get_rho(t,rho_in,reshape=False,original_L_basis=False)
            m_nonzero = rho_in.bool_mask

            # e_mask = self.electric_field_mask(pulse_number,mu_key,conjugate_flag=conjugate_flag)
            # boolean_matrix = boolean_matrix * e_mask
            # overlap_matrix = overlap_matrix * e_mask

            mu,n_nonzero=self.mask_dipole_matrix(boolean_mu,mu,m_nonzero,
                                    next_manifold_mask = next_manifold_mask)

            rho = mu.dot(rho)
        else:
            raise Exception("mu_type must be 'H' or 'L'")
            
        return rho, n_nonzero

    def mu_exp_val(self,t,rho_in,pulse_number,ket_flag=True,up_flag=False,
                   mu_type='H'):
        old_manifold_key = rho_in.manifold_key
        new_manifold_key = self.get_new_manifold_key(old_manifold_key,
                                                     ket_flag,up_flag)

        if not self.H_mu_flag:
            if mu_type == 'H':
                warnings.warn('Using L_mu for expectation values because H_mu is not available')
                mu_type = 'L'

        
        boolean_mu, mu = self.get_mu(pulse_number,old_manifold_key,
                                     new_manifold_key,ket_flag=ket_flag,
                                     up_flag=up_flag,mu_type=mu_type)
            
        if mu_type=='H':
            rho =self.get_rho(t,rho_in,reshape=True,original_L_basis=True)

            if ket_flag:
                exp_val = np.einsum('ij,jil',mu,rho)
            else:
                exp_val = np.einsum('ijl,ji',rho,mu)
            
        else:
            rho =self.get_rho(t,rho_in,reshape=False,original_L_basis=False)
            m_nonzero = rho_in.bool_mask

            # e_mask = self.electric_field_mask(pulse_number,mu_key,conjugate_flag=conjugate_flag)
            # boolean_matrix = boolean_matrix * e_mask
            # overlap_matrix = overlap_matrix * e_mask

            mu,n_nonzero=self.mask_dipole_matrix(boolean_mu,mu,m_nonzero)

            rho = mu.dot(rho)

            rho = self.rho_L_vector_to_matrix(rho,new_manifold_key,
                                              n_nonzero)
            exp_val = np.einsum('iik',rho)
            
        return exp_val
    
    def fl_exp_val(self,t,rho_in):
        rho =self.get_rho(t,rho_in,reshape=True,original_L_basis=True)
        pops = rho[...,0].diagonal()
        if 'all_manifolds' in self.manifolds:
            fl = self.fluorescence_yield['all_manifolds']
        else:
            fl = self.fluorescence_yield[rho_in.manifold_key]
        exp_val = np.dot(fl, pops)
        
        return exp_val
            
    
    def next_order(self,rho_in,*,ket_flag=True,up_flag=True,
                   next_manifold_mask = None,pulse_number = 0):
        """This function connects rho_p to rho_pj^(*) using a DFT convolution algorithm.

        Args:
            rho_in (perturbative_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)
            next_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold
        
        Return:
            rho_dict (perturbative_container): next-order density matrix
"""
        ### Generic
        tnext_order0 = time.time()
        if ket_flag == up_flag:
            # Rotating term excites the ket and de-excites the bra
            conjugate_flag = False
        else:
            # Counter-rotating term
            conjugate_flag = True

        if rho_in.pulse_number == None:
            old_pulse_time = -np.inf
        else:
            old_pulse_time = self.pulse_times[rho_in.pulse_number]
        pulse_time = self.pulse_times[pulse_number]

        if np.isclose(old_pulse_time,pulse_time):
            simultaneous = rho_in.simultaneous + 1
        else:
            simultaneous = 1

        t = self.efield_times[pulse_number] + pulse_time
        dt = self.dts[pulse_number]
        old_manifold_key = rho_in.manifold_key
        new_manifold_key = self.get_new_manifold_key(old_manifold_key,
                                                     ket_flag,up_flag)

        input_pdc = rho_in.pdc
        output_pdc = input_pdc.copy()
        if conjugate_flag:
            output_pdc[pulse_number][1] += 1
        else:
            output_pdc[pulse_number][0] += 1

        center = self.centers[pulse_number]
        efield = self.efields[pulse_number] * np.exp(-1j*center*t)

        if conjugate_flag:
            efield = np.conjugate(efield)

        if ket_flag:
            efield = 1j * efield
        else:
            efield = -1j * efield

        ### UF2 specific

        mu_type = 'H' if self.conserve_memory else 'L'

        rho,n_nonzero=self.mu_dot_rho(t,rho_in,pulse_number,mu_type=mu_type,
                                         ket_flag=ket_flag,up_flag=up_flag,
                                     next_manifold_mask=next_manifold_mask)
        
        ### Generic
        rho = rho * efield[np.newaxis,:]

        ### UF2 specific
        try:
            ev2 = self.eigenvalues['all_manifolds']
        except KeyError:
            ev2 = self.eigenvalues[new_manifold_key]

        if self.interaction_picture_shift:
            t_exp_2 = t - pulse_time
        else:
            t_exp_2 = t

        exp_factor2_arg = -ev2[n_nonzero,np.newaxis] * t_exp_2[np.newaxis,:]
        exp_factor2 = np.exp(exp_factor2_arg)
        
        if self.interaction_picture_calculations:
            rho = rho * exp_factor2
        else:
            t_exp_2b = t_exp_2 - t_exp_2[0]
            exp_factor2b_arg = ev2[n_nonzero,np.newaxis] * t_exp_2b[np.newaxis,:]
            exp_factor2b = np.exp(exp_factor2b_arg)
            print('Possible problem with non-interaction picture calculations')

        t0 = time.time()

        M = self.efield_times[pulse_number].size

        fft_convolve_fun = self.heaviside_convolve_list[pulse_number].fft_convolve2

        if M == 1:
            pass
        else:
            if self.interaction_picture_calculations:
                rho = fft_convolve_fun(rho,d=dt)
            else:
                rho = fft_convolve_fun(exp_factor2b,rho,d=dt)

        if not self.interaction_picture_calculations:
            rho = rho * exp_factor2

        t1 = time.time()
        self.convolution_time += t1-t0

        rho_out = perturbative_container(t,rho,n_nonzero,pulse_number,
                                new_manifold_key,output_pdc,pulse_time,
                                simultaneous=simultaneous)

        self.next_order_counter += 1
        self.next_order_time += time.time() - tnext_order0

        return rho_out

class Ununsed:

    def mask_negligible_states(self,t,evs):
        """Find eigenstates that have decayed to negligible values and 
            return mask to remove them
"""
        ev_t = t[0] * evs
        inds = np.where(np.real(ev_t) < -7)[0]
        mask = np.ones(evs.size,dtype='bool')
        mask[inds] = False
        return mask
