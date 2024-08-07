#Standard python libraries
import os
import warnings
import time

#Dependencies - numpy, scipy,
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.linalg import eigs

from ufss.perturbative_calculations import DPBaseClass
from ufss.perturbative_calculations import OpenBaseClass
from ufss.perturbative_calculations import RK_perturbative_container
from scipy.integrate import RK45

class RKOpenEngine(OpenBaseClass,DPBaseClass):
    """This class is designed to calculate perturbative wavepackets in the
        light-matter interaction given the eigenvalues of the unperturbed 
        hamiltonian and the material dipole operator evaluated in the
        eigenbasis of the unperturbed hamiltonian.

    Args:
        file_path (string): path to folder containing eigenvalues and the
            dipole operator for the system Hamiltonian
        detection_type (string): options are 'polarization' (default) or 'fluorescence'

"""
    def __init__(self,file_path,*,detection_type = 'polarization',
                 conserve_memory=False,method='Euler'):
        DPBaseClass.__init__(self,file_path,detection_type=detection_type)
        OpenBaseClass.__init__(self,detection_type = detection_type)
        self.slicing_time = 0
        self.interpolation_time = 0
        self.expectation_time = 0
        self.RK45_step_time = 0
        self.dipole_dot_rho_time = 0
        self.dipole_time = 0
        self.automation_time = 0
        self.diagram_to_signal_time = 0
        self.diagram_generation_counter = 0
        self.convolution_time = 0

        self.interaction_picture_calculations = True

        self.sparsity_threshold = 0.1

        self.conserve_memory = conserve_memory

        self.method = method

        self.load_L()

        self.set_rho_shapes()

        self.load_H_eigensystem()

        if not self.conserve_memory:
            self.load_mu()

        try:
            self.load_H_mu()
            self.H_mu_flag = True
        except FileNotFoundError:
            self.load_mu()
            self.H_mu_flag = False
            warnings.warn('Could not find H_mu, defaulting to L_mu for calculating expectation values, which is slower')

        self.atol = 1E-6
        self.rtol = 1E-5

        self.set_maximum_manifold()

    def set_efields(self,times_list,efields_list,centers_list,
                    phase_discrimination,*,reset_calculations = True,
                    plot_fields = False):
        
        DPBaseClass.set_efields(self,times_list,efields_list,centers_list,
                                 phase_discrimination,
                                 reset_calculations = reset_calculations,
                                 plot_fields = plot_fields)
        
        # Initialize unperturbed density matrix
        self.set_rho0_auto()

    def save_timing(self):
        save_dict = {'RKE_calculation_time':self.calculation_time}
        np.savez(os.path.join(self.base_path,'RKE_calculation_time.npz'),**save_dict)

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
        pna = ra.pulse_number
        pnb = ra.pulse_number
        ta = self.pulse_times[pna]
        tb = self.pulse_times[pnb]
        if np.allclose(ta,tb):
            t = ra.t[1:2]
            rho_a = ra.f[:,1:2]
            rho_b = rb.f[:,1:2]
            pulse_number = max(pna,pnb)
        else:
            raise Exception('Cannot add two impulsive rhos unless they have the same value of t0')

        rho = rho_a + rho_b
        rho *= 2 * simultaneous
        
        rab=RK_perturbative_container(t,rho,pulse_number,manifold_key,pdc,
                                      simultaneous = simultaneous)
        rab.one_time_step_function = self.one_time_step_function
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
        rho = ra(t) + rb(t)

        pna = ra.pulse_number
        pnb = rb.pulse_number
        pulse_number = max(pna,pnb)
        
        rab = RK_perturbative_container(t,rho,pulse_number,manifold_key,pdc)
        rab.one_time_step_function = self.one_time_step_function
        return rab

    def load_L(self):
        """Load in known eigenvalues. Must be stored as a numpy archive file,
with keys: GSM, SEM, and optionally DEM.  The eigenvalues for each manifold
must be 1d arrays, and are assumed to be ordered by increasing energy. The
energy difference between the lowest energy ground state and the lowest 
energy singly-excited state should be set to 0
"""
        L_save_name = os.path.join(self.base_path,'L.npz')
        try:
            with np.load(L_save_name,allow_pickle=True) as L_archive:
                self.L = dict()
                for key in L_archive.keys():
                    L = L_archive[key]
                    if L.dtype == np.dtype('O'):
                        self.L[key] = L[()]
                    else:
                        if self.check_sparsity(L):
                            self.L[key] = csr_matrix(L)
                        else:
                            self.L[key] = L
        except:
            self.L = {'all_manifolds':load_npz(L_save_name)}
        self.manifolds = list(self.L.keys())

        if len(self.manifolds[0]) == 2:
            self.convert_manifold_keys()

    def convert_manifold_keys(self):
        old_manifolds = self.manifolds
        new_manifolds = []
        for k,b in self.manifolds:
            new_key = ''.join([k,',',b])
            new_manifolds.append(new_key)
        for old_key,new_key in zip(old_manifolds,new_manifolds):
            self.L[new_key] = self.L.pop(old_key)

        self.manifolds = new_manifolds
        
    def dL(self,t,rho):
        try:
            L = self.L['all_manifolds']
        except KeyError:
            L = self.L[rho.manifold_key]
        return L.dot(rho)

    def get_dL_manual(self,manifold_key):
        try:
            L = self.L['all_manifolds']
        except KeyError:
            L = self.L[manifold_key]

        def L_fun(t,rho):
            return L.dot(rho)

        return L_fun

    def one_time_step_function(self,rho0,t0,tf,*,manifold_key = None):
        num_steps = 0
        if manifold_key == None:
            rk45 = RK45(self.dL,t0,rho0,tf,atol=self.atol,rtol=self.rtol)
        else:
            dL = self.get_dL_manual(manifold_key)
            rk45 = RK45(dL,t0,rho0,tf,atol=self.atol,rtol=self.rtol)
        if rk45.direction == 1:
            while rk45.t < tf:
                rk45.step()
                num_steps += 1
        else:
            while rk45.t > tf:
                rk45.step()
                num_steps += 1
        rho_final = rk45.y
        return rho_final

    def get_bottom_eigenvector(self):
        try:
            L = self.L['all_manifolds']
        except KeyError:
            L = self.L['0,0']
        if L.shape == (1,1):
            e = L[0,0]
            ev = np.array([[1]])
        else:
            e, ev = eigs(L,k=1,which='SM',maxiter=10000)
        if e.size == 1 and np.allclose(e,0):
            pass
        else:
            raise Exception('Smallest magnitude eigenvalue of L is {}. L must have a single stationary state for this code to work'.format(e))
        v = ev[:,0]
        H_size = int(np.sqrt(v.size))
        rho = v.reshape((H_size,H_size))
        trace = rho.trace()
        v = v/trace # Need to start with a trace 1 object
        return v

    def set_rho0_auto(self):
        try:
            rho0 = np.load(os.path.join(self.base_path,'rho0.npy'))
        except FileNotFoundError:
            rho0 = self.get_bottom_eigenvector()
        t = np.array([-np.inf,0,np.inf])
        rho0 = rho0[:,np.newaxis] * np.ones((rho0.size,t.size))
        pulse_number = None
        manifold_key = '0,0'
        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)
        
        self.rho0 = RK_perturbative_container(t,rho0,pulse_number,
                                              manifold_key,pdc,
                                              interp_kind = 'zero')

    def set_rho0_manual(self,rho0):
        t = np.array([-np.inf,0,np.inf])
        rho0 = rho0[:,np.newaxis] * np.ones((rho0.size,t.size))
        pulse_number = None
        manifold_key = '0,0'
        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)
        
        self.rho0 = RK_perturbative_container(t,rho0,pulse_number,
                                              manifold_key,pdc,
                                              interp_kind = 'zero')

    def set_rho_shapes(self):
        self.rho_shapes = dict()
        if 'all_manifolds' in self.manifolds:
            L_size = self.L['all_manifolds'].shape[0]
            H_size = int(np.sqrt(L_size))
            self.rho_shapes['all_manifolds'] = (H_size,H_size)
        else:
            H_sizes = dict()
            for key in self.manifolds:
                ket_key, bra_key = key.split(',')
                if ket_key == bra_key:
                    L_size = self.L[key].shape[0]
                    H_size = int(np.sqrt(L_size))
                    H_sizes[ket_key] = H_size
            for key in self.manifolds:
                ket_key, bra_key = key.split(',')
                ket_size = H_sizes[ket_key]
                bra_size = H_sizes[bra_key]
                self.rho_shapes[key] = (ket_size,bra_size)

    def get_rho(self,t,rho_obj,*,reshape=True,original_L_basis=True):
        manifold_key = rho_obj.manifold_key
        
        try:
            rho_shape = self.rho_shapes['all_manifolds']
        except KeyError:
            rho_shape = self.rho_shapes[manifold_key]
        rho_shape = rho_shape + (t.size,)
        
        rho = rho_obj(t)
        if reshape:
            rho = rho.reshape(rho_shape)
            
        return rho

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
            be stored as a .npz file, and must contain at least one array, each with three 
            indices: (new manifold index, old manifold eigenfunction, 
            cartesian coordinate)."""
        try:
            file_name = os.path.join(self.base_path,'mu_site_basis.npz')
            with np.load(file_name) as mu_archive:
                self.mu = {key:mu_archive[key] for key in mu_archive.keys()}
        except FileNotFoundError:
            try:
                file_name = os.path.join(self.base_path,'mu_original_L_basis.npz')
                with np.load(file_name) as mu_archive:
                    self.mu = {key:mu_archive[key] for key in mu_archive.keys()}
            except FileNotFoundError:
                file_name = os.path.join(self.base_path,'mu.npz')
                with np.load(file_name) as mu_archive:
                    self.mu = {key:mu_archive[key] for key in mu_archive.keys()}
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
        
    ### Tools for recursively calculating perturbed density maatrices using TDPT

    def rho_matrix_to_L_vector(self,rho,manifold_key,*,
                                diagonal_L_basis=False):
        t_size = rho.shape[-1]
        rho_vec_size = np.prod(rho.shape[:-1])
        rho = rho.reshape(rho_vec_size,t_size)
        
        return rho

    def rho_L_vector_to_matrix(self,rho,manifold_key):
        try:
            rho_shape = self.rho_shapes['all_manifolds']
        except:
            rho_shape = self.rho_shapes[manifold_key]
        tsize = rho.shape[-1]
        rho_shape = rho_shape + (tsize,)
        rho = rho.reshape(rho_shape)

        return rho

    def mu_dot_rho(self,t,rho_in,pulse_number,ket_flag=True,up_flag=True,
                   next_manifold_mask = None,mu_type='L'):
        old_manifold_key = rho_in.manifold_key
        new_manifold_key = self.get_new_manifold_key(old_manifold_key,
                                                     ket_flag,up_flag)

        mu = self.get_mu(pulse_number,old_manifold_key,
                         new_manifold_key,ket_flag=ket_flag,
                         up_flag=up_flag,mu_type=mu_type)
            
        if mu_type=='H':
            rho =self.get_rho(t,rho_in,reshape=True)

            rho = self.H_mu_dot_rho(mu,rho,ket_flag=ket_flag)
            
            rho = self.rho_matrix_to_L_vector(rho,new_manifold_key)
            
        elif mu_type=='L':
            rho =self.get_rho(t,rho_in,reshape=False)
            rho = mu.dot(rho)
        else:
            raise Exception("mu_type must be 'H' or 'L'")
            
        return rho

    def H_mu_dot_rho(self,mu,rho,ket_flag=True):
        if ket_flag:
            ein_logic = 'ij,jkl'
            operators = (mu,rho)
        else:
            ein_logic = 'ijl,jk'
            operators = (rho,mu)
            
        if 'all_manifolds' in self.manifolds:
            if len(self.rho_shapes['all_manifolds']) == 3:
                if ket_flag:
                    ein_logic = 'ij,ajkl'
                else:
                    ein_logic = 'aijl,jk'

        return np.einsum(ein_logic,*operators)

    def mu_exp_val(self,t,rho_in,pulse_number,ket_flag=True,up_flag=False,
                   mu_type='H'):
        old_manifold_key = rho_in.manifold_key
        new_manifold_key = self.get_new_manifold_key(old_manifold_key,
                                                     ket_flag,up_flag)

        if not self.H_mu_flag:
            if mu_type == 'H':
                warnings.warn('Using L_mu for expectation values because H_mu is not available')
                mu_type = 'L'
        
        mu = self.get_mu(pulse_number,old_manifold_key,
                                     new_manifold_key,ket_flag=ket_flag,
                                     up_flag=up_flag,mu_type=mu_type)
            
        if mu_type=='H':
            rho =self.get_rho(t,rho_in,reshape=True)
            if 'all_manifolds' in self.manifolds:
                if len(self.rho_shapes['all_manifolds']) == 3:
                    rho = rho[0,...]

            if ket_flag:
                exp_val = np.einsum('ij,jil',mu,rho)
            else:
                exp_val = np.einsum('ijl,ji',rho,mu)
            
        else:
            rho =self.get_rho(t,rho_in,reshape=False)
            rho = mu.dot(rho)

            rho = self.rho_L_vector_to_matrix(rho,new_manifold_key)
            exp_val = np.einsum('iik',rho)
            
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

        ##probably remove this, mu_type = 'H' always for DP methods, I think
        mu_type = 'H' if self.conserve_memory else 'L'

        rho = self.mu_dot_rho(t,rho_in,pulse_number,mu_type=mu_type,
                              ket_flag=ket_flag,up_flag=up_flag,
                              next_manifold_mask=next_manifold_mask)
        
        ### Generic
        rho = rho * efield[np.newaxis,:]

        ### DP methods

        M = t.size

        if M == 1:
            next_rho = rho
        else:
            next_rho = np.zeros(rho.shape,dtype='complex')

            dt = self.dts[pulse_number]
            
            if self.method == 'Euler':
                next_rho[:,0] = rho[:,0] * dt
                for i in range(1,t.size):
                    rho0 = next_rho[:,i-1]
                    t0 = t[i-1]
                    ta = time.time()
                    next_rho[:,i] = self.one_time_step_function(rho0,t0,t[i],manifold_key=new_manifold_key)

                    tb = time.time()
                    self.RK45_step_time += tb - ta

                    next_rho[:,i] += rho[:,i] * dt

            elif self.method == 'UF2':
                ref_time = pulse_time
                for i in range(t.size):
                    next_rho[:,i]=self.one_time_step_function(rho[:,i],t[i],
                                    ref_time,manifold_key=new_manifold_key)

                t0 = time.time()

                M = self.efield_times[pulse_number].size

                fft_convolve_fun = self.heaviside_convolve_list[pulse_number].fft_convolve2

                next_rho = fft_convolve_fun(next_rho,d=dt)

                t1 = time.time()
                self.convolution_time += t1-t0

                for i in range(t.size):
                    next_rho[:,i]=self.one_time_step_function(next_rho[:,i],
                                ref_time,t[i],manifold_key=new_manifold_key)

        rho_out = RK_perturbative_container(t,next_rho,pulse_number,
                                            new_manifold_key,output_pdc,
                                            simultaneous=simultaneous)
        rho_out.one_time_step_function = self.one_time_step_function
    
        return rho_out
