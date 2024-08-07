#Standard python libraries
import os
import warnings
import copy
import time

#Dependencies - numpy, scipy, matplotlib, pyfftw
import numpy as np
import numpy.polynomial.chebyshev as npch

from ufss import CompositeDiagrams
from ufss.perturbative_calculations import UF2BaseClass
from ufss.perturbative_calculations import ClosedBaseClass
from ufss.perturbative_calculations import perturbative_container
from ufss.perturbative_calculations import ChebPoly, cheb_perturbative_container

class UF2ClosedEngine(ClosedBaseClass,UF2BaseClass):
    """This class is designed to calculate perturbative wavepackets in the
        light-matter interaction given the eigenvalues of the unperturbed 
        hamiltonian and the material dipole operator evaluated in the
        eigenbasis of the unperturbed hamiltonian.

    Args:
        file_path (string): path to folder containing eigenvalues and the
            dipole operator for the system Hamiltonian
        initial_state (int): index of initial state for psi^0

"""
    def __init__(self,file_path,*,initial_state=0,
                 detection_type = 'polarization'):
        UF2BaseClass.__init__(self,file_path,detection_type = detection_type)
        ClosedBaseClass.__init__(self,detection_type = detection_type)
        self.slicing_time = 0
        self.interpolation_time = 0
        self.expectation_time = 0
        self.next_order_expectation_time = 0
        self.next_order_time = 0
        self.get_psi_time = 0
        self.convolution_time = 0
        self.extend_time = 0
        self.mask_time = 0
        self.dipole_time = 0
        self.automation_time = 0
        self.diagram_to_signal_time = 0
        self.efield_mask_time = 0
        self.method = 'UF2'

        self.initial_state = initial_state

        self.load_eigensystem()

        self.set_min_max_manifolds()

        self.load_mu()

    def add_impulsive_psis(self,a,b):
        """Add two perturbative_containers together
"""
        manifold_key = a.manifold_key
        pdc = a.pdc
        sim_a = a.simultaneous
        sim_b = b.simultaneous
        if sim_a != sim_b:
            raise Exception('Cannot add impulsive perturbative containers if they do not have the same number of simultaneous interactions')
        else:
            simultaneous = sim_a
        if np.allclose(a.t0,b.t0):
            t = a.t[1:2]
            f_a = a.f[:,1]
            f_b = b.f[:,1]
            pulse_number = a.pulse_number
            t0 = a.t0
        else:
            raise Exception('Cannot add two impulsive perturbative containers unless they have the same value of t0')

        f = np.zeros((a.bool_mask.size,t.size),dtype='complex')
        f[a.bool_mask,:] = f_a[:,np.newaxis]
        f[b.bool_mask,:] += f_b[:,np.newaxis]
        f *= 2 * simultaneous

        bool_mask = np.logical_or(a.bool_mask,b.bool_mask)
        f = f[bool_mask,:]
        
        ab=perturbative_container(t,f,bool_mask,pulse_number,manifold_key,
                                   pdc,t0,simultaneous = simultaneous)
        return ab

    def add_psis(self,a,b):
        """Add two perturbative_container objects together
"""
        if not np.allclose(a.pdc,b.pdc):
            raise Exception('Cannot add density matrices with different phase-discrimination conditions')
        if 'all_manifolds' in self.manifolds:
            pass
        else:
            if a.manifold_key != b.manifold_key:
                raise Exception('Cannot add rho_container objects that exist in different manifolds')
            
        if a.impulsive:
            return self.add_impulsive_psis(a,b)
        
        manifold_key = a.manifold_key
        
        pdc = a.pdc

        if self.method == 'UF2':
            tmin = min(a.t[0],b.t[0])
            tmax = max(a.t[-1],b.t[-1])
            dt = min(a.t[1]-a.t[0],b.t[1]-b.t[0])
            t = np.arange(tmin,tmax+dt*0.9,dt)
        elif self.method == 'chebyshev':
            l1,u1 = a.dom
            l2,u2 = b.dom
            dom = np.array([min(l1,l2),max(u1,u2)])
            midpoint = (dom[1] + dom[0])/2
            halfwidth = (dom[1] - dom[0])/2
            order = max(a.order,b.order)
            t = npch.chebpts1(order) * halfwidth + midpoint
            
        f = np.zeros((a.bool_mask.size,t.size),dtype='complex')
        f_a = a(t)
        f_b = b(t)

        if a.t0 > b.t0:
            t0 = a.t0
            pulse_number = a.pulse_number
        else:
            t0 = b.t0
            pulse_number = b.pulse_number

        # try:
        #     eva = self.eigenvalues['all_manifolds'][a.bool_mask]
        #     evb = self.eigenvalues['all_manifolds'][b.bool_mask]
        # except KeyError:
        #     eva = self.eigenvalues[a.manifold_key][a.bool_mask]
        #     evb = self.eigenvalues[b.manifold_key][b.bool_mask]

        # if self.interaction_picture_shift:
        #     t_exp_a = t0 - a.t0
        #     t_exp_b = t0 - b.t0
        #     f_a *= np.exp(eva[:,np.newaxis]*t_exp_a)
        #     f_b *= np.exp(evb[:,np.newaxis]*t_exp_b)
        # else:
        #     pass

        f[a.bool_mask,:] = f_a
        f[b.bool_mask,:] += f_b

        bool_mask = np.logical_or(a.bool_mask,b.bool_mask)
        f = f[bool_mask,:]

        if self.method == 'UF2':
            ab=perturbative_container(t,f,bool_mask,pulse_number,
                                      manifold_key,pdc,t0)
        elif self.method == 'chebyshev':
            ab=cheb_perturbative_container(t,f,bool_mask,pulse_number,
                                           manifold_key,pdc,t0,dom=dom)
        return ab
        
    def set_efields(self,times_list,efields_list,centers_list,
                    phase_discrimination,*,reset_calculations = True,
                    plot_fields = False):
        
        self.interaction_picture_calculations = True
        
        UF2BaseClass.set_efields(self,times_list,efields_list,centers_list,
                                 phase_discrimination,
                                 reset_calculations = reset_calculations,
                                 plot_fields = plot_fields)
        
        # Initialize unperturbed wavefunction
        self.set_psi0(self.initial_state)

    def get_psi_eigen_basis(self,t,key):
        psi_obj = self.psis[key]
        mask = psi_obj.bool_mask
        all_e = self.eigenvalues[psi_obj.manifold_key]
        e = all_e[mask]
        psi = psi_obj(t)*np.exp(-1j*e[:,np.newaxis]*t[np.newaxis,:])
        full_size = all_e.size
        total_psi = np.zeros(full_size,t.size)
        total_psi[mask,:] = psi
        return total_psi

    def get_psi_site_basis(self,t,key):
        psi_obj = self.psis[key]
        mask = psi_obj.bool_mask
        manifold_num = self.manifold_key_to_number(psi_obj.manifold_key)
        e = self.eigenvalues[manifold_num][mask]
        psi = psi_obj(t)*np.exp(-1j*e[:,np.newaxis]*t[np.newaxis,:])
        ev = self.eigenvectors[manifold_num][:,mask]
        new_psi = ev.dot(psi)
        full_size = 0
        manifold_sizes = []
        for i in range(len(self.eigenvalues)):
            manifold_sizes.append(self.eigenvectors[manifold_num].shape[0])
            full_size += manifold_sizes[-1]
        total_psi = np.zeros(full_size,t.size)
        start = 0
        for i in range(len(self.eigenvalues)):
            if i == manifold_num:
                end = start + manifold_sizes[i]
                total_psi[start:end,:] = new_psi
            else:
                start += manifold_sizes[i]
        return total_psi

    def get_psi_site_basis_by_order(self,t,order):
        keys = self.psis.keys()
        order_keys = []
        for key in keys:
            if len(key) == 2*order:
                order_keys.append(key)
        psi_total = self.get_psi_site_basis(t,order_keys.pop(0))
        for key in order_keys:
            psi_total += self.get_psi_site_basis(t,key)

        return psi_total

    def set_psi0(self,initial_state):
        """Creates the unperturbed wavefunction. This code does not 
            support initial states that are coherent super-positions of 
            eigenstates. To perform thermal averaging, recalculate spectra 
            for each initial state that contributes to the thermal ensemble.
        Args:
            initial_state (int): index for initial eigenstate in GSM
"""
        # initial state must be interpreted given the fact that masking may have been done
        try:
            trimmed_indices = np.where(self.trimming_masks[0])[0]
            initial_state = np.where(trimmed_indices == initial_state)[0]
        except AttributeError:
            pass

        t = self.efield_times[0] # can be anything of the correct length

        key = self.ordered_manifolds[0]
        psi0 = np.ones((1,t.size),dtype=complex)
        bool_mask = np.zeros(self.eigenvalues[key].size,dtype='bool')
        bool_mask[initial_state] = True

        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)

        t0 = 0

        if self.method == 'UF2':
            self.psi0 = perturbative_container(t,psi0,bool_mask,None,key,pdc,
                                               t0,interp_kind='zero',
                                               interp_left_fill=1)
        elif self.method == 'chebyshev':
            self.psi0 = cheb_perturbative_container(t,psi0,bool_mask,None,
                                                    key,pdc,t0,
                                                    interp_left_fill=1,
                                                    dom = self.doms[0])

    def load_eigensystem(self):
        """Load in known eigenvalues. Must be stored as a numpy archive file,
with keys: GSM, SEM, and optionally DEM.  The eigenvalues for each manifold
must be 1d arrays, and are assumed to be ordered by increasing energy. The
energy difference between the lowest energy ground state and the lowest 
energy singly-excited state should be set to 0
"""
        eigval_save_name = os.path.join(self.base_path,'eigenvalues.npz')
        eigvec_save_name = os.path.join(self.base_path,'eigenvectors.npz')
        with np.load(eigval_save_name) as eigval_archive:
            self.manifolds = list(eigval_archive.keys())
            self.eigenvalues = {key:eigval_archive[key] for key in self.manifolds}
        with np.load(eigvec_save_name) as eigvec_archive:
            self.eigenvectors = {key:eigvec_archive[key] for key in self.manifolds}

        if '0' in self.manifolds:
            self.ordered_manifolds = [str(i) for i in range(len(self.manifolds))]
        elif 'all_manifolds' in self.manifolds:
            self.ordered_manifolds = ['all_manifolds']
        else:
            self.ordered_manifolds = ['GSM','SEM','DEM','TEM','QEM']
        
        ### store original eigenvalues for recentering purposes
        self.original_eigenvalues = copy.deepcopy(self.eigenvalues)

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
            be stored as a .npz file, and must contain at least one array, each with three 
            indices: (upper manifold eigenfunction, lower manifold eigenfunction, 
            cartesian coordinate).  So far this code supports up to three manifolds, and
            therefore up to two dipole operators (connecting between manifolds)"""
        file_name = os.path.join(self.base_path,'mu.npz')
        file_name_pruned = os.path.join(self.base_path,'mu_pruned.npz')
        file_name_bool = os.path.join(self.base_path,'mu_boolean.npz')

        try:
            mu_boolean_archive = np.load(file_name_bool)
            # self.mu_boolean = {'ket':mu_boolean_archive['ket'],'bra':mu_boolean_archive['bra']}
            with np.load(file_name_bool) as mu_boolean_archive:
                self.mu_boolean = {key:mu_boolean_archive[key] for key in mu_boolean_archive.keys()}
            pruned = True
            file_name = file_name_pruned
        except FileNotFoundError:
            pruned = False

        with np.load(file_name) as mu_archive:
            self.H_mu = {key:mu_archive[key] for key in mu_archive.keys()}
        if pruned == False:
            self.mu_boolean = dict()
            for key in self.H_mu.keys():
                self.mu_boolean[key] = np.ones(self.H_mu[key].shape[:2],dtype='bool')
        
    ### Tools for recursively calculating perturbed wavepackets using TDPT

    def get_H_mu(self,pulse_number,key,rotating_flag=True):
        """Calculates the dipole matrix given the electric field polarization vector"""
        overlap_matrix = ClosedBaseClass.get_H_mu(self,pulse_number,key,
                                                 rotating_flag=rotating_flag)

        boolean_matrix = np.ones(overlap_matrix.shape,dtype='bool')

        return boolean_matrix, overlap_matrix

    def electric_field_mask(self,pulse_number,key,conjugate_flag=False):
        """This method determines which molecular transitions will be 
supported by the electric field.  We assume that the electric field has
0 amplitude outside the minimum and maximum frequency immplied by the 
choice of dt and num_conv_points.  Otherwise we will inadvertently 
alias transitions onto nonzero electric field amplitudes.
"""
        ef_mask_t0 = time.time()
        if conjugate_flag:
            ending_key, starting_key = key.split('_to_')
        else:
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
        
        diff = eig_ending[:,np.newaxis] - eig_starting[np.newaxis,:]
        
        if efield_t.size == 1:
            mask = np.ones(diff.shape,dtype='bool')
        else:
            # The only transitions allowed by the electric field shape are
            inds_allowed = np.where((diff - center > efield_w[0]) & (diff - center < efield_w[-1]))
            mask = np.zeros(diff.shape,dtype='bool')
            mask[inds_allowed] = 1

        self.efield_mask_time += time.time() - ef_mask_t0
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

    def mu_dot_psi(self,t,psi_in,pulse_number,up_flag=True,
                   next_manifold_mask = None):
        old_manifold_key = psi_in.manifold_key
        new_manifold_key=self.get_new_manifold_key(old_manifold_key,up_flag)
        m_nonzero = psi_in.bool_mask

        if up_flag:
            mu_key = old_manifold_key + '_to_' + new_manifold_key
        else:
            mu_key = new_manifold_key + '_to_' + old_manifold_key

        boolean_mu, mu = self.get_H_mu(pulse_number,mu_key,
                                       rotating_flag=up_flag)
            
        psi = self.get_psi(t,psi_in)

        conjugate_flag = not up_flag

        # e_mask = self.electric_field_mask(pulse_number,mu_key,
        #                                   conjugate_flag=conjugate_flag)
        # boolean_mu = boolean_mu * e_mask
        # mu = mu * e_mask

        mu, n_nonzero = self.mask_dipole_matrix(boolean_mu,mu,m_nonzero,
                                    next_manifold_mask=next_manifold_mask)

        psi = mu.dot(psi)
            
        return psi, n_nonzero

    def mu_exp_val(self,t,psis_in,pulse_number,*,mu_type = None,
                   ket_flag = None, up_flag = None):
        """Keyword arguments are provided simply to make compatible with 
            open system framework
"""
        bra_in, ket_in = psis_in
        bra_key = bra_in.manifold_key
        ket_key = ket_in.manifold_key

        bra_num = self.manifold_key_to_number(bra_key)
        ket_num = self.manifold_key_to_number(ket_key)
        
        bra_nonzero = bra_in.bool_mask
        ket_nonzero = ket_in.bool_mask

        if 'all_manifolds' in self.manifolds:
            pass
        elif np.abs(bra_num - ket_num) != 1:
            warnings.warn('Dipole only connects manifolds 0 to 1 or 1 to 2')
            return None
        t0 = time.time()
        if bra_num > ket_num:
            bra_new_mask = ket_in.bool_mask
            bra, n_mask = self.mu_dot_psi(t,bra_in,-1,up_flag=False,
                                          next_manifold_mask=bra_new_mask)
            ket = self.get_psi(t,ket_in)
        else:
            ket_new_mask = bra_in.bool_mask
            ket, n_mask = self.mu_dot_psi(t,ket_in,-1,up_flag=False,
                                          next_manifold_mask=ket_new_mask)
            bra = self.get_psi(t,bra_in)

        exp_val = np.sum(np.conjugate(bra) * ket,axis = 0)
            
        return exp_val
    
    def next_order(self,psi_in,*,up_flag=True,
                   next_manifold_mask = None,pulse_number = 0):
        """This function connects psi_p to psi_pj^(*) using a DFT convolution algorithm.

        Args:
            psi_in (psi_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            next_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold
        
        Return:
            psi_dict (psi_container): next-order wavefunction
"""
        t_next_order0 = time.time()
        # Rotating term excites the ket
        conjugate_flag = not up_flag

        if psi_in.pulse_number == None:
            old_pulse_time = -np.inf
        else:
            old_pulse_time = self.pulse_times[psi_in.pulse_number]
        pulse_time = self.pulse_times[pulse_number]

        if np.isclose(old_pulse_time,pulse_time):
            simultaneous = psi_in.simultaneous + 1
        else:
            simultaneous = 1

        t = self.efield_times[pulse_number] + pulse_time
        dt = self.dts[pulse_number]
        old_manifold_key = psi_in.manifold_key
        new_manifold_key=self.get_new_manifold_key(old_manifold_key,up_flag)

        input_pdc = psi_in.pdc
        output_pdc = input_pdc.copy()
        if conjugate_flag:
            output_pdc[pulse_number][1] += 1
        else:
            output_pdc[pulse_number][0] += 1

        center = self.centers[pulse_number]
        efield = self.efields[pulse_number] * np.exp(-1j*center*t)

        if conjugate_flag:
            efield = np.conjugate(efield)

        # factor of 1j comes from perturbation theory
        efield = 1j * efield
        
        t0 = time.time()
        psi,n_nonzero=self.mu_dot_psi(t,psi_in,pulse_number,up_flag=up_flag,
                                     next_manifold_mask=next_manifold_mask)
        
        t1 = time.time()
        self.next_order_expectation_time += t1-t0

        psi = psi * efield[np.newaxis,:]
        
        ### UF2 specific
        try:
            ev2 = self.eigenvalues['all_manifolds']
        except KeyError:
            ev2 = self.eigenvalues[new_manifold_key]

        exp_factor_ending = np.exp(1j*ev2[n_nonzero,np.newaxis]*t[np.newaxis,:])

        psi = psi*exp_factor_ending

        t0 = time.time()

        M = self.efield_times[pulse_number].size

        convolve_fun = self.heaviside_convolve_list[pulse_number].fft_convolve2

        if M == 1:
            pass
        else:
            if self.method == 'UF2':
                psi = convolve_fun(psi,d=self.dt)
            elif self.method == 'chebyshev':
                dom = self.doms[pulse_number] + pulse_time
                chp = ChebPoly(t,psi,dom = dom)
                chp.integrate()
                psi = chp(t)

        t1 = time.time()
        self.convolution_time += t1-t0

        if self.method == 'UF2':
            psi_out = perturbative_container(t,psi,n_nonzero,pulse_number,
                                new_manifold_key,output_pdc,pulse_time,
                                simultaneous=simultaneous)
        elif self.method == 'chebyshev':
            psi_out = cheb_perturbative_container(t,psi,n_nonzero,
                                                  pulse_number,
                                                  new_manifold_key,
                                                  output_pdc,pulse_time,
                                                  simultaneous=simultaneous,
                                                  dom = dom)

        self.next_order_time += time.time() - t_next_order0
    
        return psi_out

    def get_psi(self,t,psi_obj,*,original_H_basis=False):
        t_get_psi0 = time.time()
        manifold_key = psi_obj.manifold_key
        mask = psi_obj.bool_mask
        
        try:
            e = self.eigenvalues['all_manifolds'][mask]
            ev = self.eigenvectors['all_manifolds'][:,mask]
        except KeyError:
            e = self.eigenvalues[manifold_key][mask]
            ev = self.eigenvectors[manifold_key][:,mask]

        psi = psi_obj(t) * np.exp(-1j * e[:,np.newaxis]*t[np.newaxis,:])
        if original_H_basis:
            psi = ev.dot(psi)
        else:
            pass

        self.get_psi_time += time.time() - t_get_psi0

        return psi
        
