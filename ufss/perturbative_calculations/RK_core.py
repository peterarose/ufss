#Standard python libraries
import os
import warnings
import time

#Dependencies - numpy, scipy
import numpy as np
from scipy.sparse import csr_matrix, identity, load_npz
from scipy.sparse.linalg import eigsh
from scipy.integrate import RK45

#Other parts of this code
from ufss.perturbative_calculations import DPBaseClass
from ufss.perturbative_calculations import ClosedBaseClass
from ufss.perturbative_calculations import RK_perturbative_container

class RKClosedEngine(ClosedBaseClass,DPBaseClass):
    """This class is designed to calculate perturbative wavepackets in the
        light-matter interaction given the hamiltonians of each optical
        manifold

    Args:
        file_path (string): path to folder containing eigenvalues and the
            dipole operator for the system Hamiltonian
        initial_state (int): index of initial state for psi^0 NOT IMPLEMENTED YET

"""
    def __init__(self,file_path,*,initial_state=0,
                 detection_type = 'polarization'):
        DPBaseClass.__init__(self,file_path,detection_type = detection_type)
        ClosedBaseClass.__init__(self,detection_type = detection_type)
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
        self.RK45_step_time = 0

        self.sparsity_threshold = 0.1

        self.interaction_picture_calculations = True

        self.load_mu()

        self.load_H()

        self.set_min_max_manifolds()

        self.rtol = 1E-6
        self.atol = 1E-6

        self.time_to_extend = 0
        self.time_for_next_order = 0

    def check_sparsity(self,mat):
        csr_mat = csr_matrix(mat)
        sparsity = csr_mat.nnz / (csr_mat.shape[0]*csr_mat.shape[1])
        if sparsity < self.sparsity_threshold:
            return True
        else:
            return False

    def set_efields(self,times_list,efields_list,centers_list,
                    phase_discrimination,*,reset_calculations = True,
                    plot_fields = False):

        DPBaseClass.set_efields(self,times_list,efields_list,centers_list,
                                 phase_discrimination,
                                 reset_calculations = reset_calculations,
                                 plot_fields = plot_fields)
        
        # Initialize unperturbed density matrix
        self.set_psi0_auto()

    def add_impulsive_psis(self,a,b):
        """Add two perturbative_containers together
"""
        manifold_key = a.manifold_key
        pdc = a.pdc
        sim_a = a.simultaneous
        sim_b = b.simultaneous
        if sim_a != sim_b:
            raise Exception('Cannot add impulsive psis if they do not have the same number of simultaneous interactions')
        else:
            simultaneous = sim_a
        pna = a.pulse_number
        pnb = a.pulse_number
        ta = self.pulse_times[pna]
        tb = self.pulse_times[pnb]
        if np.allclose(ta,tb):
            t = a.t[1:2]
            psi_a = a.f[:,1:2]
            psi_b = b.f[:,1:2]
            pulse_number = max(pna,pnb)
        else:
            raise Exception('Cannot add two impulsive psis unless they have the same value of t0')

        psi = psi_a + psi_b
        psi *= 2 * simultaneous
        
        ab=RK_perturbative_container(t,psi,pulse_number,manifold_key,pdc,
                                     simultaneous = simultaneous)
        ab.one_time_step_function = self.one_time_step_function
        return ab
        

    def add_psis(self,a,b):
        """Add two density matrix objects together
"""
        if not np.allclose(a.pdc,b.pdc):
            raise Exception('Cannot add density matrices with different phase-discrimination conditions')
        if 'all_manifolds' in self.manifolds:
            pass
        else:
            if a.manifold_key != b.manifold_key:
                raise Exception('Cannot add perturbative_container objects that exist in different manifolds')
            
        if a.impulsive:
            return self.add_impulsive_psis(a,b)
        
        manifold_key = a.manifold_key
        
        pdc = a.pdc
        
        tmin = min(a.t[0],b.t[0])
        tmax = max(a.t[-1],b.t[-1])
        dt = min(a.t[1]-a.t[0],b.t[1]-b.t[0])
        t = np.arange(tmin,tmax+dt*0.9,dt)
        psi = a(t) + b(t)

        pna = a.pulse_number
        pnb = b.pulse_number
        pulse_number = max(pna,pnb)
        
        ab = RK_perturbative_container(t,psi,pulse_number,manifold_key,pdc)
        ab.one_time_step_function = self.one_time_step_function
        return ab

    def get_bottom_eigenvector(self):
        try:
            H = self.H['all_manifolds']
        except KeyError:
            H = self.H['0']
        if H.shape == (1,1):
            e = H[0,0]
            ev = np.array([[1]])
        else:
            e, ev = eigsh(H,k=1,which='SM',maxiter=10000)

        for key in self.H.keys():
            sh = self.H[key].shape
            if type(self.H[key]) is np.ndarray:
                self.H[key] -= e * np.eye(sh[0])
            else:
                self.H[key] -= e * identity(sh[0])

        v = ev[:,0]
        return e, v

    def set_psi0_auto(self):
        try:
            psi0 = np.load(os.path.join(self.base_path,'psi0.npy'))
        except FileNotFoundError:
            e0, psi0 = self.get_bottom_eigenvector()
        self.set_psi0_manual(psi0)

    def set_psi0_manual(self,psi0):
        t = np.array([-np.inf,0,np.inf])
        psi0 = psi0[:,np.newaxis] * np.ones((psi0.size,t.size))
        pulse_number = None
        manifold_key = self.ordered_manifolds[0]
        pdc = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)
        
        self.psi0 = RK_perturbative_container(t,psi0,pulse_number,
                                              manifold_key,pdc,
                                              interp_kind = 'zero')

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
            be stored as a .npz file, and must contain at least one array, each with three 
            indices: (new manifold index, old manifold eigenfunction, 
            cartesian coordinate)."""
        try:
            file_name = os.path.join(self.base_path,'mu_site_basis.npz')
            with np.load(file_name) as mu_archive:
                self.H_mu={key:mu_archive[key] for key in mu_archive.keys()}
        except FileNotFoundError:
            try:
                file_name = os.path.join(self.base_path,'mu_original_H_basis.npz')
                with np.load(file_name) as mu_archive:
                    self.H_mu = {key:mu_archive[key] for key
                                 in mu_archive.keys()}
            except FileNotFoundError:
                file_name = os.path.join(self.base_path,'mu.npz')
                with np.load(file_name) as mu_archive:
                    self.H_mu = {key:mu_archive[key] for key
                                 in mu_archive.keys()}
        sparse_flags = []
        for key in self.H_mu.keys():
            mu_2D = np.sum(np.abs(self.H_mu[key])**2,axis=-1)
            sparse_flags.append(self.check_sparsity(mu_2D))
        sparse_flags = np.array(sparse_flags)
        if np.allclose(sparse_flags,True):
            self.sparse_mu_flag = True
        else:
            self.sparse_mu_flag = False

        for key in self.H_mu.keys():
            mu_x = self.H_mu[key][...,0]
            mu_y = self.H_mu[key][...,1]
            mu_z = self.H_mu[key][...,2]

            if self.sparse_mu_flag:
                self.H_mu[key] = [csr_matrix(mu_x),csr_matrix(mu_y),
                                  csr_matrix(mu_z)]
            else:
                self.H_mu[key] = [mu_x,mu_y,mu_z]

        print('RKE_sparse_mu_flag',self.sparse_mu_flag)

    def load_H(self):
        """Load in known eigenvalues. Must be stored as a numpy archive file,
with keys: GSM, SEM, and optionally DEM.  The eigenvalues for each manifold
must be 1d arrays, and are assumed to be ordered by increasing energy. The
energy difference between the lowest energy ground state and the lowest 
energy singly-excited state should be set to 0
"""
        H_file_name = os.path.join(self.base_path,'H.npz')
        try:
            with np.load(H_file_name,allow_pickle=True) as H_archive:
                self.H = dict()
                for key in H_archive.keys():
                    H = H_archive[key]
                    if H.dtype == np.dtype('O'):
                        self.H[key] = H[()]
                    else:
                        if self.check_sparsity(H):
                            self.H[key] = csr_matrix(H)
                        else:
                            self.H[key] = H
        except:
            self.H = {'all_manifolds':load_npz(H_file_name)}
        self.manifolds = list(self.H.keys())

        if '0' in self.manifolds:
            self.ordered_manifolds = [str(i) for i in range(len(self.manifolds))]
        elif 'all_manifolds' in self.manifolds:
            self.ordered_manifolds = ['all_manifolds']
        else:
            self.ordered_manifolds = ['GSM','SEM','DEM','TEM','QEM']

    def dH(self,t,psi):
        try:
            H = self.H['all_manifolds']
        except KeyError:
            H = self.H[psi.manifold_key]
        return -1j*H.dot(psi)

    def get_dH_manual(self,manifold_key):
        try:
            H = self.H['all_manifolds']
        except KeyError:
            H = self.H[manifold_key]

        def H_fun(t,psi):
            return -1j*H.dot(psi)

        return H_fun

    def one_time_step_function(self,psi0,t0,tf,*,manifold_key = None):
        num_steps = 0
        if manifold_key == None:
            rk45 = RK45(self.dH,t0,psi0,tf,atol=self.atol,rtol=self.rtol)
        else:
            dH = self.get_dH_manual(manifold_key)
            rk45 = RK45(dH,t0,psi0,tf,atol=self.atol,rtol=self.rtol)
        if rk45.direction == 1:
            while rk45.t < tf:
                rk45.step()
                num_steps += 1
        else:
            while rk45.t > tf:
                rk45.step()
                num_steps += 1
        psi_final = rk45.y
        
        return psi_final


    def mu_dot_psi(self,t,psi_in,pulse_number,up_flag=True,
                   next_manifold_mask = None):
        old_manifold_key = psi_in.manifold_key
        new_manifold_key=self.get_new_manifold_key(old_manifold_key,up_flag)

        if up_flag:
            mu_key = old_manifold_key + '_to_' + new_manifold_key
        else:
            mu_key = new_manifold_key + '_to_' + old_manifold_key

        mu = self.get_H_mu(pulse_number,mu_key,rotating_flag=up_flag)
            
        psi = self.get_psi(t,psi_in)

        if psi.size == 0:
            psi = np.array([])
        else:
            psi = mu.dot(psi)
            
        return psi

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

        if 'all_manifolds' in self.manifolds:
            pass
        elif np.abs(bra_num - ket_num) != 1:
            warnings.warn('Dipole only connects manifolds 0 to 1 or 1 to 2')
            return None
        t0 = time.time()
        if bra_num > ket_num:
            bra = self.mu_dot_psi(t,bra_in,-1,up_flag=False)
            ket = self.get_psi(t,ket_in)
        else:
            ket = self.mu_dot_psi(t,ket_in,-1,up_flag=False)
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
        psi = self.mu_dot_psi(t,psi_in,pulse_number,up_flag=up_flag,
                              next_manifold_mask=next_manifold_mask)
        
        t1 = time.time()
        self.next_order_expectation_time += t1-t0

        psi = psi * efield[np.newaxis,:]

        M = t.size

        if M == 1:
            next_psi = psi
        else:
            next_psi = np.zeros(psi.shape,dtype='complex')

            dt = self.dts[pulse_number]
            
            if self.method == 'Euler':
                next_psi[:,0] = psi[:,0] * dt
                for i in range(1,t.size):
                    psi0 = next_psi[:,i-1]
                    t0 = t[i-1]
                    ta = time.time()
                    next_psi[:,i] = self.one_time_step_function(psi0,t0,t[i],manifold_key=new_manifold_key)

                    tb = time.time()
                    self.RK45_step_time += tb - ta

                    next_psi[:,i] += psi[:,i] * dt

            elif self.method == 'UF2':
                ref_time = pulse_time
                for i in range(t.size):
                    next_psi[:,i]=self.one_time_step_function(psi[:,i],t[i],
                                    ref_time,manifold_key=new_manifold_key)

                t0 = time.time()

                M = self.efield_times[pulse_number].size

                fft_convolve_fun = self.heaviside_convolve_list[pulse_number].fft_convolve2

                next_psi = fft_convolve_fun(next_psi,d=dt)

                t1 = time.time()
                self.convolution_time += t1-t0

                for i in range(t.size):
                    next_psi[:,i]=self.one_time_step_function(next_psi[:,i],
                                ref_time,t[i],manifold_key=new_manifold_key)

        psi_out = RK_perturbative_container(t,next_psi,pulse_number,
                                            new_manifold_key,output_pdc,
                                            simultaneous=simultaneous)
        psi_out.one_time_step_function = self.one_time_step_function
    
        return psi_out

    def get_psi(self,t,psi_obj):
        psi = psi_obj(t)
        return psi
