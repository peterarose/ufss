import os
import time
import yaml

import numpy as np
from scipy.sparse import csr_matrix

from ufss import CompositeDiagrams

class OpenBaseClass(CompositeDiagrams):
    def __init__(self,*,detection_type='polarization'):
        CompositeDiagrams.__init__(self,detection_type=detection_type)

        self.number_of_diagrams_calculated = 0
        
        self.KB_dict = {'Bu':self.bra_up,'Ku':self.ket_up,
                        'Kd':self.ket_down,'Bd':self.bra_down}

        self.rhos = dict()
        self.composite_rhos = dict()

    def load_H_eigensystem(self):
        """Look for the eigenvalues and eigenvectors of H, and load them 
            if available. Useful as a convenience for assigning spectral 
            positions"""
        parent_dir = os.path.split(self.base_path)[0]
        H_e_name = os.path.join(parent_dir,'closed','eigenvalues.npz')
        H_ev_name = os.path.join(parent_dir,'closed','eigenvectors.npz')
        try:
            with np.load(H_e_name) as eigval_archive:
                manifolds = list(eigval_archive.keys())
                self.H_eigenvalues = {key:eigval_archive[key] for key in manifolds}
        except FileNotFoundError:
            pass
        try:
            with np.load(H_ev_name) as eigvec_archive:
                manifolds = list(eigvec_archive.keys())
                self.H_eigenvectors = {key:eigvec_archive[key] for key in manifolds}
        except FileNotFoundError:
            pass

    def load_fl_yield(self):
        parent_dir = os.path.split(self.base_path)[0]
        file_name = os.path.join(parent_dir,'closed','fluorescence_yield.npz')

        with np.load(file_name) as fl_archive:
            self.fluorescence_yield = {key:fl_archive[key] for key in fl_archive.keys()}

    ### common method for loading and using the correct mu
    def load_H_mu(self):
        parent_dir = os.path.split(self.base_path)[0]
        params_file = os.path.join(parent_dir,'params.yaml')
        with open(params_file) as yamlstream:
            params = yaml.load(yamlstream,Loader=yaml.SafeLoader)
        if 'ManualD' in params.keys():
            if params['H_eigenbasis']:
                file_name = os.path.join(parent_dir,'closed','mu.npz')
            else:
                file_name = os.path.join(parent_dir,'closed',
                                     'mu_original_H_basis.npz')
        elif ('site_bath' in params['bath'].keys() or 
            'vibration_bath' in params['bath'].keys()):
            file_name = os.path.join(parent_dir,'closed','mu.npz')
        else:
            file_name = os.path.join(parent_dir,'closed',
                                     'mu_original_H_basis.npz')

        with np.load(file_name) as mu_archive:
            self.H_mu = {key:mu_archive[key] for key in mu_archive.keys()}

    def get_H_mu(self,pulse_number,key,rotating_flag=True):
        """Calculates the dipole matrix given the electric field 
            polarization vector"""
        t0 = time.time()
        pol = self.polarization_sequence[pulse_number]

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        try:
            mu = self.H_mu[key]
        except KeyError:
            try:
                key = 'up'
                mu = self.H_mu[key]
            except KeyError:
                key = 'ket_up'
                mu = self.H_mu[key]
            
        if np.all(pol == x):
            overlap_matrix = mu[:,:,0]#.copy()
        elif np.all(pol == y):
            overlap_matrix = mu[:,:,1]#.copy()
        elif np.all(pol == z):
            overlap_matrix = mu[:,:,2]#.copy()
        else:
            overlap_matrix = np.tensordot(mu,pol,axes=(-1,0))

        if not rotating_flag:
            overlap_matrix = overlap_matrix.T

        t1 = time.time()
        self.dipole_time += t1-t0

        return overlap_matrix

    def dipole_matrix(self,pulse_number,key,ket_flag=True,up_flag=True):
        """Calculates the dipole matrix given the electric field polarization vector,
            if ket_flag = False then uses the bra-interaction"""
        t0 = time.time()
        pol = self.polarization_sequence[pulse_number]
            
        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        try:
            mu = self.mu[key]
        except KeyError:
            if ket_flag:
                key = 'ket'
            else:
                key = 'bra'
            if up_flag:
                key += '_up'
            else:
                key += '_down'
            mu = self.mu[key]
            
        if np.all(pol == x):
            overlap_matrix = mu[0]#.copy()
        elif np.all(pol == y):
            overlap_matrix = mu[1]#.copy()
        elif np.all(pol == z):
            overlap_matrix = mu[2]#.copy()
        else:
            overlap_matrix = mu[0]*pol[0] + mu[1]*pol[1] + mu[2]*pol[2]

        t1 = time.time()
        self.dipole_time += t1-t0

        return overlap_matrix

    def get_mu(self,pulse_number,old_manifold_key,new_manifold_key,
               ket_flag=True,up_flag=True,mu_type='L'):
        if mu_type=='H':
            old_k,old_b = old_manifold_key.split(',')
            new_k,new_b = new_manifold_key.split(',')

            if ket_flag:
                if up_flag:
                    H_mu_key = old_k + '_to_' + new_k
                else:
                    H_mu_key = new_k + '_to_' + old_k

                rotating_flag = up_flag
            else:
                if up_flag:
                    H_mu_key = old_b + '_to_' + new_b
                else:
                    H_mu_key = new_b + '_to_' + old_b
                rotating_flag = not up_flag

            overlap_matrix = self.get_H_mu(pulse_number,
                                H_mu_key,rotating_flag=rotating_flag)
        else:
            mu_key = old_manifold_key + '_to_' + new_manifold_key
            overlap_matrix = self.dipole_matrix(pulse_number,mu_key,
                                        ket_flag=ket_flag,up_flag=up_flag)

        return overlap_matrix

    ### Generic methods for calculating individual diagrams

    def execute_diagram(self,instructions):
        self.number_of_diagrams_calculated += 1
        r = self.rho0
        name = ''
        for i in range(len(instructions)):
            key, num = instructions[i]
            name += key+str(num)
            # Try to re-use previous calculations, if they exist
            try:
                new_r = self.rhos[name]
            except KeyError:
                new_r = self.KB_dict[key](r,pulse_number=num)
                self.rhos[name] = new_r
            r = new_r
            
        return r

    def execute_composite_diagrams(self):
        rhos = self.composite_rhos
        old_rhos = [self.rho0]
            
        for i in range(self.highest_order):
            new_rhos = []
            pdcs = set()
            for old_rho in old_rhos:
                old_pdc = old_rho.pdc_tuple
                next_instructions = self.get_next_interactions(old_pdc)
                for key in next_instructions.keys():
                    pdcs.add(key)
                    if key in rhos:
                        # do not do calculations if they are already done
                        pass
                    else:
                        ket_ins,bra_ins = next_instructions[key]
                        new_rho_k =self.execute_interaction(old_rho,ket_ins)
                        new_rho_b =self.execute_interaction(old_rho,bra_ins)
                        new_rhos.append(new_rho_k)
                        new_rhos.append(new_rho_b)
            
            for new_rho in new_rhos:
                if new_rho.pdc_tuple in rhos.keys():
                    partial_rho = rhos[new_rho.pdc_tuple]
                    new_rho = self.add_rhos(partial_rho,new_rho)
                rhos[new_rho.pdc_tuple] = new_rho

            old_rhos = [rhos[pdc] for pdc in pdcs]

        return None

    def remove_calculations_by_pulse_number(self,pulse_number):
        num = str(pulse_number)
        keys = self.rhos.keys()
        keys_to_remove = []
        for key in keys:
            flag = key.find(num)
            if flag >= 0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.rhos.pop(key)

    def remove_composite_calculations_by_pulse_number(self,pulse_number):
        keys = self.composite_rhos.keys()
        keys_to_remove = []
        for key in keys:
            ket_ints,bra_ints = key[pulse_number]
            if ket_ints > 0 or bra_ints > 0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.composite_rhos.pop(key)

    def set_current_diagram_instructions(self,arrival_times):
        self.diagram_generation_counter += 1
        t0a = time.time()
        self.current_instructions = self.get_diagram_dict(arrival_times)
        t0b = time.time()
        self.diagram_generation_time = t0b - t0a

    def reset(self):
        self.rhos = dict()
        self.composite_rhos = dict()

    def convert_mu_keys(self):
        old_keys = list(self.mu.keys())
        print(old_keys)
        for key in old_keys:
            key1,key2 = key.split('_to_')
            old_k,old_b = key1
            new_k,new_b = key2
            new_key = ''.join([old_k,',',old_b,'_to_',new_k,',',new_b])
            self.mu[new_key] = self.mu.pop(key)
            self.mu_boolean[new_key] = self.mu_boolean.pop(key)

    def set_maximum_manifold(self,num='auto'):
        
        if num == 'auto':
            if 'all_manifolds' in self.manifolds:
                self.maximum_manifold = np.inf
            else:
                manifold_numbers = []
                for key in self.manifolds:
                    a,b = key.split(',')
                    if a == b:
                        manifold_numbers.append(int(a))
                self.maximum_manifold = max(manifold_numbers)
        else:
            self.maximum_manifold = num

    def manifold_array_to_key(self,manifold):
        """Inverse of self.manifold_key_to_array"""
        if manifold.size != 2 or manifold.dtype != int:
            raise Exception('manifold array must contain exactly 2 integer') 
        return str(manifold[0]) + ',' + str(manifold[1])

    def manifold_key_to_array(self,key):
        """Key must be a string containing two integers separated by a comma, 
            the first describing
            the ket manifold, the second the bra manifold.  If the density 
            matrix is represented in the full space, rather than being divided
            into manifolds, the first integer reperesents the total number of
            excitations to the ket side, and the second integer represents 
            the sum of all excitations to the bra side."""
        return np.array([int(char) for char in key.split(',')],dtype=int)

    def execute_interaction(self,rho_in,interaction):
        """This method connects the input density matrix of order n to a
            density matrix that contributes to order n + 1, according to the
            specified interaction type and pulse number
        Args:
            rho_in (DensityMatrices) : input density matrix
            interaction (tuple) : specifies interaction type and pulse 
                number in format (str, int)
"""
        int_type, pulse_num = interaction
        return self.KB_dict[int_type](rho_in,pulse_number = pulse_num)
            
    def ket_up(self,rho_in,*,next_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (perturbative_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) 
            next_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (perturbative_container): output from method next_order
"""

        return self.next_order(rho_in,ket_flag=True,up_flag=True,
                               next_manifold_mask = next_manifold_mask,
                               pulse_number = pulse_number)

    def ket_down(self,rho_in,*,next_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (perturbative_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) 
            next_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (perturbative_container): output from method next_order
"""

        return self.next_order(rho_in,ket_flag=True,up_flag=False,
                               next_manifold_mask = next_manifold_mask,
                               pulse_number = pulse_number)

    def bra_up(self,rho_in,*,next_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (perturbative_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)
            next_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (perturbative_container): output from method next_order
"""

        return self.next_order(rho_in,ket_flag=False,up_flag=True,
                               next_manifold_mask = next_manifold_mask,
                               pulse_number = pulse_number)

    def bra_down(self,rho_in,*,next_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            rho_in (perturbative_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...)
            next_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (perturbative_container): output from method next_order
"""

        return self.next_order(rho_in,ket_flag=False,up_flag=False,
                               next_manifold_mask = next_manifold_mask,
                               pulse_number = pulse_number)

    def get_rho_by_key(self,t,key,*,reshape=True,original_L_basis=True):
        try:
            rho = self.composite_rhos[key]
        except KeyError:
            rho = self.rhos[key]
        return self.get_rho(t,rho,reshape=reshape,
                            original_L_basis=original_L_basis)

    def get_rho_by_order(self,t,order,*,reshape=True,original_L_basis=True):
        if len(self.composite_rhos) > 0:
            keys = self.composite_rhos.keys()
            order_keys = []
            for key in keys:
                pdc = self.pdc_tup_to_arr(key)
                if pdc.sum() == order:
                    order_keys.append(key)
        else:
            keys = self.rhos.keys()
            order_keys = []
            for key in keys:
                if len(key) == 3*order:
                    order_keys.append(key)
        
        rho_total = self.get_rho_by_key(t,order_keys.pop(0),reshape=reshape,
                                        original_L_basis=original_L_basis)
        for key in order_keys:
            rho_total += self.get_rho_by_key(t,key,reshape=reshape,
                                        original_L_basis=original_L_basis)

        return rho_total

    def check_sparsity(self,mat):
        csr_mat = csr_matrix(mat)
        sparsity = csr_mat.nnz / (csr_mat.shape[0]*csr_mat.shape[1])
        if sparsity < self.sparsity_threshold:
            return True
        else:
            return False

    def get_new_manifold_key(self,old_manifold_key,ket_flag,up_flag):
        
        if up_flag:
            change = 1
        else:
            change = -1
        if ket_flag:
            manifold_change = np.array([change,0],dtype=int)
        else:
            manifold_change = np.array([0,change],dtype=int)
        old_manifold = self.manifold_key_to_array(old_manifold_key)
        new_manifold = old_manifold + manifold_change
        new_manifold_key = self.manifold_array_to_key(new_manifold)
        
        return new_manifold_key
