import os
import time
import yaml

import numpy as np

from ufss import CompositeDiagrams

class ClosedBaseClass(CompositeDiagrams):
    def __init__(self,*,detection_type='polarization'):
        CompositeDiagrams.__init__(self,detection_type=detection_type)
        
        self.K_dict = {'u':self.up,'d':self.down}

        self.psis = dict()
        self.composite_psis = dict()

    def set_min_max_manifolds(self):
        if 'all_manifolds' in self.manifolds:
            pass
        else:
            manifold_integers = [self.manifold_key_to_number(man) for man
                                 in self.manifolds]
            self.maximum_manifold = max(manifold_integers)
            self.minimum_manifold = min(manifold_integers)

    def up(self,psi_in,*,next_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            psi_in (psi_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            next_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (psi_container): output from method next_order
"""

        return self.next_order(psi_in,up_flag=True,
                               next_manifold_mask = next_manifold_mask,
                               pulse_number = pulse_number)

    def down(self,psi_in,*,next_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            psi_in (psi_container): input density matrix
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            next_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (psi_container): output from method next_order
"""

        return self.next_order(psi_in,up_flag=False,
                               next_manifold_mask = next_manifold_mask,
                               pulse_number = pulse_number)

    def execute_diagram(self,instructions):
        num_instructions = len(instructions['ket'])+len(instructions['bra'])
        
        ket = self.psi0
        bra = self.psi0
        ketname = ''
        braname = ''
        ket_instructions = instructions['ket']
        bra_instructions = instructions['bra']
        for i in range(len(ket_instructions)):
            key, num = ket_instructions[i]
            ketname += key+str(num)
            # Try to re-use previous calculations, if they exist
            try:
                new_ket = self.psis[ketname]
            except KeyError:
                new_ket = self.K_dict[key](ket,pulse_number=num)
                self.psis[ketname] = new_ket
            ket = new_ket
                

        for i in range(len(bra_instructions)):
            key, num = bra_instructions[i]
            braname += key+str(num)
            # Try to re-use previous calculations, if they exist
            try:
                new_bra = self.psis[braname]
            except KeyError:
                new_bra = self.K_dict[key](bra,pulse_number=num)
                self.psis[braname] = new_bra
            bra = new_bra

        rho = (bra,ket)
        return rho

    def check_pulse_ordering_and_overlap(self,pdc_in,pdc_ref,pulse_num):
        """Have to replace the original method here, since some of the logic
            in the original method only applies for density matrix 
            calculations
"""
        remaining_pdc = pdc_ref - pdc_in
        if np.any(remaining_pdc < 0):
            # input pdc has more interactions with a pulse than reference
            return False
        else:
            return True

    def test_RWA(self,old_manifold_key,ins):
        if 'all_manifolds' in self.manifolds:
            test = True
        else:
            man_num = self.manifold_key_to_number(old_manifold_key)
            if ins[0][-1] == 'u':
                change = 1
            else:
                change = -1
            new_man_num = man_num + change
            if new_man_num < self.minimum_manifold:
                test = False
            elif new_man_num > self.maximum_manifold:
                test = False
            else:
                test = True

        return test

    def execute_composite_diagrams(self):
        self.composite_psis[self.psi0.pdc_tuple] = self.psi0
        psis = self.composite_psis
        old_psis = [self.psi0]
            
        for i in range(self.highest_order):
            new_psis = []
            pdcs = set()
            for old_psi in old_psis:
                old_pdc = old_psi.pdc_tuple
                next_instructions = self.get_next_interactions(old_pdc)
                for key in next_instructions.keys():
                    ket_ins, bra_ins = next_instructions[key]
                    test = self.test_RWA(old_psi.manifold_key,ket_ins)
                    if not test:
                        pass
                    else:
                        pdcs.add(key)
                        if key in psis:
                            # do not do calculations if they are already done
                            pass
                        else:
                            new_psi=self.execute_interaction(old_psi,ket_ins)
                            new_psis.append(new_psi)
            
            for new_psi in new_psis:
                if new_psi.pdc_tuple in psis.keys():
                    partial_psi = psis[new_psi.pdc_tuple]
                    new_psi = self.add_psis(partial_psi,new_psi)
                psis[new_psi.pdc_tuple] = new_psi

            old_psis = [psis[pdc] for pdc in pdcs]

        return None

    def execute_interaction(self,psi_in,interaction):
        """This method connects the input density matrix of order n to a
            density matrix that contributes to order n + 1, according to the
            specified interaction type and pulse number
        Args:
            rho_in (DensityMatrices) : input density matrix
            interaction (tuple) : specifies interaction type and pulse 
                number in format (str, int)
"""
        int_type, pulse_num = interaction
        int_type = int_type.strip('K')
        return self.K_dict[int_type](psi_in,pulse_number = pulse_num)

    def set_signal_pdcs(self,pdcs):
        """Specific to closed systems, needed for calculating wavefunctions
        Args:
            pdcs (list) : list of pdcs as np.ndarrays or tuples
"""
        if type(pdcs[0]) is tuple:
            self.singal_pdcs = pdcs
            signal_arrs = [self.pdc_tup_to_arr(pdc) for pdc in pdcs]
        else:
            signal_arrs = pdcs
            self.signal_pdcs = [self.pdc_arr_to_tup(pdc) for pdc in pdcs]
        
        cc_signal_pdcs = [self.pdc_arr_to_tup(arr[:,::-1])
                          for arr in signal_arrs]
        self.all_pdcs = list(set(self.signal_pdcs + cc_signal_pdcs))

        sh = signal_arrs[0].shape
        sh = sh + (len(self.all_pdcs),)
        all_pdcs_arr = np.zeros(sh,dtype='int')
        for i in range(len(self.all_pdcs)):
            all_pdcs_arr[...,i] = self.pdc_tup_to_arr(self.all_pdcs[i])
        self.max_pdc = np.max(all_pdcs_arr,axis=-1)
        self.set_interaction_types()

        return None

    def remove_calculations_by_pulse_number(self,pulse_number):
        num = str(pulse_number)
        keys = self.psis.keys()
        keys_to_remove = []
        for key in keys:
            flag = key.find(num)
            if flag >= 0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.psis.pop(key)

    def remove_composite_calculations_by_pulse_number(self,pulse_number):
        keys = self.composite_psis.keys()
        keys_to_remove = []
        for key in keys:
            r_ints,cr_ints = key[pulse_number]
            if r_ints > 0 or cr_ints > 0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.composite_psis.pop(key)

    def set_current_diagram_instructions(self,arrival_times):
        self.current_instructions = self.get_wavefunction_diagram_dict(arrival_times)

    # def fluorescence_detection_signal(self,bra_dict,ket_dict,*,time_index = -1):
    #     """Calculate inner product given an input bra and ket dictionary
    #         at a time given by the time index argument.  Default is -1, since
    #         2DFS is concerned with the amplitude of arriving in the given manifold
    #         after the pulse has finished interacting with the system."""
    #     bra = np.conjugate(bra_dict['psi'][:,-1])
    #     ket = ket_dict['psi'][:,-1]
    #     return np.dot(bra,ket)

    def reset(self):
        self.psis = dict()
        self.composite_psis = dict()

    def manifold_key_to_number(self,key):
        num = self.ordered_manifolds.index(key)
        return num

    def manifold_number_to_key(self,num):
        if num < 0:
            raise Exception('Manifold number must be positive')
        key = self.ordered_manifolds[num]
        return key

    def get_new_manifold_key(self,old_manifold_key,up_flag):
        if up_flag:
            change = 1
        else:
            change = -1
        if 'all_manifolds' in self.manifolds:
            new_manifold_key = 'all_manifolds'
        else:
            old_manifold = self.manifold_key_to_number(old_manifold_key)
            new_manifold = old_manifold + change
            new_manifold_key = self.manifold_number_to_key(new_manifold)
        return new_manifold_key
    
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
        if type(mu) is list:
            if np.all(pol == x):
                overlap_matrix = mu[0]#.copy()
            elif np.all(pol == y):
                overlap_matrix = mu[1]#.copy()
            elif np.all(pol == z):
                overlap_matrix = mu[2]#.copy()
            else:
                overlap_matrix = mu[0]*pol[0] + mu[1]*pol[1] + mu[2]*pol[2]
        else:
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
