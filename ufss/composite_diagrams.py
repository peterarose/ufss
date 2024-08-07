#Standard python libraries
import os
import warnings
import copy
import time
import itertools
import functools

#Dependencies - numpy
import numpy as np

#This package
from ufss import DiagramGenerator

class CompositeDiagrams(DiagramGenerator):
    def __init__(self,*,detection_type='polarization'):
        DiagramGenerator.__init__(self,detection_type=detection_type)

    def set_interaction_types(self):
        """Sets a list of allowed interactions, given the maximum
            phase-discrimination condition. This takes on a similar role as  
            the attribute "ordered_interactions" in the diagram generator.
"""
        self.interaction_types = []
        for i in range(self.max_pdc.shape[0]):
            if self.max_pdc[i,0] > 0:
                self.interaction_types.append((i,'+'))
            if self.max_pdc[i,1] > 0:
                self.interaction_types.append((i,'-'))

        return None

    def set_pdc(self):
        DiagramGenerator.set_pdc(self)
        
        self.set_interaction_types()

        return None

    def include_higher_orders(self,pulse_orders):
        DiagramGenerator.include_higher_orders(self,pulse_orders)

        self.set_interaction_types()

        return None

    def get_output_pdc(self,pdc_in,pulse_number,pm_flag):
        pdc_out = pdc_in.copy()
        if pm_flag == '+':
            pdc_out[pulse_number,0] += 1
        elif pm_flag == '-':
            pdc_out[pulse_number,1] += 1
        else:
            raise Exception('Cannot parse pm_flag')

        return pdc_out

    def get_next_interactions(self,pdc_in):
        """Determines what the next valid interactions are, given an input
            partial pdc
        Args:
            pdc_in (tuple) : the input partial phase-discrimination condition
        Returns:
            (dict) : dictionary with next instructions for perturbative
                calculations; diciontary keys are the new partial phase-
                discrimination conditions
"""
        # convert tuple into an array
        pdc_in = np.array(pdc_in)
        next_interactions = dict()
        for pulse_num, pm_flag in self.interaction_types:
            
            if self.check_causality(pdc_in,pulse_num):            
                pdc_out = self.get_output_pdc(pdc_in,pulse_num,pm_flag)
                if self.check_partial_pdc(pdc_out,pulse_num):
                    pdc_key = self.pdc_arr_to_tup(pdc_out)
                    ket_key,bra_key = self.wavevector_dict[pm_flag]
                    next_interactions[pdc_key] = [(ket_key,pulse_num),
                                                  (bra_key,pulse_num)]

        return next_interactions

    def check_causality(self,pdc,next_pulse):
        """Check to make sure that an interaction with pulse numbered
            next_pulse will be non-zero, given causality. Any calculation 
            that obeys the partial phase-discrimination condition, pdc, can
            only be non-zero after all of the pulses that have contributed to
            it have arrived.
        Args:
            pdc (np.ndarray) : partial phase-discrimination condition
            next_pulse (int) : pulse number for next pulse interaction
        Returns:
            (bool) : True if interaction with next_pulse will give a non-zero
                calculation
"""
        current_pulses = pdc.sum(axis=1).astype('bool')
        num_pulses = pdc.shape[0]

        # when each pulse starts, accoridng to current pulse times
        efield_t0s = np.array([self.efield_times[i][0] + self.pulse_times[i]
                               for i in range(num_pulses)])
        # causality dictates that the calculation is 0 before any pulse
        # that contributes to the partial pdc
        t0 = np.max(current_pulses * efield_t0s)
        # next pulse turns off at tmax
        tmax = self.efield_times[next_pulse][-1]+self.pulse_times[next_pulse]
        ret_flag = True
        if tmax < t0:
            ret_flag = False
        return ret_flag

    def check_pdc_satisfy_signal_condition(self,pdc):
        """Checks to see whether or not the given pdc obeys the same signal
            detection condition as the baseline pdc originally specified
        Args:
            pdc (np.ndarray) : trial phase-discrimination condition
        Returns:
            (bool) : True if the input pdc is detected in conjunction with
                the baseline pdc, self.pdc
"""
        flag = False
        ref = self.contract_pdc(self.pdc)
        if self.contract_pdc(pdc) == ref:
            flag = True
        return flag

    def check_pulse_ordering_and_overlap(self,pdc_in,pdc_ref,pulse_num):
        """Check to see if an interaction with pulse number pulse_num, which
            resulted in a calculation that obeys pdc_in, will lead to zero
            calculations in the future, given that we are building towards a
            desired pdc of pdc_ref.
        Args:
            pdc_in (np.ndarray) : trial pdc
            pdc_ref (np.ndararay) : pdc of desired signal
            pulse_num (int) : pulse number that led to pdc_in
        Return:
            (bool) : True if future calculations that build towards pdc_ref 
                will be non-zero
"""
        remaining_pdc = pdc_ref - pdc_in
        if np.any(remaining_pdc < 0):
            # input pdc has more interactions with a pulse than reference
            return False
        
        remaining_pulses = remaining_pdc.sum(axis=1).astype('bool')

        output_t = self.efield_times[pulse_num] + self.pulse_times[pulse_num]
        output_t0 = output_t[0]
        for i in range(remaining_pulses.size):
            if remaining_pulses[i]:
                test_t = self.efield_times[i] + self.pulse_times[i]
                if test_t[-1] < output_t0:
                    return False
        return True

    def check_partial_pdc(self,pdc,pulse_num):
        if self.check_pdc_satisfy_signal_condition(pdc):
            return True
        # check that output_pdc is not higher-order than requested
        if np.sum(pdc) > self.highest_order:
            return False
        # exclude cases with too many interactions with the same pulse
        if not np.all(self.max_pdc - pdc >= 0):
            return False

        # Check to see if pulses overlap or are correctly ordered for any
        # signal order
        for key in self.all_pdcs:
            if self.check_pulse_ordering_and_overlap(pdc,np.array(key),
                                                     pulse_num):
                return True
            
        return False

    def generate_all_partial_pdcs(self):
        """Essentially a dummy calculation. Starting from a pdc of zeros,
            which is an unperturbed density matrix, this function cycles
            through each step in the calculation process, and generates the
            all of the partial pdcs that are needed to calculate the desired
            signal(s). This is meant to give a roadmap of how to use the
            composite diagram engine.
        Returns:
            (list) : a list of all partial pdcs, in the format of tuples
"""
        initial_pdc_arr = np.zeros(self.pdc.shape,dtype='int')
        initial_pdc_tup = self.pdc_arr_to_tup(initial_pdc_arr)
        all_keys = []
        old_keys = [initial_pdc_tup]
        for i in range(self.highest_order):
            new_keys = []
            for key in old_keys:
                next_instructions = self.get_next_interactions(key)
                new_keys += list(next_instructions.keys())
            new_keys = set(new_keys)
            old_keys = []
            for key in new_keys:
                all_keys.append(key)
                old_keys.append(key)

        return all_keys

    def include_additional_pdc(self,pdc):
        """Include another phase-discrimination condition for signal
            calculation purposes. 
        Args:
            pdc (tuple) : additional phase-discrimination condition
"""
        pdc_arr = self.pdc_tup_to_arr(pdc)
        if pdc_arr.sum() > self.highest_order:
            raise Exception('Requested additional pdc goes to higher order in perturbation theory than currently set. Please first use method include_higher_orders with the appropriate pulse orders, and then try this method again')
        if np.any(self.max_pdc - pdc_arr < 0):
            raise Exception('Requested additional pdc requires more pulse interactions than are currently set. Please first use method include_higher_orders with the appropriate pulse orders, and then try this method again')
        CD = CompositeDiagrams(detection_type=self.detection_type)
        CD.set_phase_discrimination(pdc)
        CD.set_pdc()
        CD.efield_times = self.efield_times
        CD.max_pdc = self.max_pdc
        CD.pulse_orders = self.pulse_orders
        CD.highest_order = self.highest_order
        CD.set_all_signal_matching_pdcs()
        
        new_signal_pdcs = self.signal_pdcs + CD.signal_pdcs
        self.set_signal_pdcs(new_signal_pdcs)
        return None

    def remove_signal_pdcs(self,pdc):
        """Remove pdcs satisfying a particular signal detection condition
        Args:
            pdc (tuple) : pdc condition to remove
"""
        ref_pdc_arr = self.pdc_tup_to_arr(pdc)
        new_signal_pdcs = []
        for pdc in self.signal_pdcs:
            pdc_arr = self.pdc_tup_to_arr(pdc)
            
            if self.contract_pdc(pdc_arr) == self.contract_pdc(ref_pdc_arr):
                pass
            else:
                new_signal_pdcs.append(pdc)
        self.set_signal_pdcs(new_signal_pdcs)
        return None
