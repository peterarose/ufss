#Standard python libraries
import os
import warnings
import copy
import time
import itertools
import functools

#Dependencies - numpy, scipy, matplotlib, pyfftw
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq
from scipy.interpolate import interp1d as sinterp1d
import scipy
import pyx

class DiagramDrawer:
    def __init__(self):
        self.draw_functions = {'Ku':self.draw_Ku,'Kd':self.draw_Kd,'Bu':self.draw_Bu,'Bd':self.draw_Bd}
        self.pulse_labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
                             'n','o','p','q','r','s','t','u','v','w','x','y','z']

    def save_diagram(self,diagram,*,folder_name = ''):
        os.makedirs(folder_name,exist_ok=True)
        self.c = pyx.canvas.canvas()
        filename = ''
        interaction_counter = 0
        for KB,n in diagram:
            self.draw_functions[KB](n,interaction_counter)
            filename += KB + str(n)
            interaction_counter += 1
        self.c.writePDFfile(os.path.join(folder_name,filename))

    def save_diagrams(self,diagrams,*,folder_name=''):
        for diagram in diagrams:
            self.save_diagram(diagram,folder_name=folder_name)

    def display_diagram(self,diagram):
        self.c = pyx.canvas.canvas()
        interaction_counter = 0
        for KB,n in diagram:
            self.draw_functions[KB](n,interaction_counter)
            interaction_counter += 1
        display(self.c)

    def display_diagrams(self,diagrams):
        for diagram in diagrams:
            self.display_diagram(diagram)

    def double_sided(self,pulse_num):
        self.c.stroke(pyx.path.line(0,pulse_num,0,pulse_num+1))
        self.c.stroke(pyx.path.line(1, pulse_num, 1, pulse_num+1))

    def K_circle(self,pulse_num):
        self.c.fill(pyx.path.circle(0,pulse_num+0.5,0.1))

    def B_circle(self,pulse_num):
        self.c.fill(pyx.path.circle(1,pulse_num+0.5,0.1))

    def right_arrow(self,x,y,pulse_num):
        xf, yf = (x+0.5,y+0.5)
        self.c.stroke(pyx.path.line(x,y,xf,yf),[pyx.deco.earrow(size=0.3)])
        return xf,yf

    def left_arrow(self,x,y,pulse_num):
        xf,yf = (x-.5,y+.5)
        self.c.stroke(pyx.path.line(x,y,xf,yf),[pyx.deco.earrow(size=0.3)])
        return xf,yf

    def pulse_text(self,x,y,pulse_num):
        text = self.pulse_labels[pulse_num]
        self.c.text(x,y,text)

    def draw_Bd(self,pulse_num,interaction_num):
        m = interaction_num
        n = pulse_num
        self.double_sided(m)
        self.B_circle(m)
        x,y = (1,m+0.5)
        xf,yf = self.right_arrow(x,y,m)
        self.pulse_text(xf-0.1,yf-0.5,n)

    def draw_Bu(self,pulse_num,interaction_num):
        m = interaction_num
        n = pulse_num
        self.double_sided(m)
        self.B_circle(m)
        x,y = (1.5,m)
        self.left_arrow(x,y,m)
        self.pulse_text(x-0.1,y+0.1,n)

    def draw_Kd(self,pulse_num,interaction_num):
        m = interaction_num
        n = pulse_num
        self.double_sided(m)
        self.K_circle(m)
        xi,yi = (0,m+0.5)
        xf,yf = self.left_arrow(xi,yi,m)
        self.pulse_text(xf-0.1,yf-0.5,n)

    def draw_Ku(self,pulse_num,interaction_num):
        m = interaction_num
        n = pulse_num
        self.double_sided(m)
        self.K_circle(m)
        xi,yi = (-0.5,m)
        xf,yf = self.right_arrow(xi,yi,m)
        self.pulse_text(xi-0.1,yi+0.1,n)

    

class DiagramGenerator(DiagramDrawer):
    """

    Args:
        detection_type (str): either 'polarization' or 'fluorescence'

"""
    def __init__(self,*,detection_type = 'polarization'):
        DiagramDrawer.__init__(self)

        # Code will not actually function until the following three empty lists are set by the user
        self.efield_times = [] #initialize empty list of times assoicated with each electric field shape
        self.efield_wavevectors = []

        # Change this, if applicable, to the maximum number of manifolds in the system under study
        self.maximum_manifold = np.inf

        # Change this to a negative number, possibly -np.inf, if the initial state can be de-excited
        self.minimum_manifold = 0

        # Used for automatically generating diagram instructions
        self.wavevector_dict = {'-':('Bu','Kd'),'+':('Ku','Bd')}

        # Used to find resonant-only contributions
        self.instruction_to_manifold_transition = {'Bu':np.array([0,1]),
                                                   'Bd':np.array([0,-1]),
                                                   'Ku':np.array([1,0]),
                                                   'Kd':np.array([-1,0])}

        self.detection_type = detection_type
        
        if detection_type == 'polarization':
            self.filter_instructions = self.polarization_detection_filter_instructions
        elif detection_type == 'integrated_polarization':
            self.filter_instructions = self.polarization_detection_filter_instructions
        elif detection_type == 'fluorescence':
            self.filter_instructions = self.fluorescence_detection_filter_instructions

    def interaction_tuple_to_str(self,tup):
        """Converts a tuple, tup = (nr,nc) into a string of +'s and -'s
"""
        s = '+'*tup[0] + '-'*tup[1]
        return s

    def set_phase_discrimination(self,interaction_list):
        if type(interaction_list[0]) is str:
            new_list = interaction_list
        elif type(interaction_list[0]) is tuple:
            new_list = [self.interaction_tuple_to_str(el) for el in interaction_list]

        self.efield_wavevectors = new_list
            

    def polarization_detection_filter_instructions(self,instructions):
        rho_manifold = np.array([0,0])
        for ins in instructions:
            rho_manifold += self.instruction_to_manifold_transition[ins]
            if rho_manifold[0] < self.minimum_manifold or rho_manifold[1] < self.minimum_manifold:
                return False
            if rho_manifold[0] > self.maximum_manifold or rho_manifold[1] > self.maximum_manifold:
                return False
        if rho_manifold[0] - rho_manifold[1] == 1:
            return True
        else:
            return False

    def fluorescence_detection_filter_instructions(self,instructions):
        rho_manifold = np.array([0,0])
        for ins in instructions:
            rho_manifold += self.instruction_to_manifold_transition[ins]
            if rho_manifold[0] < self.minimum_manifold or rho_manifold[1] < self.minimum_manifold:
                return False
            if rho_manifold[0] > self.maximum_manifold or rho_manifold[1] > self.maximum_manifold:
                return False
        if abs(rho_manifold[1]-rho_manifold[1]) == 0 and rho_manifold[1] != 0:
            return True
        else:
            return False

    def set_f_list(self):
        f_list = []
        for i in range(len(self.efield_wavectors)):
            f_list.append(self.wavevector_dict[i])
        return f_list

    def instructions_from_permutation(self,perm):
        f_list = []
        efield_order = []
        for i,k in perm:
            f_list.append(self.wavevector_dict[k])
            efield_order.append(i)
        all_instructions = itertools.product(*f_list)
        filtered_instructions = []
        for ins in all_instructions:
            if self.filter_instructions(ins):
                filtered_instructions.append(tuple(zip(ins,efield_order)))
        return filtered_instructions

    def wavefunction_instructions_from_permutation(self,perm):
        rho_instructions = self.instructions_from_permutation(perm)
        psi_instructions = []
        for instructions in rho_instructions:
            new_instructions = self.convert_rho_instructions_to_psi_instructions(instructions)
            if new_instructions in psi_instructions:
                pass
            else:
                psi_instructions.append(new_instructions)

        return psi_instructions

    def convert_rho_instructions_to_psi_instructions(self,instructions):
        psi_instructions = {'ket':[],'bra':[]}
        for key, pulse_num in instructions:
            if key[0] == 'K':
                psi_instructions['ket'].append((key[1],pulse_num))
            elif key[0] == 'B':
                psi_instructions['bra'].append((key[1],pulse_num))
        return psi_instructions

    def convert_rho_instructions_list_to_psi_instructions_list(self,instructions_list):
        psi_instructions_list = []
        for instructions in instructions_list:
            psi_instructions = self.convert_rho_instructions_to_psi_instructions(instructions)
            if psi_instructions not in psi_instructions_list:
                psi_instructions_list.append(psi_instructions)
        return psi_instructions_list

    def relevant_permutations(self,pulse_time_meshes):
        self.set_ordered_interactions()
        all_permutations = set(itertools.permutations(self.ordered_interactions))
        filtered_permutations = []
            
        for perm in all_permutations:
            remove = False
            for i in range(len(perm)-1):
                indi = perm[i][0]
                ti = pulse_time_meshes[indi]
                for j in range(i+1,len(perm)):
                    indj = perm[j][0]
                    tj = pulse_time_meshes[indj]
                    if ti[0] > tj[-1]:
                        remove = True
                        break
                if remove == True:
                    break
            if not remove:
                filtered_permutations.append(perm)
        return filtered_permutations

    def set_ordered_interactions(self):
        """Sets up a list of the time-ordered interactions, along with associated wavevector ('+' or '-')
"""
        num_pulses = len(self.efield_wavevectors)
        self.ordered_interactions = []
        for i in range(num_pulses):
            ks = self.efield_wavevectors[i]
            for k in ks:
                self.ordered_interactions.append((i,k))

    def get_diagrams(self,arrival_times):
        
        if self.detection_type == 'polarization':
            times = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times)-1)]
        elif self.detection_type == 'integrated_polarization':
            times = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times)-1)]
        elif self.detection_type == 'fluorescence':
            times = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times))]
        
        efield_permutations = self.relevant_permutations(times)
        all_instructions = []
        for perm in efield_permutations:
            all_instructions += self.instructions_from_permutation(perm)
        return all_instructions

    def get_wavefunction_diagrams(self,arrival_times):
        rho_instructions_list = self.get_diagrams(arrival_times)
        wavefunction_instructions = self.convert_rho_instructions_list_to_psi_instructions_list(rho_instructions_list)
        return wavefunction_instructions
        
