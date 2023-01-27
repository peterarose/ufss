#Standard python libraries
import os
import warnings
import copy
import time
import itertools
import functools

#Dependencies - numpy, scipy, matplotlib, pyfftw
import numpy as np
import pyx

class DiagramDrawer:
    """This class is used to draw double-sided Feynman diagrams and save them as pdf files
"""
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

    def display_diagram(self,diagram,*,exclude="image/png"):
        """Displays a diagram in a Jupyter notebook environment or similar
        Args:
            diagram: Feynman diagram to be drawn, must be a list or tuple of tuples
            exclude: MIME types to exclude when attempting to display diagram 
"""
        self.c = pyx.canvas.canvas()
        interaction_counter = 0
        for KB,n in diagram:
            self.draw_functions[KB](n,interaction_counter)
            interaction_counter += 1
        display(self.c,exclude=exclude)

    def display_diagrams(self,diagrams,*,exclude="image/png"):
        for diagram in diagrams:
            self.display_diagram(diagram,exclude=exclude)

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
        detection_type (str): default is 'polarization', other options are 'integrated_polarization' or 'fluorescence'

"""
    def __init__(self,*,detection_type = 'polarization'):
        DiagramDrawer.__init__(self)

        # Code will not actually function until the following three empty
        # lists are set by the user
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
        
        if detection_type == 'polarization' or detection_type == 'integrated_polarization':
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
        self.set_pdc()

        # If pulses and/or phase discrimination are changed, these two attributes 
        # must be reset or removed:
        try:
            del self.pulse_sequence
            del self.pulse_overlap_array
        except AttributeError:
            pass

    def set_pdc(self):
        num_pulses = len(self.efield_wavevectors)
        pdc = np.zeros((num_pulses,2),dtype='int')
        for i in range(num_pulses):
            for j in range(len(self.efield_wavevectors[i])):
                if self.efield_wavevectors[i][j] == '+':
                    pdc[i,0] += 1
                elif self.efield_wavevectors[i][j] == '-':
                    pdc[i,1] += 1
                else:
                    raise Exception('Could not set phase-discrimination condition')
        self.pdc = pdc
        self.pdc_tuple = tuple(tuple(pdc[i,:]) for i in range(pdc.shape[0]))

    def save_diagrams_as_text(self,diagrams,*,file_name = 'auto'):
        if file_name == 'auto':
            file_name = ''
            for nr,nc in self.pdc_tuple:
                file_name += '{1}_{2}_'.format(nr,nc)
            file_name = file_name[:-1]
            file_name += '.txt'
        with open(file_name,'w') as txtfile:
            txtfile.writelines('['+str(diagrams[0]))
            for diagram in diagrams[1:]:
                txtfile.writelines(',\n '+str(diagram))
            txtfile.writelines(']')

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

    def check_new_diagram_conditions(self,arrival_times):
        new = np.array(arrival_times)
        new_pulse_sequence = np.argsort(new)
        new_pulse_overlap_array = np.ones((len(arrival_times),
                                            len(arrival_times)),
                                           dtype='bool')

        intervals = self.arrival_times_to_pulse_intervals(arrival_times)
        
        for i in range(len(intervals)):
            ti = intervals[i]
            for j in range(i+1,len(intervals)):
                tj = intervals[j]
                if ti[0] > tj[-1]:
                    new_pulse_overlap_array[i,j] = False
                elif ti[-1] < tj[0]:
                    new_pulse_overlap_array[i,j] = False
        try:
            logic_statement = (np.allclose(new_pulse_overlap_array,self.pulse_overlap_array)
                and np.allclose(new_pulse_sequence,self.pulse_sequence))
            if logic_statement:
                calculate_new_diagrams = False
            else:
                calculate_new_diagrams = True
        except AttributeError:
            calculate_new_diagrams = True
        
        self.pulse_sequence = new_pulse_sequence
        self.pulse_overlap_array = new_pulse_overlap_array

        return calculate_new_diagrams

    def arrival_times_to_pulse_intervals(self,arrival_times):
        if self.detection_type == 'polarization' or self.detection_type == 'integrated_polarization':
            if len(arrival_times) == len(self.efield_wavevectors) + 1:
                # If the arrival time of the local oscillator was included in the list arrival_times,
                # remove it, it is not relevant to diagram generation
                arrival_times = arrival_times[:-1]
        intervals = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times))]

        return intervals

    def set_diagrams(self,arrival_times):
        intervals = self.arrival_times_to_pulse_intervals(arrival_times)
        
        efield_permutations = self.relevant_permutations(intervals)
        all_instructions = []
        for perm in efield_permutations:
            all_instructions += self.instructions_from_permutation(perm)
        self.current_diagrams = all_instructions
        

    def get_diagrams(self,arrival_times):
        calculate_new_diagrams = self.check_new_diagram_conditions(arrival_times)
        if calculate_new_diagrams:
            self.set_diagrams(arrival_times)
        return self.current_diagrams

    def get_wavefunction_diagrams(self,arrival_times):
        rho_instructions_list = self.get_diagrams(arrival_times)
        wavefunction_instructions = self.convert_rho_instructions_list_to_psi_instructions_list(rho_instructions_list)
        return wavefunction_instructions

    def get_diagram_final_state(self,diagram):
        """Returns ket and bra manifold indices after all diagram interactions
        Args:
            diagram (list) : list of ordered interactions
        Returns: list with the first entry the ket manifold index, and 
            the second entry the bra manifold index
"""
        rho_manifold = np.array([0,0],dtype='int')
        for ins,pulse_num in diagram:
            rho_manifold += self.instruction_to_manifold_transition[ins]
        return list(rho_manifold)

    def get_diagram_excitation_manifold(self,diagram,*,number_of_interactions=2):
        rho_manifold = np.array([0,0])
        for ins,pulse_num in diagram[:number_of_interactions]:
            rho_manifold += self.instruction_to_manifold_transition[ins]
        if rho_manifold[0] == rho_manifold[1]:
            return rho_manifold[0]
        else:
            return None

    def filter_diagrams_by_final_state(self,diagrams,state):
        """Returns all diagrams that end in the specified state
        Args:
            diagrams (list) : list of diagrams to filter
            state (list) : list with the first entry the ket manifold index, 
                and the second entry the bra manifold index
"""
        new_diagrams = []
        for diagram in diagrams:
            diagram_state = self.get_diagram_final_state(diagram)
            if diagram_state == state:
                new_diagrams.append(diagram)
        return new_diagrams

    def filter_diagrams_by_excitation_manifold(self,diagrams,*,manifold=1,number_of_interactions=2):
        new_diagrams = []
        for diagram in diagrams:
            man = self.get_diagram_excitation_manifold(diagram,number_of_interactions=number_of_interactions)
            if man == manifold:
                new_diagrams.append(diagram)
        return new_diagrams

    def get_diagram_sign(self,diagram):
        if len(diagram) % 2:
            sign = 1j
        else:
            sign = 1
        for ins,pulse_num in diagram:
            if 'K' in ins:
                sign *= -1j
            elif 'B' in ins:
                sign *= 1j
            else:
                raise Exception('diagram is not in proper format')
            
        return sign

    def filter_diagrams_by_sign(self,diagrams,*,sign=1):
        new_diagrams = []
        for diagram in diagrams:
            diagram_sign = self.get_diagram_sign(diagram)
            if diagram_sign == sign:
                new_diagrams.append(diagram)
        return new_diagrams

    def remove_all_permutations(self,diagrams):
        new_diagrams = []
        diagram_weights = []
        set_list = []
        for diagram in diagrams:
            new_set = set(diagram)
            try:
                ind = set_list.index(new_set)
                diagram_weights[ind] += 1
            except ValueError:
                new_diagrams.append(diagram)
                diagram_weights.append(1)
                set_list.append(new_set)

        return diagram_weights, new_diagrams
