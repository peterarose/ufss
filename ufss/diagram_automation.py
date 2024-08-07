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

        self.include_state_labels = True
        self.include_emission_arrow = True
        self.include_pulse_labels = True
        self.sawtooth = None
        self.brightness = 0
        self.saturation = 1
        self.hues = []
        self.vertical_scale = 1
        self.diagram_size = 'tall'
        self.rails_hsb = (1,1,0)

    def set_pulse_colors(self,hues='auto',brightness='auto',saturation='auto'):
        """Set the color used to draw each pulse arrow and text. If called with
            no arguments, it will automatically asign colors to each pulse. If
            this method is never called, all pulses will be drawn in black.

        Args:
            hues (list) : list of hue values for each pulse
            brightness (list) : list of brightness values for each pulse
            saturation (list) : list of saturation values for each pulse
            """
        n = self.pdc.shape[0]
        if hues == 'auto':
            hues = np.arange(n)/n + 0.1
            self.hues = np.mod(hues,1)
        else:
            self.hues = hues
        if brightness == 'auto':
            self.brightness = [1]*n
        else:
            self.brightness = brightness
        if saturation == 'auto':
            self.saturation = [1]*n
        else:
            self.saturation = saturation
    
    def color_function(self,pulse_num):
        """Get the color of each pulse as a pyx color object (default is black)

        Args:
            pulse_num (int) : pulse number
        Returns:
            pyx.color.hsb : pyx color object for the given pulse
        """
        if self.hues == []:
            return pyx.color.hsb(1,self.saturation,self.brightness)
        else:
            return pyx.color.hsb(self.hues[pulse_num],self.saturation[pulse_num],
                                 self.brightness[pulse_num])

    def save_diagram(self,diagram,*,folder_name = ''):
        """Saves a pdf of an individual Feynman diagram

        Args:
            diagram (tuple) : Feynman diagram
            folder_name (str, optional) : relative or absolute path to folder 
                where diagram will be saved
        """
        os.makedirs(folder_name,exist_ok=True)
        filename = self.draw_diagram(diagram)
        self.c.writePDFfile(os.path.join(folder_name,filename))

    def save_diagrams(self,diagrams,*,folder_name=''):
        """Saves pdf files for each Feynman diagram supplied

        Args:
            diagrams (list) : list of Feynman diagrams
            folder_name (str,optional) : relative or absolute path to folder
                where diagram will be saved
        """
        for diagram in diagrams:
            self.save_diagram(diagram,folder_name=folder_name)

    def draw_diagram(self,diagram):
        """Draws Feynman diagram for viewing in a Jupyter notebook

        Args:
            diagram (tuple) : Feynman diagram
        Returns:
            str : unique filename for this diagram (used only for saving)
        """
        if self.diagram_size == 'medium':
            filename = self.draw_medium_diagram(diagram)
        elif self.diagram_size == 'compact':
            filename = self.draw_compact_diagram(diagram)
        elif self.diagram_size == 'tall':
            filename = self.draw_tall_diagram(diagram)
        else:
            raise Exception('Attribute diagram_size has an invalid string')

        return filename
        
    def draw_tall_diagram(self,diagram):
        """Draws Feynman diagram for viewing in a Jupyter notebook. Tall 
            indicates that the diagrams will be spread out vertically. This
            makes diagrams take up a lot of space but ensures that arrows
            never overlap.

        Args:
            diagram (tuple) : Feynman diagram
        Returns:
            str : unique filename for this diagram (used only for saving)
        """
        self.vertical_scale = 1
        self.c = pyx.canvas.canvas()
        interaction_counter = 0
        self.diagram_drawer_manifolds = np.array([0,0],dtype='int')
        self.y0 = 0

        filename = ''
        ymin1 = 0
        if self.include_state_labels:
            ymin1 = -0.4
            self.c.stroke(pyx.path.line(0,ymin1,0,0),[pyx.color.hsb(*self.rails_hsb)])
            self.c.stroke(pyx.path.line(1,ymin1,1,0),[pyx.color.hsb(*self.rails_hsb)])
            k,b = list(self.diagram_drawer_manifolds)
            rho_str = r'$|{}\rangle\langle{}|$'.format(k,b)
            self.c.text(0.1,-0.25,rho_str)
            
        if self.sawtooth == 'bottom':
            ymin2 = ymin1 - 0.3
            self.c.stroke(pyx.path.line(0,ymin2,0,ymin1),[pyx.color.hsb(*self.rails_hsb)])
            self.c.stroke(pyx.path.line(1,ymin2,1,ymin1),[pyx.color.hsb(*self.rails_hsb)])
            self.draw_sawtooth(ymin2)
        
        for KB,n in diagram:
            self.draw_functions[KB](n,interaction_counter)
            filename += KB + str(n)
            interaction_counter += 1
        if interaction_counter % 2:
            if self.include_emission_arrow:
                self.draw_emission()
        if self.sawtooth == 'top':
            y = self.y0
            y_extra = y + 0.45
            self.c.stroke(pyx.path.line(0, y, 0, y_extra),[pyx.color.hsb(*self.rails_hsb)])
            self.c.stroke(pyx.path.line(1, y, 1, y_extra),[pyx.color.hsb(*self.rails_hsb)])
            self.draw_sawtooth(y_extra)
        return filename

    def draw_medium_diagram(self,diagram):
        """Draws Feynman diagram for viewing in a Jupyter notebook. Medium 
            indicates that the diagrams will be less spread out vertically, as
            commpared with "tall". Arrows may overlap to some extent.

        Args:
            diagram (tuple) : Feynman diagram
        Returns:
            str : unique filename for this diagram (used only for saving)
        """
        self.vertical_scale = 0.6
        self.c = pyx.canvas.canvas()
        interaction_counter = 0
        self.diagram_drawer_manifolds = np.array([0,0],dtype='int')
        self.y0 = 0

        filename = ''
        ymin1 = 0
        if self.include_state_labels:
            k,b = list(self.diagram_drawer_manifolds)
            rho_str = r'$|{}\rangle\langle{}|$'.format(k,b)
            self.c.text(0.1,0.15,rho_str)
            
        if self.sawtooth == 'bottom':
            ymin2 = ymin1 - 0.3
            self.c.stroke(pyx.path.line(0, ymin2, 0, ymin1),[pyx.color.hsb(*self.rails_hsb)])
            self.c.stroke(pyx.path.line(1, ymin2, 1, ymin1),[pyx.color.hsb(*self.rails_hsb)])
            self.draw_sawtooth(ymin2)
        
        for KB,n in diagram:
            self.draw_functions[KB](n,interaction_counter)
            filename += KB + str(n)
            interaction_counter += 1
        if interaction_counter % 2:
            if self.include_emission_arrow:
                yf = self.y0 + 1.1
                self.c.stroke(pyx.path.line(0, self.y0, 0, yf),[pyx.color.hsb(*self.rails_hsb)])
                self.c.stroke(pyx.path.line(1, self.y0, 1, yf),[pyx.color.hsb(*self.rails_hsb)])
                self.draw_emission()
                self.y0 = yf
        if self.sawtooth == 'top':
            y = self.y0
            y_extra = y + 0.45
            self.c.stroke(pyx.path.line(0, y, 0, y_extra),[pyx.color.hsb(*self.rails_hsb)])
            self.c.stroke(pyx.path.line(1, y, 1, y_extra),[pyx.color.hsb(*self.rails_hsb)])
            self.draw_sawtooth(y_extra)
        return filename

    def draw_compact_diagram(self,diagram):
        """Draws Feynman diagram for viewing in a Jupyter notebook. Compact 
            indicates that the diagrams will be very compact vertically. 
            Arrows may overlap a great deal. Typically should only be used
            with attribute "include_pulse_labels" set to False.

        Args:
            diagram (tuple) : Feynman diagram
        Returns:
            str : unique filename for this diagram (used only for saving)
        """
        self.vertical_scale = 0.3
        self.c = pyx.canvas.canvas()
        interaction_counter = 0
        self.diagram_drawer_manifolds = np.array([0,0],dtype='int')
        self.y0 = 0

        filename = ''
        ymin1 = 0
        if self.include_state_labels:
            k,b = list(self.diagram_drawer_manifolds)
            rho_str = r'$|{}\rangle\langle{}|$'.format(k,b)
            self.c.text(0.1,0.15,rho_str)
            
        if self.sawtooth == 'bottom':
            ymin2 = ymin1 - 0.3
            self.c.stroke(pyx.path.line(0, ymin2, 0, ymin1),[pyx.color.hsb(*self.rails_hsb)])
            self.c.stroke(pyx.path.line(1, ymin2, 1, ymin1),[pyx.color.hsb(*self.rails_hsb)])
            self.draw_sawtooth(ymin2)

        old_labels_flag = self.include_state_labels
        self.include_state_labels = False
        for KB,n in diagram:
            self.draw_functions[KB](n,interaction_counter)
            filename += KB + str(n)
            interaction_counter += 1
        
        self.include_state_labels = old_labels_flag
        if self.include_state_labels:
            yf = self.y0 + 0.75
            self.c.stroke(pyx.path.line(0, self.y0, 0, yf),[pyx.color.hsb(*self.rails_hsb)])
            self.c.stroke(pyx.path.line(1, self.y0, 1, yf),[pyx.color.hsb(*self.rails_hsb)])
            k,b = list(self.diagram_drawer_manifolds)
            rho_str = r'$|{}\rangle\langle{}|$'.format(k,b)
            self.c.text(0.1,self.y0 + 0.45,rho_str)
            self.y0 = yf

            if self.sawtooth == 'top':
                y = self.y0
                y_extra = y + 0.05
                self.c.stroke(pyx.path.line(0, y, 0, y_extra),[pyx.color.hsb(*self.rails_hsb)])
                self.c.stroke(pyx.path.line(1, y, 1, y_extra),[pyx.color.hsb(*self.rails_hsb)])
                self.draw_sawtooth(y_extra)
        else:
            if self.sawtooth == 'top':
                y = self.y0
                y_extra = y + 0.75
                self.c.stroke(pyx.path.line(0, y, 0, y_extra),[pyx.color.hsb(*self.rails_hsb)])
                self.c.stroke(pyx.path.line(1, y, 1, y_extra),[pyx.color.hsb(*self.rails_hsb)])
                self.draw_sawtooth(y_extra)
            
        if interaction_counter % 2:
            if self.include_emission_arrow:
                yf = self.y0 + 0.75
                self.c.stroke(pyx.path.line(0, self.y0, 0, yf),[pyx.color.hsb(*self.rails_hsb)])
                self.c.stroke(pyx.path.line(1, self.y0, 1, yf),[pyx.color.hsb(*self.rails_hsb)])
                self.draw_emission()

        return filename
        
    def display_diagram(self,diagram,*,exclude="image/png"):
        """Displays a diagram in a Jupyter notebook environment or similar

        Args:
            diagram (tuple) : Feynman diagram to be drawn, must be a list or 
                tuple of tuples
            exclude (str,optional) : MIME types to exclude when attempting to 
                display diagram
        """
        filename = self.draw_diagram(diagram)
        display(self.c,exclude=exclude)

    def display_diagrams(self,diagrams,*,exclude="image/png"):
        """Displays diagrams in a Jupyter notebook environment or similar

        Args:
            diagrams (list) : list of Feynman diagram to be drawn, each of 
                which must be a list or tuple of tuples
            exclude (str,optional) : MIME types to exclude when attempting to 
                display diagram
        """
        for diagram in diagrams:
            self.display_diagram(diagram,exclude=exclude)

    def draw_emission(self):
        """Draws the emission arrow at the top of a diagram"""
        change = np.array([-1,0])
        self.diagram_drawer_manifolds += change
        yf = self.y0 + self.vertical_scale
        self.double_sided(yf)
        x,y = (0,self.y0+0.5)
        x2,y2 = (x-.5,y+.5)
        self.c.stroke(pyx.path.line(x,y,x2,y2),[pyx.style.linestyle.dashed,pyx.deco.earrow(size=0.3)])

    def draw_sawtooth(self,y0):
        """Draws a sawtooth line indicating that the diagram has been truncated
        """
        num_teeth = 3
        x_len = 1/num_teeth/2
        y_len = x_len * np.tan(np.pi/3)
        x0 = 0
        y1 = y0 + y_len
        for i in range(num_teeth):
            x1 = x0 + x_len
            self.c.stroke(pyx.path.line(x0,y0,x1,y1),[pyx.color.hsb(*self.rails_hsb)])
            self.c.stroke(pyx.path.line(x1,y1,x1+x_len,y0),[pyx.color.hsb(*self.rails_hsb)])
            x0 = x1 + x_len
        return None

    def double_sided(self,yf):
        """Draws the double-sided rails for the diagram

        Args:
            yf (float) : height of Feynman diagram
        """
        self.c.stroke(pyx.path.line(0, self.y0, 0, yf),[pyx.color.hsb(*self.rails_hsb)])
        self.c.stroke(pyx.path.line(1, self.y0, 1, yf),[pyx.color.hsb(*self.rails_hsb)])
        
        if self.include_state_labels:
            k,b = list(self.diagram_drawer_manifolds)
            rho_str = r'$|{}\rangle\langle{}|$'.format(k,b)
            self.c.text(0.1,self.y0 + 0.75, rho_str)

    def K_circle(self,interaction_num,pulse_num):
        """Draws a circle on the ket-side rail of the diagram

        Args:
            interaction_num (int) : interaction number
            pulse_num (int) : pulse number
        """
        self.c.fill(pyx.path.circle(0,interaction_num+0.5,0.1),[self.color_function(pulse_num)])

    def B_circle(self,interaction_num,pulse_num):
        """Draws a circle on the bra-side rail of the diagram

        Args:
            interaction_num (int) : interaction number
            pulse_num (int) : pulse number
        """
        self.c.fill(pyx.path.circle(1,interaction_num+0.5,0.1),[self.color_function(pulse_num)])

    def right_arrow(self,x,y,interaction_num,pulse_num):
        """Draws an arrow pointing up and to the right, depicting an 
            interaction with the rotating term

        Args:
            interaction_num (int) : interaction number
            pulse_num (int) : pulse number
        """
        xf, yf = (x+0.5,y+0.5)
        self.c.stroke(pyx.path.line(x,y,xf,yf),[self.color_function(pulse_num),pyx.deco.earrow(size=0.3)])
        return xf,yf

    def left_arrow(self,x,y,interaction_num,pulse_num):
        """Draws an arrow pointing up and to the left, depicting an 
            interaction with the counter-rotating term

        Args:
            interaction_num (int) : interaction number
            pulse_num (int) : pulse number
        """
        xf,yf = (x-.5,y+.5)
        self.c.stroke(pyx.path.line(x,y,xf,yf),[self.color_function(pulse_num),pyx.deco.earrow(size=0.3)])
        return xf,yf

    def pulse_text(self,x,y,pulse_num):
        """Rights the pulse label next to the associated interaction arrow

        Args:
            x (float) : x-position on the diagram where the text will go
            y (float) : y-position on the diagram where the text will go
            pulse_num (int) : pulse number
        """
        if self.include_pulse_labels:
            text = self.pulse_labels[pulse_num]
            self.c.text(x,y,text)

    def draw_Bd(self,pulse_num,interaction_num):
        """Draws the bra-down (Bd) interaction, which is the rotating term
            interacting with the bra-side

        Args:
            pulse_num (int) : pulse number
            interaction_num (int) : interaction number
        """
        change = np.array([0,-1])
        self.diagram_drawer_manifolds += change
        m = interaction_num
        n = pulse_num
        yf = self.y0 + self.vertical_scale
        self.double_sided(yf)
        self.B_circle(self.y0,n)
        x,y = (1,self.y0+0.5)
        x2,y2 = self.right_arrow(x,y,self.y0,n)
        self.pulse_text(x2-0.1,y2-0.5,n)
        self.y0 = yf

    def draw_Bu(self,pulse_num,interaction_num):
        """Draws the bra-up (Bu) interaction, which is the counter-rotating
            term interacting with the bra-side

        Args:
            pulse_num (int) : pulse number
            interaction_num (int) : interaction number
        """
        change = np.array([0,1])
        self.diagram_drawer_manifolds += change
        m = interaction_num
        n = pulse_num
        yf = self.y0 + self.vertical_scale
        self.double_sided(yf)
        self.B_circle(self.y0,n)
        x,y = (1.5,self.y0)
        self.left_arrow(x,y,self.y0,n)
        self.pulse_text(x-0.1,y+0.1,n)
        self.y0 = yf

    def draw_Kd(self,pulse_num,interaction_num):
        """Draws the ket-down (Kd) interaction, which is the counter-rotating
            term interacting with the ket-side

        Args:
            pulse_num (int) : pulse number
            interaction_num (int) : interaction number
        """
        change = np.array([-1,0])
        self.diagram_drawer_manifolds += change
        m = interaction_num
        n = pulse_num
        yf = self.y0 + self.vertical_scale
        self.double_sided(yf)
        self.K_circle(self.y0,n)
        x,y = (0,self.y0+0.5)
        x2,y2 = self.left_arrow(x,y,self.y0,n)
        self.pulse_text(x2-0.1,y2-0.5,n)
        self.y0 = yf

    def draw_Ku(self,pulse_num,interaction_num):
        """Draws the ket-up (Ku) interaction, which is the rotating term 
            interacting with the ket-side

        Args:
            pulse_num (int) : pulse number
            interaction_num (int) : interaction number
        """
        change = np.array([1,0])
        self.diagram_drawer_manifolds += change
        m = interaction_num
        n = pulse_num
        yf = self.y0 + self.vertical_scale
        self.double_sided(yf)
        self.K_circle(self.y0,n)
        x,y = (-0.5,self.y0)
        x2,y2 = self.right_arrow(x,y,self.y0,n)
        self.pulse_text(x-0.1,y+0.1,n)
        self.y0 = yf

    

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
        self.wavevector_dict = {'-':('Kd','Bu'),'+':('Ku','Bd')}

        # Used to find resonant-only contributions
        self.instruction_to_manifold_transition = {'Bu':np.array([0,1]),
                                                   'Bd':np.array([0,-1]),
                                                   'Ku':np.array([1,0]),
                                                   'Kd':np.array([-1,0])}
        self.detection_type = detection_type
        
        if 'polarization' in detection_type:
            self.filter_instructions = self.polarization_detection_filter_instructions
        elif detection_type == 'fluorescence':
            self.filter_instructions = self.fluorescence_detection_filter_instructions
        else:
            raise Exception('Detection type not supported')

    def interaction_tup_to_str(self,tup):
        """Converts a tuple, tup = (nr,nc) into a string of +'s and -'s, used
            for converting between different internal methods of representing
            the number of interactions with (counter-)rotating terms for each
            pulse

        Args:
            tup (tuple) : format (nr,nc) where nr (nc) is the number of 
                interactions with the rotating (counter-rotating) term
        Returns:
            str : repeats "+" nr times and "-" nc times
        """
        s = '+'*tup[0] + '-'*tup[1]
        return s

    def set_phase_discrimination(self,interaction_list):
        """Sets the efield wavevectors and pdc (phase discrimination 
            condition), given a list of tuples describing the number of
            interactions with the (counter-)rotating terms of each pulse
        """
        if type(interaction_list[0]) is str:
            new_list = interaction_list
        elif type(interaction_list[0]) is tuple:
            new_list = [self.interaction_tup_to_str(el)
                        for el in interaction_list]

        self.efield_wavevectors = new_list
        self.set_pdc()

        # If pulses and/or phase discrimination are changed, these two
        # attributes must be reset or removed:
        
        try:
            del self.pulse_sequence
            del self.pulse_overlap_array
        except AttributeError:
            pass

    def pdc_arr_to_tup(self,pdc_arr):
        """Converts a pdc represented as a numpy array to a tuple of tuples

        Args:
            pdc_arr (np.ndarray) : pdc as a numpy array
        Returns:
            tuple : pdc as a tuple of tuples
        """
        return tuple(tuple(pdc_arr[i,:]) for i in range(pdc_arr.shape[0]))

    def pdc_tup_to_arr(self,pdc_tup):
        """Converts a pdc represented as a tuple of tuples to a numpy array

        Args:
            pdc_arr (np.ndarray) : pdc as a tuple of tuples
        Returns:
            tuple : pdc as a numpy array
        """
        return np.array(pdc_tup)
    
    def set_pdc(self):
        """Sets pdc using the attribute "efield_wavevectors"
        """
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

        self.pulse_orders = pdc.sum(axis=1).tolist()
        self.highest_order = np.sum(pdc)
        self.pdc_tuple = self.pdc_arr_to_tup(pdc)
        self.pdc = pdc
        self.max_pdc = pdc.copy()
        self.set_signal_pdcs([pdc])

    def set_signal_pdcs(self,pdcs):
        """Takes a list of pdcs, converts each pdc to a tuple if necessary, 
            and sets them all to the attribute "signal_pdcs"

        Args:
            pdcs (list) : list of pdcs as np.ndarrays or tuples
        """
        if type(pdcs[0]) is tuple:
            self.signal_pdcs = pdcs
        else:
            self.signal_pdcs = [self.pdc_arr_to_tup(pdc) for pdc in pdcs]
            
        self.all_pdcs = set(self.signal_pdcs)

        return None

    def include_higher_orders(self,pulse_orders):
        """Adds additional higher-order pdcs based upon the baseline pdc 
            stored in attribute "pdc". Additional pdcs are of the same or 
            lower order in each pulse, as specified by pulse_orders
            
        Args:
            pulse_orders (list) : list of integers specifying what order in
                perturbation theory to go to for each pulse
        """
        self.pulse_orders = pulse_orders
        lowest_orders = self.pdc.sum(axis=1)
        highest_orders = np.zeros(lowest_orders.shape,dtype=
                                  lowest_orders.dtype)
        for i in range(len(self.pulse_orders)):
            trial = lowest_orders.copy()
            trial[i] = pulse_orders[i]
            highest_orders[i] = trial.sum()
        self.highest_order = np.max(highest_orders)
        
        self.max_pdc = self.pdc.copy()
        # additional interactions
        add_ints = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)
        for i in range(len(pulse_orders)):
            # additional number of interactions
            add_num_ints = pulse_orders[i] - np.sum(self.pdc[i,:])
            if add_num_ints%2 != 0:
                warnings.warn('Cannot go to order {:0} in pulse {:1}, can only go to order {:2} while satisfying the phase-discrimination condition specified'.format(pulse_orders[i],i,pulse_orders[i]-1))
            # additional interactions are equally divided between the rotating and counter-rotating terms
            add_ints[i,:] = add_num_ints//2
        self.max_pdc += add_ints

        self.set_all_signal_matching_pdcs()

        return None

    def set_all_signal_matching_pdcs(self):
        """Determines all possible higher order pdcs that both match the 
            lowest order pdc, and which are compatible with the specified 
            maximum pulse orders
        """
        signal_pdcs = [self.pdc]
        num_pulses = len(self.pulse_orders)
        for n in range(num_pulses):
            min_order = self.pdc[n,:].sum()
            remainder = self.pulse_orders[n] - min_order
            pdc_add = np.zeros(self.pdc.shape,dtype=self.pdc.dtype)
            pdc_add[n,:] = 1
            new_pdcs = []
            for i in range(remainder // 2):
                for pdc in signal_pdcs:
                    new_pdc = pdc + pdc_add * (i+1)
                    if new_pdc.sum() <= self.highest_order:
                        new_pdcs.append(new_pdc)

            signal_pdcs += new_pdcs

        self.set_signal_pdcs(signal_pdcs)

        return None

    def contract_pdc(self,pdc):
        """Takes a phase-discrimination condition and returns a tuple 
            specifying the wavevector of the resultant signal

        Args:
            pdc (np.ndarray) : input pdc
        Returns:
            tuple : wavevector of resultant signal"""
        sum_rule = np.ones(pdc.shape,dtype=pdc.dtype)
        sum_rule[:,1] = -1
        return tuple(np.sum(sum_rule*pdc,axis=1))

    def save_diagrams_as_text(self,diagrams,*,file_name = 'auto'):
        """Saves diagrams to a text file with each diagram on a separate line

        Args:
            diagrams (list) : list of diagrams
            file_name (str, optional) : filename to use for saving diagrams
        """
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
        """Filter function to test whether a given diagram ends in a coherence
            between two states that are only one excitation apart

        Args:
            instructions (tuple) : diagram
        Returns:
            bool : True if diagram obeys the polarization detection condition
        """
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
        """Filter function to test whether a given diagram ends in an 
            excited-state population

        Args:
            instructions (tuple) : diagram
        Returns:
            bool : True if diagram obeys the fluorescence detection condition
        """
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

    def RWA_filter_instructions(self,instructions):
        """Filter function to test whether a given diagram obeys the rotating-wave
        approximation (RWA), given that resonance excitations cannot de-excite the
        initial density matrix below attribute "minimum_manifold" and cannot excite
        the initial density matrix above attribute "maximum_manifold"

        Args:
            instructions (tuple) : diagram
        Returns:
            bool : True if diagram obeys the RWA
        """
        rho_manifold = np.array([0,0])
        for ins in instructions:
            rho_manifold += self.instruction_to_manifold_transition[ins]
            if rho_manifold[0] < self.minimum_manifold or rho_manifold[1] < self.minimum_manifold:
                return False
            if rho_manifold[0] > self.maximum_manifold or rho_manifold[1] > self.maximum_manifold:
                return False
        return True

    def instructions_from_permutation(self,perm):
        """Generates a list of diagrams given an input permutation
        
        Args:
            perm (tuple) : tuple of tuples giving a particular ordering of pulse
                interactions
        Returns:
            list : list of Feynman diagrams that obey the input interaction order
                and the detection condition stored in attribute "detection_type"
        """
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
        """Generates a list of closed-system diagrams given an input permutation
        
        Args:
            perm (tuple) : tuple of tuples giving a particular ordering of pulse
                interactions
        Returns:
            list : list of Feynman diagrams that obey the input interaction order
                and the detection condition stored in attribute "detection_type"
        """
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
        """Converts a diagram used for calculating signals using density matrices
            into a diagram that can be used for calculating signals using 
            wavefunctions (sometimes called a loop diagram)
            
        Args:
            instructions (tuple) : a diagram for density matrices
        Returns:
            dict : a diagram for wavefunctions
        """
        psi_instructions = {'ket':[],'bra':[]}
        for key, pulse_num in instructions:
            if key[0] == 'K':
                psi_instructions['ket'].append((key[1],pulse_num))
            elif key[0] == 'B':
                psi_instructions['bra'].append((key[1],pulse_num))
        return psi_instructions

    def convert_rho_instructions_list_to_psi_instructions_list(self,instructions_list):
        """Converts diagrams used for calculating signals using density matrices
            into diagrams that can be used for calculating signals using 
            wavefunctions (sometimes called a loop diagram)
            
        Args:
            instructions (list) : a list of diagrams for density matrices
        Returns:
            list : a list of diagrams for wavefunctions
        """
        psi_instructions_list = []
        for instructions in instructions_list:
            psi_instructions = self.convert_rho_instructions_to_psi_instructions(instructions)
            if psi_instructions not in psi_instructions_list:
                psi_instructions_list.append(psi_instructions)
        return psi_instructions_list

    def relevant_permutations(self,pulse_time_meshes):
        """Determines which permutations of pulse orderings will result in non-zero 
            calculations, given the pulse durations

        Args:
            pulse_time_meshes (list) : list of pulse durations
        Returns:
            list : all permutations of pulses that obey causality
        """
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
        """Sets up a list of the time-ordered interactions, along with associated 
            wavevector ('+' or '-') for each interaction.
        """
        num_pulses = len(self.efield_wavevectors)
        self.ordered_interactions = []
        for i in range(num_pulses):
            ks = self.efield_wavevectors[i]
            for k in ks:
                self.ordered_interactions.append((i,k))

    def check_new_diagram_conditions(self,arrival_times):
        """Check to see if pulse overlaps have changed. If so, a new list of 
            diagrams must be calculated. If not, old list may be reused.
        """
        new = np.array(arrival_times)
        new_pulse_sequence = np.argsort(new)
        new_pulse_overlap_array = np.zeros((len(arrival_times),
                                            len(arrival_times)),
                                           dtype='bool')

        intervals = self.arrival_times_to_pulse_intervals(arrival_times)
        
        for i in range(len(intervals)):
            ti = intervals[i]
            for j in range(i+1,len(intervals)):
                tj = intervals[j]
                if ti[0] >= tj[0] and ti[0] <= tj[-1]:
                    new_pulse_overlap_array[i,j] = True
                elif ti[-1] >= tj[0] and ti[-1] <= tj[-1]:
                    new_pulse_overlap_array[i,j] = True
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
        """Converts the input pulse durations, which are all centered on zero,
            into the intervals of time where each pulse is non-zero, which are
            called pulse intervals.
        
        Args:
            arrival_times (list) : list of pulse arrival times
        Returns:
            list : pulse intervals
        """
        if self.detection_type == 'polarization' or self.detection_type == 'integrated_polarization':
            if len(arrival_times) == len(self.efield_wavevectors) + 1:
                # If the arrival time of the local oscillator was included in the list arrival_times,
                # remove it, it is not relevant to diagram generation
                arrival_times = arrival_times[:-1]
        intervals = [self.efield_times[i] + arrival_times[i] for i in range(len(arrival_times))]

        return intervals

    def set_diagrams(self,arrival_times):
        """Given the input arrival times of each pulse, determine which diagrams
            obey causality, and thus yield a non-zero signal

        Args:
            arrival_times (list) : list of pulse arrival times
        """
        intervals = self.arrival_times_to_pulse_intervals(arrival_times)
        
        efield_permutations = self.relevant_permutations(intervals)
        all_instructions = []
        for perm in efield_permutations:
            all_instructions += self.instructions_from_permutation(perm)
        self.current_diagrams = all_instructions
        

    def get_diagrams(self,arrival_times,*,recalculate=True):
        """Given the input arrival times of each pulse, determine if any pulse
            overlap conditions have changed. If so, calculate a new list of
            relevant diagrams. If not, return previously calculated diagrams.

        Args:
            arrival_times (list) : list of pulse arrival times
        Keyword Args:
            recalculate (bool) : if True, recalculate diagrams
        """
        if recalculate:
            calculate_new_diagrams = True
        else:
            calculate_new_diagrams = self.check_new_diagram_conditions(arrival_times)
        if calculate_new_diagrams:
            self.set_diagrams(arrival_times)
        return self.current_diagrams

    def set_diagram_dict(self,arrival_times):
        """Sets a dictionary of diagrams to calculate, with each key corresponding
            to one pdc stored in attribute "signal_pdcs". Allows for efficient 
            calculation of many different pdcs

        Args:
            arrival_times (list) : list of pulse arrival times
        """
        self.current_diagram_dict = {}
        for pdc in self.signal_pdcs:
            dg = DiagramGenerator(detection_type=self.detection_type)
            dg.efield_times = self.efield_times
            dg.set_phase_discrimination(pdc)
            dg.minimum_manifold = self.minimum_manifold
            dg.maximum_manifold = self.maximum_manifold
            dg.set_diagrams(arrival_times)
            self.current_diagram_dict[pdc] = dg.current_diagrams
        return None

    def get_diagram_dict(self,arrival_times,*,recalculate=False):
        """Given the input arrival times of each pulse, determine if any pulse
            overlap conditions have changed. If so, calculate a new dictionary of
            relevant diagrams. Each key is a pdc from "signal_pdcs".

        Args:
            arrival_times (list) : list of pulse arrival times
        Keyword Args:
            recalculate (bool) : if True, recalculate diagrams
        """
        if recalculate:
            calculate_new_diagrams = True
        else:
            calculate_new_diagrams = self.check_new_diagram_conditions(arrival_times)
        if calculate_new_diagrams:
            self.set_diagram_dict(arrival_times)

        return self.current_diagram_dict

    def get_wavefunction_diagrams(self,arrival_times):
        """Given the input arrival times of each pulse, determine if any pulse
            overlap conditions have changed. If so, calculate a new list of
            relevant wavefunciton diagrams. If not, return previously calculated 
            diagrams.

        Args:
            arrival_times (list) : list of pulse arrival times
        """
        rho_instructions_list = self.get_diagrams(arrival_times)
        wavefunction_instructions = self.convert_rho_instructions_list_to_psi_instructions_list(rho_instructions_list)
        return wavefunction_instructions

    def get_wavefunction_diagram_dict(self,arrival_times):
        """Given the input arrival times of each pulse, determine if any pulse
            overlap conditions have changed. If so, calculate a new dictionary of
            relevant wavefunction diagrams. Each key is a pdc from "signal_pdcs".

        Args:
            arrival_times (list) : list of pulse arrival times
        """
        rho_dict = self.get_diagram_dict(arrival_times)
        psi_dict = {}
        for pdc in self.signal_pdcs:
            psi_dict[pdc] = self.convert_rho_instructions_list_to_psi_instructions_list(rho_dict[pdc])
        return psi_dict

    def get_diagram_final_state(self,diagram):
        """Returns ket and bra manifold indices after all diagram interactions

        Args:
            diagram (list) : list of ordered interactions
        Returns
            list : list with the first entry the ket manifold index, and 
            the second entry the bra manifold index
        """
        rho_manifold = np.array([0,0],dtype='int')
        for ins,pulse_num in diagram:
            rho_manifold += self.instruction_to_manifold_transition[ins]
        return list(rho_manifold)

    def get_diagram_excitation_manifold(self,diagram,*,number_of_interactions=2):
        """Returns the excitation number of the density matrix after the 
            specified number of interactions have occurred. Returns None if the
            density matrix is in a coherence after the given number of
            interactions

        Args:
            diagram (list) : single diagram
        Keyword Args:
            number_of_interactions (int) : check excitation manifold after this 
                number of interactions have occurred
        Returns
            int : excitation number of the density matrix, if it ends in a 
                population (otherwise returns NaN)
        """
        truncated_diagram = diagram[:number_of_interactions]
        rho_manifold = self.get_diagram_final_state(truncated_diagram)
        if rho_manifold[0] == rho_manifold[1]:
            return rho_manifold[0]
        else:
            return np.nan

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
        """Returns all diagrams that end in the specified excitation manifold or
            excitation number

        Args:
            diagrams (list) : list of diagrams to filter
        Keyword Args:
            manifold (int) : excitation manifold/number to filter for
            number_of_interactions (int) : check excitation manifold after this 
                number of interactions have occurred
        """
        new_diagrams = []
        for diagram in diagrams:
            man = self.get_diagram_excitation_manifold(diagram,number_of_interactions=number_of_interactions)
            if man == manifold:
                new_diagrams.append(diagram)
        return new_diagrams

    def get_diagram_sign(self,diagram):
        """Calculates the overall sign of the input diagram
        
        Args:
            diagram (tuple) : diagram
        """
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
        """Filter out only the diagrams that have the specified sign
        
        Args:
            diagrams (list) : list of diagram
        Keyword Args:
            sign (int) : sign of signal (either +1 or -1)
        """
        new_diagrams = []
        for diagram in diagrams:
            diagram_sign = self.get_diagram_sign(diagram)
            if diagram_sign == sign:
                new_diagrams.append(diagram)
        return new_diagrams

    def remove_all_permutations(self,diagrams):
        """Given a list of input diagrams, strip each diagram of duplicate 
            interaction types, and return a list of diagrams where each diagram
            contains only one of each type of interaction. DEPRECATED?
        """
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
