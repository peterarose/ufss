import numpy as np
import ufss
import os
from ufss.signals import SpectroscopyBase

class TransientAbsorption(SpectroscopyBase):
    def __init__(self,file_name,*,engine_name = 'UF2',include_linear=False,
                 conserve_memory=False,detection_type='polarization'):
        super().__init__(file_name,engine_name=engine_name,
                         conserve_memory=conserve_memory,
                         detection_type=detection_type)

        
        if include_linear:
            self.pdc = ((0,0),(1,0))
        else:
            self.pdc = ((1,1),(1,0))
        self.save_name = 'transient_absorption_signals'
        
    def set_hightest_order(self,highest_order):
        self.engine.include_higher_orders((highest_order-1,1))

    def get_signal(self,lam,*,max_order=np.inf):
        """lam is the perturbative parameter
        Args:
            lam (float) : unit-less electric field amplitude scaling factor
"""
        lam_list = [lam,1]
        signal = super().get_signal(lam_list,max_order=max_order)

        return signal

    def set_pulse_delays(self,pulse_delays):
        super().set_pulse_delays(pulse_delays)
        self.T = pulse_delays[0]
    
    def plot_signal(self,lam):
        signal = self.get_signal(lam)
        wt = self.engine.w
        T = self.engine.all_pulse_delays[0]
        ufss.signals.plot2D(T,wt,signal,part='real')

    def plot_order(self,order,T_index):
        signal = self.get_signal_order(order)
        wt = self.engine.w
        T = self.engine.all_pulse_delays[0]
        ufss.signals.plot2D(T,wt,signal,part='real')
