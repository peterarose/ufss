import numpy as np
import ufss
from ufss.signals import SpectroscopyBase
import matplotlib.pyplot as plt

class Spectroscopy2D(SpectroscopyBase):

    def set_pulse_delays(self,pulse_delays):
        super().set_pulse_delays(pulse_delays)
        self.tau = pulse_delays[0]
        self.T = pulse_delays[1]

    def tau_dft(self,signal,*,add_heaviside=True,zero_pad=True):
        sig = signal.copy()
        if add_heaviside:
            sig[0,...] *= 0.5
        tau = self.tau
        t = self.engine.t
        if zero_pad:
            sig_shape = list(sig.shape)
            new_tau_size = tau.size*2-1
            sig_shape[0] = new_tau_size
            new_sig = np.zeros(sig_shape,dtype='complex')
            new_sig[new_tau_size//2:,...] = sig
            new_tau = np.zeros(tau.size*2-1,dtype=tau.dtype)
            new_tau[new_tau_size//2:] = tau
            new_tau[:new_tau_size//2+1] = -tau[::-1]
            tau = new_tau
            sig = new_sig
        if self.rephasing:
            wtau,signal_ft = ufss.signals.SignalProcessing.ft1D(tau,sig,
                                                                axis=0)
            signal_ft = signal_ft/(2*np.pi)
        else:
            wtau,signal_ft = ufss.signals.SignalProcessing.ift1D(tau,sig,
                                                                 axis=0)
        self.wtau = wtau
        return wtau, signal_ft

    def get_signal(self,lam,*,tau_dft = True):
        """lam is the perturbative parameter
        Args:
            lam (float) : unit-less electric field amplitude scaling factor
"""
        lam_list = [lam,lam,1]
        signal = super().get_signal(lam_list)

        if tau_dft == True:
            wtau, signal = self.tau_dft(signal)

        return signal

    def get_signal_order(self,order,*,tau_dft = True):
        signal = super().get_signal_order(order)

        if tau_dft == True:
            wtau, signal = self.tau_dft(signal)

        return signal

    def plot_signal(self,lam,T_index):
        signal = self.get_signal(lam)[:,T_index,:]
        wt = self.engine.w
        ufss.signals.plot2D(self.wtau,wt,signal,part='real')
        plt.xlabel(r'$\omega_{\tau}$')
        plt.ylabel(r'$\omega_t - \omega_c$')

    def plot_order(self,order,T_index):
        signal = self.get_signal_order(order)[:,T_index,:]
        wt = self.engine.w
        ufss.signals.plot2D(self.wtau,wt,signal,part='real')
        plt.xlabel(r'$\omega_{\tau}$')
        plt.ylabel(r'$\omega_t - \omega_c$')

class Rephasing(Spectroscopy2D):

    def __init__(self,file_name,*,engine_name = 'UF2',conserve_memory=False):
        super().__init__(file_name,engine_name = engine_name,
                         detection_type='complex_polarization',
                         conserve_memory=conserve_memory)

        pdc = ((0,1),(1,0),(1,0))
        self.set_phase_discrimination(pdc)
        self.save_name = 'rephasing_signals'
        self.rephasing = True #used for tau Fourier convention

    def set_highest_order(self,highest_order):
        pulse_orders = (highest_order-2,highest_order-2,1)
        self.engine.include_higher_orders(pulse_orders)

class NonRephasing(Spectroscopy2D):

    def __init__(self,file_name,*,engine_name = 'UF2',conserve_memory=False):
        super().__init__(file_name,engine_name = engine_name,
                         detection_type='complex_polarization',
                         conserve_memory=conserve_memory)

        pdc = ((1,0),(0,1),(1,0))
        self.set_phase_discrimination(pdc)
        self.save_name = 'nonrephasing_signals'
        self.rephasing = False #used for tau Fourier convention

    def set_highest_order(self,highest_order):
        pulse_orders = (highest_order-2,highest_order-2,1)
        self.engine.include_higher_orders(pulse_orders)

class PumpPumpProbe(Spectroscopy2D):
    def __init__(self,file_name,*,engine_name = 'UF2',conserve_memory=False,
                 include_linear=False,include_wtau_DC_terms=False):
        super().__init__(file_name,engine_name=engine_name,
                         conserve_memory=conserve_memory)
        self.pdc = ((0,0),(0,0),(1,0))
        self.save_name = 'pump_pump_probe_signals'
        self.rephasing = True #used for tau Fourier convention
        self.include_linear = include_linear
        self.include_wtau_DC_terms = include_wtau_DC_terms

    def set_identical_gaussians(self,sigma,center,*,M=51,delta=10):
        t = np.linspace(-sigma,sigma,num=M,endpoint=True)*delta/2
        f = ufss.gaussian(t,sigma)
        pump = f * np.exp(-1j*center*t)
        probe = f
        times = [t,t,t]
        efields = [pump,pump,probe]
        centers = [0,0,center]
        self.engine.set_efields(times,efields,centers,self.pdc)
        self.engine.set_polarization_sequence(['x']*3)

    def set_impulsive_pulses(self,center):
        self.engine.set_impulsive_pulses(self.pdc)
        self.engine.set_polarization_sequence(['x']*3)
        self.engine.centers = [0,0,center]
            
    def set_highest_order(self,highest_order):
        pulse_orders = [highest_order-1,highest_order-1,1]
        lowest_order_pdcs = []
        for i in range(highest_order//2):
            r,nr = self.pump_pump_probe_pdcs(i+1)
            lowest_order_pdcs += [r,nr]

        self.engine.include_higher_orders(pulse_orders)
        for pdc in lowest_order_pdcs:
            self.engine.include_additional_pdc(pdc)
        if self.include_wtau_DC_terms:
            if not self.include_linear:
                self.engine.signal_pdcs.remove(self.pdc)
        else:
            self.engine.remove_signal_pdcs(self.pdc)
            if self.include_linear:
                self.engine.signal_pdcs.append(self.pdc)

    def pump_pump_probe_pdcs(self,n):
        """Returns (non-)rephasing lowest order nQ pdc"""
        r = ((0,n),(n,0),(1,0))
        nr = ((n,0),(0,n),(1,0))
        return r,nr
    
    def load(self,file_name,*,use_base_path=True):
        self.engine.load(file_name,pulse_delay_names=[],
                         use_base_path=use_base_path)
        self.tau = self.engine.all_pulse_delays[0]
        self.T = self.engine.all_pulse_delays[1]
        

