#Standard python libraries
import os

#Dependencies
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq

def plot2D(x,y,z,contour_levels = None,part = 'complex',norm = 1,vmax='max',contours_only = False,
           contour_colormap='seismic',contour_colors=None,ax=None,fig=None,colorbar=True,colorbar_ax=None,
           ticks = [],alpha=None,colormap='seismic',figsize='default'):
    if ax == None:
        if figsize == 'default':
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=figsize)
    X,Y = np.meshgrid(x,y,indexing='ij')
    if type(contour_levels) is np.ndarray:
        num_contours = contour_levels.size
        contour_flag = True
    elif type(contour_levels) is list:
        contour_levels = np.array(contour_levels)
        num_contours = contour_levels.size
        contour_flag = True
    else:
        contour_flag = False
        
    if part == 'real':
        Z = z.real
    elif part == 'imag':
        Z = z.imag
    elif part == 'abs':
        Z = np.abs(z)
    elif part == 'complex':
        Z = z
    if norm == 'self':
        norm = np.max(np.abs(Z))
    Z = Z/norm
    if contour_flag:
        if part == 'complex':
            contour_Z = np.abs(Z)
        else:
            contour_Z = Z
        if norm == 1:
            contour_scaling = np.max(np.abs(contour_Z))
        else:
            contour_scaling = 1
        contour_levels = contour_levels * contour_scaling
        if contour_colormap != None:
            handle = ax.contour(X,Y,contour_Z,num_contours,cmap=contour_colormap,
                                levels = contour_levels)
        else:
            handle = ax.contour(X,Y,contour_Z,num_contours,colors=contour_colors,
                                levels = contour_levels)
    if contours_only:
        if colorbar:
            set_colorbar(fig,handle,ticks,colorbar_ax)
    else:
        if part == 'complex':
            handle = ax.imshow(complex_array_to_rgb(Z.T,alpha=alpha), extent=(x[0],x[-1],y[0],y[-1]),origin='lower')
        else:
            if vmax == 'max':
                vmax = np.max(np.abs(Z))
            handle = ax.pcolormesh(X,Y,Z,cmap=colormap,vmin=-vmax,vmax=vmax,shading='Gouraud')
        if colorbar:
            set_colorbar(fig,handle,ticks,colorbar_ax)
    return fig, ax

def set_colorbar(fig,handle,ticks,cax):
    if ticks:
        if cax:
            fig.colorbar(handle,ticks=ticks,cax=cax)
        else:
            fig.colorbar(handle,ticks=ticks)
    else:
        if cax:
            fig.colorbar(handle,cax=cax)
        else:
            fig.colorbar(handle)

def complex_array_to_rgb(X, theme='light', rmax=None,alpha = None):
    '''Taken from https://stackoverflow.com/questions/15207255/is-there-any-way-to-use-bivariate-colormaps-in-matplotlib/17113417#17113417
    Takes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''
    absmax = rmax or np.abs(X).max()
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = matplotlib.colors.hsv_to_rgb(Y)
    if alpha == None:
        print(alpha)
        pass
    else:
        newY = np.zeros(X.shape + (4,), dtype=Y.dtype)
        newY[...,:3] = Y
        newY[...,3] = 1
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if np.all(Y[i,j,:3] == 1):
                    newY[i,j,3] = alpha
        Y = newY
    return Y

class Plotter(object):
    def universal_plotting_commands(self,create_figure=True,save_name=None):
        if create_figure:
            plt.figure()
        if save_name != None:
            plt.savefig(save_name)

    def plotTA(self,*,frequency_range=[-1000,1000], subtract_DC = True, create_figure=True,
               color_range = 'auto',draw_colorbar = True,save_fig=True,scale_factor = 1,
               delay_time_start = -100,save_name='default'):
        """Plots the transient absorption spectra with detection frequency on the
        y-axis and delay time on the x-axis.

        Args:
            frequency_range (list): sets the min (list[0]) and max (list[1]) detection frequency for y-axis
            subtract_DC (bool): if True subtract the DC component of the TA
            color_range (list): sets the min (list[0]) and max (list[1]) value for the colorbar
            draw_colorbar (bool): if True add a colorbar to the plot
            save_fig (bool): if True save the figure that is produced
        """
        # Cut out unwanted detection frequency points
        w_ind = np.where((self.w > frequency_range[0]) & (self.w < frequency_range[1]))[0]
        w = self.w[w_ind]
        sig = self.signal[w_ind,:]*scale_factor

        if subtract_DC:
            sig = self.subtract_DC(sig)
        ww, tt = np.meshgrid(self.delay_times,w)
        if create_figure:
            plt.figure()
        if color_range == 'auto':
            plt.pcolormesh(ww,tt,sig)
        else:
            plt.pcolormesh(ww,tt,sig,vmin=color_range[0],vmax=color_range[1])
        if draw_colorbar:
            plt.colorbar()
        plt.xlabel('Delay time ($\omega_0^{-1}$)',fontsize=16)
        plt.ylabel('Detection Frequency ($\omega_0$)',fontsize=16)
        if save_fig:
            if save_name == 'default':
                plt.savefig(os.path.join(self.base_path,'TA_spectra_iso_ave'))
            else:
                plt.savefig(os.path.join(self.base_path,save_name))
            
    def plotTA_ft(self,*,delay_time_start = 1,create_figure=True,color_range = 'auto',subtract_DC=True,
                   draw_colorbar = True,frequency_range=[-1000,1000],normalize=False,part='abs',
                   save_fig=True,wT_frequency_range = 'auto',omega0 = 1):
        w_ind = np.where((self.w > frequency_range[0]) & (self.w < frequency_range[1]))[0]
        w = self.w[w_ind] * omega0
        sig = self.signal[w_ind,:]

        delay_time_indices = np.where(self.delay_times > delay_time_start)[0]
        delay_times = self.delay_times[delay_time_indices]
        sig = sig[:,delay_time_indices]
        if normalize:
            sig /= np.dot(self.dipoles,self.dipoles)**2
        wT = fftshift(fftfreq(delay_times.size,d=(delay_times[1] - delay_times[0])))*2*np.pi*omega0
        sig_fft = fft(sig,axis=1)
        if subtract_DC:
            sig_fft[:,0] = 0
        sig_fft = fftshift(sig_fft,axes=(1))
        
        ww, wTwT = np.meshgrid(wT,w)

        if create_figure:
            plt.figure()

        if part == 'real':
            plt.title('Real Part')
            plot_sig = np.real(sig_fft)
        elif part == 'imag':
            plt.title('Imag Part')
            plot_sig = np.imag(sig_fft)
        elif part == 'abs':
            plt.title('Magnitude')
            plot_sig = np.abs(sig_fft)
        elif part == 'phase':
            plt.title('Phase')
            plot_sig = np.arctan2(np.imag(sig_fft),np.real(sig_fft))
        else:
            raise Exception('Unknown part keyword argument')
        if color_range == 'auto':
            plt.pcolormesh(ww,wTwT,plot_sig)
        else:
            plt.pcolormesh(ww,wTwT,plot_sig,vmin=color_range[0],vmax=color_range[1])
        if draw_colorbar:
            plt.colorbar()
        if omega0 == 1:
            plt.xlabel('$\omega_T$ ($\omega_0$)',fontsize=16)
            plt.ylabel('Detection Frequency ($\omega_0$)',fontsize=16)
        else:
            plt.xlabel('$\omega_T$ ($cm^{-1}$)',fontsize=16)
            plt.ylabel('Detection Frequency ($cm^{-1}$)',fontsize=16)
            
        if wT_frequency_range == 'auto':
            plt.xlim([0,np.max(wT)])
        else:
            plt.xlim(wT_frequency_range)
        if save_fig:
            plt.savefig(self.base_path + 'TA_spectra_fft')

    def load_eigen_params(self):
        with open(os.path.join(self.base_path,'eigen_params.yaml'),'r') as yamlstream:
            eigen_params = yaml.load(yamlstream,Loader=yaml.SafeLoader)
            self.truncation_size = eigen_params['final truncation size']
            self.ground_ZPE = eigen_params['ground zero point energy']
            self.ground_to_excited_transition = eigen_params['ground to excited transition']

    def plotTA_units(self,*,frequency_range=[-1000,1000], subtract_DC = True, create_figure=True,
                           color_range = 'auto',draw_colorbar = True,save_fig=True,omega_0 = 1):
        """Plots the transient absorption spectra with detection frequency on the
        y-axis and delay time on the x-axis.

        Args:
            frequency_range (list): sets the min (list[0]) and max (list[1]) detection frequency for y-axis
            subtract_DC (bool): if True subtract the DC component of the TA
            color_range (list): sets the min (list[0]) and max (list[1]) value for the colorbar
            draw_colorbar (bool): if True add a colorbar to the plot
            save_fig (bool): if True save the figure that is produced
            omega_0 (float): convert from unitless variables, omega_0 should be provided in wavenumbers
        """
        self.load_eigen_params()
        f0_thz = omega_0 * 3E10/1.0E12 # omega_0 in wavenumbers
        T_ps = self.delay_times / f0_thz / (2*np.pi)
        # Cut out unwanted detection frequency points
        self.w += self.ground_to_excited_transition + self.center
        self.w *= omega_0
        w_ind = np.where((self.w > frequency_range[0]) & (self.w < frequency_range[1]))[0]
        w = self.w[w_ind]
        sig = self.signal[w_ind,:]

        if omega_0 == 1:
            xlab = r'Delay time ($\omega_0^{-1}$)'
            ylab = r'Detection Frequency ($\omega_0$)'
        else:
            xlab = 'Delay time (ps)'
            ylab = r'Detection Frequency (cm$^{-1}$)'

        if subtract_DC:
            sig_fft = fft(sig,axis=1)
            sig_fft[:,0] = 0
            sig = np.real(ifft(sig_fft))
            ww, tt = np.meshgrid(T_ps,w)
        if create_figure:
            plt.figure()
        if color_range == 'auto':
            plt.pcolormesh(ww,tt,sig)
        else:
            plt.pcolormesh(ww,tt,sig,vmin=color_range[0],vmax=color_range[1])
        if draw_colorbar:
            plt.colorbar()
        plt.xlabel(xlab,fontsize=16)
        plt.ylabel(ylab,fontsize=16)
        if save_fig:
            plt.savefig(self.base_path + 'TA_spectra_iso_ave')

    def plot_integrated_TA_units(self,*, create_figure=True,save_fig=True,omega_0 = 1,marker='-o'):
        """Plots the transient absorption spectra with detection frequency on the
        y-axis and delay time on the x-axis.

        Args:
            frequency_range (list): sets the min (list[0]) and max (list[1]) detection frequency for y-axis
            subtract_DC (bool): if True subtract the DC component of the TA
            color_range (list): sets the min (list[0]) and max (list[1]) value for the colorbar
            draw_colorbar (bool): if True add a colorbar to the plot
            save_fig (bool): if True save the figure that is produced
            omega_0 (float): convert from unitless variables, omega_0 should be provided in wavenumbers
        """
        self.load_eigen_params()
        f0_thz = omega_0 * 3E10/1.0E12 # omega_0 in wavenumbers
        T_ps = self.delay_times / f0_thz / (2*np.pi)
        
        sig = np.trapz(self.signal,x=self.w,axis=0)


        xlab = 'Delay time (ps)'

        if create_figure:
            plt.figure()
        plt.plot(T_ps,sig,marker)
        plt.xlabel('Delay time (ps)',fontsize=16)
        plt.ylabel('$S_{TA,FI}(T)$ (arb.)',fontsize=16)
        if save_fig:
            plt.savefig(self.base_path + 'TA_spectra_iso_ave')

    def plot_integrated_signal(self,create_figure = True,scale_factor = 1,
                               frequency_range = 'all',save_fig = False,save_name='default'):
        if create_figure:
            plt.figure()
        plt.plot(self.delay_times,self.integrated_signal(frequency_range=frequency_range)*scale_factor)
        plt.xlabel('Delay time ($\omega_0^{-1}$)')
        plt.ylabel('Frequency-Integrated TA')
        if save_fig:
            if save_name == 'default':
                plt.savefig(os.path.join(self.base_path,'TA_integrated_iso_ave'))
            else:
                plt.savefig(os.path.join(self.base_path,save_name))
            

    def plot_T_slice(self,T,*,create_figure = True,scale_factor = 1,subtract_DC = True):
        if subtract_DC:
            sig = self.subtract_DC(self.signal)
        else:
            sig = self.signal
        T_ind, T = self.get_closest_index_and_value(T,self.delay_times)
        sig = sig[:,T_ind] * scale_factor
        if create_figure:
            plt.figure()
        plt.title('T = {:.2f}'.format(T))
        plt.plot(self.w,sig)

    def plot_w_slice(self,w,*,create_figure = True,scale_factor = 1,subtract_DC = True):
        if subtract_DC:
            sig = self.subtract_DC(self.signal)
        w_ind, w = self.get_closest_index_and_value(w,self.w)
        sig = sig[w_ind,:] * scale_factor
        if create_figure:
            plt.figure()
        plt.title('w = {:.2f}'.format(w))
        plt.plot(self.delay_times,sig)

    def plot_wT_slice(self,wT,*,create_figure = True,scale_factor = 1,subtract_DC = True,part='real'):
        sig = self.ft_axis1(self.signal,zero_DC = subtract_DC)
        wT_ind,wT = self.get_closest_index_and_value(wT,self.wT)
        sig = sig[:,wT_ind]
        if part == 'real':
            plt_title = 'Real Part'
            plot_sig = np.real(sig)
        elif part == 'imag':
            plt_title = 'Imag Part'
            plot_sig = np.imag(sig)
        elif part == 'abs':
            plt_title = 'Magnitude'
            plot_sig = np.abs(sig)
        elif part == 'phase':
            plt_title = 'Phase'
            plot_sig = np.arctan2(np.imag(sig),np.real(sig))
        else:
            raise Exception('Unknown part keyword argument')
        if create_figure:
            plt.figure()
        plt_title += ', $\omega_T$ = {:.2f}'.format(wT)
        plt.title(plt_title)
        plt.plot(self.w,plot_sig,'.')
