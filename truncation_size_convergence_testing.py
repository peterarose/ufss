from ufss.UF2 import DensityMatrices
import ufss
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from ufss import Convergence

# Fixed parameters
site_energies = [0,0.75]
site_couplings = [0.3307]
dipoles = [[1,0,0],[1,0,0]]
d = 0.2
folder = 'convergence_test_truncation_size'
os.makedirs(folder,exist_ok=True)

vibrations = [{'displacement':d,'site_label':0,'omega_g':1},
               {'displacement':d,'site_label':1,'omega_g':1.001}]

overdamped_bath = {'cutoff_frequency':1,
                   'coupling':0.1,
                   'temperature':0.25,
                   'cutoff_function':'lorentz-drude',
                   'spectrum_type':'ohmic'}

bath = {'secular':True,
        'site_bath':overdamped_bath,
        'vibration_bath':overdamped_bath}

def save_params(trunc_size):
    # saves fixed parameters, along with input trunc_size
    params = {
        'site_energies':site_energies,
        'site_couplings':site_couplings,
        'dipoles':dipoles,
        'truncation_size':trunc_size,
        'num_eigenvalues':'full',
        'eigenvalue_precision':1,
        'vibrations':vibrations,
        'maximum_manifold':2,
        'bath':bath}

    with open(os.path.join(folder,'simple_params.yaml'),'w+') as new_file:
        yaml.dump(params,new_file)

def run_TA(trunc_size):
    save_params(trunc_size)
    ufss.HLG.run(folder)
    TA = DensityMatrices(os.path.join(folder,'open'),detection_type='polarization')

    sigma = 1
    Delta = 10*sigma
    dt = 0.25*sigma

    tmax = Delta/2
    n = round(tmax/dt)
    t = np.arange(-n,n+1/2,1)*dt*sigma
    dt = t[1] - t[0]
    tmax = t[-1]
    ef = ufss.efield_shapes.gaussian(t,sigma)

    TA.set_polarization_sequence(['x','x'])
    TA.maximum_manifold = 2

    TA.set_efields([t,t],[ef,ef],[0,0],[(1,1),(1,0)])
    
    TA.set_t(0.05,dt=1)

    T = np.arange(0,50,2)
    TA.set_pulse_delays([T])
    TA.calculate_signal_all_delays()
    
    return TA


def main():
    truncation_sizes = list(range(1,6)) # test sizes 1 to 5
    conv_params = [truncation_sizes]
    ref_truncation_size = 6 # test against truncation size of 6
    ref_params = [ref_truncation_size]
    
    # Convergence calculate the function run_TA at the reference parameter
    # and then does a loop over the truncation_sizes specified. For each
    # truncation size studied, it takes the L2 norm with respect to the
    # reference calculation.
    c = Convergence(run_TA,conv_params,ref_params)
    print(c.ref_params)
    c.run()
    plt.figure()
    plt.semilogy(truncation_sizes,c.L2norms,'-o')
    plt.axhline(y=1E-2,linestyle='--',color='k')
    plt.xlabel('Truncation size')
    plt.ylabel('$L_2$ norm')
    plt.show()

if __name__ == '__main__':
    main()
