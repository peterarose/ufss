from ufss.UF2 import DensityMatrices
import ufss
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from ufss import efieldConvergence

# Fixed parameters
site_energies = [0,1]
site_couplings = [0]
dipoles = [[1,0,0],[1,0,0]]
d = 0
folder = 'UF2_test'
os.makedirs(folder,exist_ok=True)
vibrations = []
# vibrations = [{'displacement':d,'site_label':0,'omega_g':1},
               # {'displacement':d,'site_label':1,'omega_g':1.001}]

overdamped_bath = {'cutoff_frequency':1,
                   'coupling':0.1,
                   'temperature':0.25,
                   'cutoff_function':'lorentz-drude',
                   'spectrum_type':'ohmic'}

bath = {'secular':True,
        'site_bath':overdamped_bath,
        'vibration_bath':overdamped_bath}

def save_params(trunc_size):
    params = {
        'site_energies':site_energies,
        'site_couplings':site_couplings,
        'dipoles':dipoles,
        'truncation_size':trunc_size,
        'num_eigenvalues':'full',
        'eigenvalue_precision':1,
        'vibrations':vibrations,
        'maximum_manifold':1,
        'bath':bath}

    with open(os.path.join(folder,'simple_params.yaml'),'w+') as new_file:
        yaml.dump(params,new_file)

def run_TA(dt,Delta,*,sigma=1):
    TA = DensityMatrices(os.path.join(folder,'open'),detection_type='polarization')
    tmax = Delta/2
    n = round(tmax/dt)
    t = np.arange(-n,n+1/2,1)*dt*sigma
    dt = t[1] - t[0]
    tmax = t[-1]
    ef = ufss.efield_shapes.gaussian(t,sigma)

    TA.set_polarization_sequence(['x','x'])
    TA.maximum_manifold = 1

    TA.set_efields([t,t],[ef,ef],[0,0],[(1,1),(1,0)])
    
    TA.set_t(0.05,dt=1)

    T = np.arange(0,52,2)
    TA.set_pulse_delays([T])
    TA.calculate_signal_all_delays()

    return TA


def main():
    save_params(1)
    ufss.HLG.run(folder)

    sigma = 1
    
    dts = np.logspace(-3.2,0,num=20)
    Deltas = np.array([4,6,8,10,12,14,20])
    f = lambda x,y: run_TA(x,y,sigma=sigma)
    c = efieldConvergence(f,dts,Deltas)
    print(c.ref_params)
    c.run()
    print('Minimum M for 1% threshold',c.find_minimum_M(signal_threshold=1E-2))
    c.plot()
    plt.savefig('efield_convergence_sigma_{:.2f}.png'.format(sigma))
    plt.show()

if __name__ == '__main__':
    main()
