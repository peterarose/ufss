import os
import yaml
from ufss.HLG import run

def make_2LS(folder_name,omega0,tau_deph,kT,secular = True,tau_relax = 'auto',
             conserve_memory = False):
    """Makes a 2LS with the energy separation between the two levels set to 1
    Args:
        folder_name (str) : folder for the calculations
        omega0 (radians / fs) : energy separation of two levels, divided by h bar
        tau_deph (fs) : dephasing time for the system
        kT (radians / fs) : temperature

    Kwargs:
        secular (bool) : use secular Redfield if True, otherwise use full Redfield
        tau_relax (fs) : relaxation time for the excited setate. If 'auto', use tau_deph*1000
    """
    if tau_relax == 'auto':
        tau_relax = tau_deph * 1000

    taud = float(tau_deph * omega0)
    taur = float(tau_relax * omega0)
    kT = float(kT / omega0) # kT must be provided in the units of radians / time, to match omega0

    site_energies = [1] # hbar*omega0 / hbar*omega0

    site_couplings = []

    dipoles = [[1,0,0]]

    trunc_size = 1

    os.makedirs(folder_name,exist_ok=True)

    site_bath = {'dephasing_rate':1/taud, # additonal "pure" dephasing
              'relaxation_rate':0,
               'temperature':kT,
                'spectrum_type':'white-noise'}

    vibration_bath = site_bath # use same bath for sites and vibrations

    relax_bath = {'dephasing_rate':0, # additonal "pure" dephasing
              'relaxation_rate':1/taur,
               'temperature':kT,
                'spectrum_type':'white-noise'}

    Redfield_bath = {'secular':secular,
        'site_bath':site_bath,
        'vibration_bath':vibration_bath,
        'site_internal_conversion_bath':relax_bath}

    vibrations = []

    params = {
        'site_energies':site_energies,
        'site_couplings':site_couplings,
        'dipoles':dipoles,
        'truncation_size':trunc_size,
        'vibrations':vibrations,
        'bath':Redfield_bath,
        'maximum_manifold':3}

    with open(os.path.join(folder_name,'simple_params.yaml'),'w+') as new_file:
        yaml.dump(params,new_file)

    run(folder_name,conserve_memory=conserve_memory)

def make_3LS(folder_name,omega0,tau_deph,kT,secular = True,tau_relax = 'auto',
             conserve_memory = False,omega21 = 'auto',dipole12 = 'auto',
             tau_deph2 = 'auto',tau_relax2 = 'auto'):
    """Makes a 2LS with the energy separation between the two levels set to 1
    Args:
        folder_name (str) : folder for the calculations
        omega0 (radians / fs) : energy separation of two levels, divided by h bar
        tau_deph (fs) : dephasing time for the system
        kT (radians / fs) : temperature

    Kwargs:
        secular (bool) : use secular Redfield if True, otherwise use full Redfield
        tau_relax (fs) : relaxation time for the excited setate. If 'auto', use tau_deph*1000
    """
    if tau_relax == 'auto':
        tau_relax = tau_deph * 1000
    if omega21 == 'auto':
        omega21 = 2 * omega0
    if dipole12 == 'auto':
        dipole12 = [1,0,0]
    if tau_deph2 == 'auto':
        tau_deph2 = tau_deph
    if tau_relax2 == 'auto':
        tau_relax2 = tau_relax

    taud = float(tau_deph * omega0)
    taud2 = float(tau_deph2 * omega0)
    taur = float(tau_relax * omega0)
    taur2 = float(tau_relax2 * omega0)
    kT = float(kT / omega0) # kT must be provided in the units of radians / time, to match omega0
    site_energies = [[1,float(omega21/omega0)],]

    site_couplings = [[],[],[]]

    dipole01 = [1,0,0]
    dipole02 = [0,0,0]

    dipoles = [[dipole01,dipole12,dipole02],]

    trunc_size = 1

    os.makedirs(folder_name,exist_ok=True)

    site_bath = {'dephasing_rate':1/taud, # additonal "pure" dephasing
              'relaxation_rate':0,
               'temperature':kT,
                'spectrum_type':'white-noise'}
    
    site_bath2 = {'dephasing_rate':1/taud2, # additonal "pure" dephasing
              'relaxation_rate':0,
               'temperature':kT,
                'spectrum_type':'white-noise'}

    vibration_bath = site_bath # use same bath for sites and vibrations

    relax_bath = {'dephasing_rate':0, # additonal "pure" dephasing
              'relaxation_rate':1/taur,
               'temperature':kT,
                'spectrum_type':'white-noise'}
    
    relax_bath2 = {'dephasing_rate':0, # additonal "pure" dephasing
              'relaxation_rate':1/taur2,
               'temperature':kT,
                'spectrum_type':'white-noise'}

    Redfield_bath = {'secular':secular,
        'site_bath':site_bath,
        'site_bath2':site_bath2,
        'vibration_bath':vibration_bath,
        'site_internal_conversion_bath':relax_bath,
        'site_internal_conversion_bath21':relax_bath2}

    vibrations = []

    params = {
        'site_energies':site_energies,
        'site_couplings':site_couplings,
        'dipoles':dipoles,
        'truncation_size':trunc_size,
        'vibrations':vibrations,
        'bath':Redfield_bath,
        'maximum_manifold':3}

    with open(os.path.join(folder_name,'simple_params.yaml'),'w+') as new_file:
        yaml.dump(params,new_file)

    run(folder_name,conserve_memory=conserve_memory)
