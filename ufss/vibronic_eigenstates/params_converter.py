import yaml
import os
import itertools
import copy
import shutil
import math

def convert_simple_dict(vib_dict,total_num_sites):
    site = vib_dict['site_label']
    d = vib_dict['displacement']
    try:
        alpha = vib_dict['alpha']
    except KeyError:
        alpha = 1
    p0 = [1]
    p1 = [alpha**2]
    k0 = [1]
    k1 = [1]
    try:
        l = vib_dict['reorganization']
    except KeyError:
        l = -alpha**2 * d**2
    
    k_modifier = [alpha**(i/4) for i in range(2,len(k0)+2)]
    new_k0 = [a*b for a,b in zip(k0,k_modifier)]
    new_k1 = [a*b for a,b in zip(k1,k_modifier)]
    
    p_modifier = [alpha**(-i/4) for i in range(2,len(p0)+2)]
    new_p0 = [a*b for a,b in zip(p0,p_modifier)]
    new_p1 = [a*b for a,b in zip(p1,p_modifier)]
    
    new_d = d*alpha**(1/4)/2
    
    total_sites = total_num_sites
    SEM_indices = list(range(total_sites))
    displacement1 = [-new_d for i in SEM_indices]
    displacement1[site] = new_d
    reorganization1 = [0 for i in SEM_indices]
    reorganization1[site] = l
    potential1 = [new_p0 for i in SEM_indices]
    potential1[site] = new_p1
    kinetic1 = [new_k0 for i in SEM_indices]
    kinetic1[site] = new_k1
    
    DEM_indices = list(itertools.combinations(range(total_sites),2))
    displacement2 = [-new_d for i in DEM_indices]
    reorganization2 = [0 for i in DEM_indices]
    potential2 = [new_p0 for i in DEM_indices]
    kinetic2 = [new_k0 for i in DEM_indices]
    for i in range(len(DEM_indices)):
        a,b = DEM_indices[i]
        if site == a or site == b:
            displacement2[i] = new_d
            potential2[i] = new_p1
            kinetic2[i] = new_k1
            reorganization2[i] = l
    new_vib_dict = {'displacement0':[-new_d],'displacement1':displacement1,'displacement2':displacement2,
                    'omega_g':vib_dict['omega_g'],'kinetic0':[new_k0],'kinetic1':kinetic1,'kinetic2':kinetic2,
                    'potential0':[new_p0],'potential1':potential1,'potential2':potential2,
                   'reorganization0':[0],'reorganization1':reorganization1,'reorganization2':reorganization2}
    try:
        new_vib_dict['condon_violation'] = vib_dict['condon_violation']
    except KeyError:
        pass
    return new_vib_dict

def convert_simple_dict_open(vib_dict):
    site = vib_dict['site_label']
    d = vib_dict['displacement']
    try:
        alpha = vib_dict['alpha']
    except KeyError:
        alpha = 1
    p0 = [1]
    p1 = [alpha**2]
    k0 = [1]
    k1 = [1]
    try:
        l = vib_dict['reorganization']
    except KeyError:
        l = -alpha**2 * d**2
    
    k_modifier = [alpha**(i/4) for i in range(2,len(k0)+2)]
    new_k0 = [a*b for a,b in zip(k0,k_modifier)]
    new_k1 = [a*b for a,b in zip(k1,k_modifier)]
    
    p_modifier = [alpha**(-i/4) for i in range(2,len(p0)+2)]
    new_p0 = [a*b for a,b in zip(p0,p_modifier)]
    new_p1 = [a*b for a,b in zip(p1,p_modifier)]
    
    new_d = d*alpha**(1/4)/2

    potential1 = new_p1
    kinetic1 = new_k1

    new_vib_dict = {'displacement':[-new_d,new_d],'reorganization':[0,l],
                    'omega_g':vib_dict['omega_g'],'kinetic':[new_k0,new_k1],
                    'potential':[new_p0,new_p1],'site_label':site}
    try:
        new_vib_dict['condon_violation'] = vib_dict['condon_violation']
    except KeyError:
        pass
    return new_vib_dict

def convert_dict(vib_dict,total_num_sites):
    site = vib_dict['site_label']
    d = vib_dict['displacement']
    try:
        k0 = vib_dict['kinetic0']
    except KeyError:
        k0 = [1]
    try:
        k1 = vib_dict['kinetic1']
    except KeyError:
        k1 = [1]
    try:
        p0 = vib_dict['potential0']
    except KeyError:
        p0 = [1]
    try:
        p1 = vib_dict['potential1']
    except KeyError:
        p1 = [1]
    alpha = math.sqrt(p1[0]/p0[0])
    try:
        l = vib_dict['reorganization']
    except KeyError:
        l = -alpha**2 * d**2
    
    k_modifier = [alpha**(i/4) for i in range(2,len(k0)+2)]
    new_k0 = [a*b for a,b in zip(k0,k_modifier)]
    new_k1 = [a*b for a,b in zip(k1,k_modifier)]
    
    p_modifier = [alpha**(-i/4) for i in range(2,len(p0)+2)]
    new_p0 = [a*b for a,b in zip(p0,p_modifier)]
    new_p1 = [a*b for a,b in zip(p1,p_modifier)]
    
    new_d = d*alpha**(1/4)/2
    
    total_sites = total_num_sites
    SEM_indices = list(range(total_sites))
    displacement1 = [-new_d for i in SEM_indices]
    displacement1[site] = new_d
    reorganization1 = [0 for i in SEM_indices]
    reorganization1[site] = l
    potential1 = [new_p0 for i in SEM_indices]
    potential1[site] = new_p1
    kinetic1 = [new_k0 for i in SEM_indices]
    kinetic1[site] = new_k1
    
    DEM_indices = list(itertools.combinations(range(total_sites),2))
    displacement2 = [-new_d for i in DEM_indices]
    reorganization2 = [0 for i in DEM_indices]
    potential2 = [new_p0 for i in DEM_indices]
    kinetic2 = [new_k0 for i in DEM_indices]
    for i in range(len(DEM_indices)):
        a,b = DEM_indices[i]
        if site == a or site == b:
            displacement2[i] = new_d
            potential2[i] = new_p1
            kinetic2[i] = new_k1
            reorganization2[i] = l
    new_vib_dict = {'displacement0':[-new_d],'displacement1':displacement1,'displacement2':displacement2,
                    'omega_g':vib_dict['omega_g'],'kinetic0':[new_k0],'kinetic1':kinetic1,'kinetic2':kinetic2,
                    'potential0':[new_p0],'potential1':potential1,'potential2':potential2,
                   'reorganization0':[0],'reorganization1':reorganization1,'reorganization2':reorganization2}
    try:
        new_vib_dict['condon_violation'] = vib_dict['condon_violation']
    except KeyError:
        pass
    return new_vib_dict

def convert_open(base_path,*,convert_type='simple'):
    simple_file_name = os.path.join(base_path,'simple_params.yaml')
    parameter_file_name = os.path.join(base_path,'open_params.yaml')
    with open(simple_file_name,'r') as yamlstream:
        simple = yaml.load(yamlstream,Loader=yaml.SafeLoader)
    
    params = dict()
    params['dipoles'] = simple['dipoles']
    params['site_energies'] = simple['site_energies']
    params['site_couplings'] = simple['site_couplings']
    params['initial truncation size'] = simple['truncation_size']
    try:
        params['number eigenvalues'] = simple['num_eigenvalues']
        params['eigenvalue precision'] = simple['eigenvalue_precision']
    except KeyError:
        pass

    try:
        params['maximum_manifold'] = simple['maximum_manifold']
    except KeyError:
        pass

    try:
        params['kT'] = simple['kT']
    except KeyError:
        pass

    try:
        params['site_to_site_relaxation_gamma'] = simple['site_to_site_relaxation_gamma']
    except KeyError:
        pass

    try:
        params['site_to_site_dephasing_gamma'] = simple['site_to_site_dephasing_gamma']
    except KeyError:
        pass

    try:
        params['vibrational_gamma'] = simple['vibrational_gamma']
    except KeyError:
        pass

    try:
        params['optical_dephasing_gamma'] = simple['optical_dephasing_gamma']
    except KeyError:
        pass

    try:
        params['optical_relaxation_gamma'] = simple['optical_relaxation_gamma']
    except KeyError:
        pass
    
    simple_vibrations = simple['vibrations']
    
    if convert_type == 'simple':
        vibrations = [convert_simple_dict_open(vib) for vib in simple_vibrations]

    params['vibrations'] = vibrations

    with open(parameter_file_name,'w+') as yamlstream:
        yamlstream.write('### This file was generated by UFSS, do not modify unless you know what you are doing \n')
        yaml.dump(params,yamlstream)

def convert(base_path,*,convert_type='simple'):
    simple_file_name = os.path.join(base_path,'simple_params.yaml')
    parameter_file_name = os.path.join(base_path,'params.yaml')
    with open(simple_file_name,'r') as yamlstream:
        simple = yaml.load(yamlstream,Loader=yaml.SafeLoader)
    
    params = dict()
    params['dipoles'] = simple['dipoles']
    params['site_energies'] = simple['site_energies']
    params['site_couplings'] = simple['site_couplings']
    params['number eigenvalues'] = simple['num_eigenvalues']
    params['eigenvalue precision'] = simple['eigenvalue_precision']
    params['initial truncation size'] = simple['truncation_size']
    try:
        params['auto_DEM'] = simple['auto_DEM']
    except KeyError:
        params['auto_DEM'] = True

    simple_vibrations = simple['vibrations']
    n = len(simple['site_energies'])
    
    if convert_type == 'simple':
        vibrations = [convert_simple_dict(vib,n) for vib in simple_vibrations]
    elif convert_type == 'anharmonic':
        vibrations = [convert_dict(vib,n) for vib in simple_vibrations]

    params['vibrations'] = vibrations

    with open(parameter_file_name,'w+') as yamlstream:
        yamlstream.write('### This file was generated by UFSS, do not modify unless you know what you are doing \n')
        yaml.dump(params,yamlstream)

    
