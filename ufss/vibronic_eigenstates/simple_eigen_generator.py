#Standard python libraries
import numpy as np
import os
import yaml
import warnings
import numbers
import itertools
from datetime import datetime
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from .eigenstates import PolymerVibrations
from .base_class import DataOrganizer

class SimpleEigenGenerator(PolymerVibrations,DataOrganizer):
    """This class generates the eigenvalues and eigenvectors
        needed for the specified system in the params file.  This
        class should be initialized once at the beginning of any calculation 
        run, and the generate and save methods should both be called once."""
    def __init__(self,parameter_file_path,*,
                 qr_flag=False,mask_by_occupation_num=True):
        parameter_file = os.path.join(parameter_file_path,'params.yaml')
        super().__init__(parameter_file,qr_flag=qr_flag,
                         mask_by_occupation_num=mask_by_occupation_num)
        self.base_path = parameter_file_path
        self.load_params()
        self.set_diagrams_and_manifolds()
        self.generate_eigenvalues_and_vectors_low_memory()
        self.combine_eigensystem()

    def set_diagrams_and_manifolds(self):
        if len(self.energies) == 2:
            self.diagrams = ['GSB', 'SE']
            self.manifolds = ['GSM','SEM']
            self.diagrams_by_manifold = {'GSM':['GSB'],'SEM':['SE']}
        elif len(self.energies) == 3:
            self.diagrams = ['GSB','SE','ESA']
            self.manifolds = ['GSM','SEM','DEM']
            self.diagrams_by_manifold = {'GSM':['GSB'],'SEM':['SE','ESA']}

    def generate_eigenvalues_and_vectors_low_memory(self):
        """Use the methods from PolymerVibrations to construct the 
            eigenvalues and eigenmatrices for the GSM, SEM, and if 
            applicable, the DEM
"""
        self.time_to_find_eigensystem = []

        N = self.params['number eigenvalues']

        for i in range(len(self.energies)):
            t0 = time.time()
            if N == 'full':
                print('diagonalizing 1 manifold...')
                eigvals, eigvecs = self.eig(i)
            else:
                eigvals, eigvecs = self.eigs(N,i)
            t1 = time.time()
            np.save(os.path.join(self.base_path,'Manifold{}_eigenvalues.npy'.format(i)),eigvals)
            np.save(os.path.join(self.base_path,'Manifold{}_eigenvectors.npy'.format(i)),eigvecs)
            del eigvals
            del eigvecs
            self.time_to_find_eigensystem.append(t1-t0)
        times = np.array([self.time_to_find_eigensystem])
        np.save(os.path.join(self.base_path,'Times_to_diagonalize.npy'),times)

    def combine_eigensystem(self):
        self.eigenvalues = []
        self.eigenvectors = []
        for i in range(len(self.energies)):
            eigvals = np.load(os.path.join(self.base_path,'Manifold{}_eigenvalues.npy'.format(i)))
            eigvecs = np.load(os.path.join(self.base_path,'Manifold{}_eigenvectors.npy'.format(i)))
            self.eigenvalues.append(eigvals)
            self.eigenvectors.append(eigvecs)
        try:
            times = self.time_to_find_eigensystem
        except:
            times = np.load(os.path.join(self.base_path,'Times_to_diagonalize.npy'))
            times = [float(time) for time in times]

        self.ground_ZPE = np.min(self.eigenvalues[0])
        for i in range(len(self.eigenvalues)):
            self.eigenvalues[i] -= self.ground_ZPE
            
        # The RWA, using the ground electronic ground vibrational to the
        # lowest singly-excited vibronic transition
        self.ground_to_excited_transition = np.min(self.eigenvalues[1])
        for i in range(1,len(self.energies)):
            self.eigenvalues[i] -= i * self.ground_to_excited_transition

        eigval_save_name = os.path.join(self.base_path,'eigenvalues.npz')
        eigvec_save_name = os.path.join(self.base_path,'eigenvectors.npz')
        eigval_dict = {key:arr for key,arr in zip(self.manifolds,self.eigenvalues)}
        eigvec_dict = {key:arr for key,arr in zip(self.manifolds,self.eigenvectors)}
        np.savez(eigval_save_name,**eigval_dict)
        np.savez_compressed(eigvec_save_name,**eigvec_dict)
        eigen_params = {'final truncation size':self.truncation_size,
                        'ground to excited transition':float(self.ground_to_excited_transition),
                        'ground zero point energy':float(self.ground_ZPE),
                        'diagonalization time':times}
        with open(os.path.join(self.base_path,'eigen_params.yaml'),'w+') as yamlstream:
            yamlstream.write( yaml.dump(eigen_params,
                                        default_flow_style=False))

