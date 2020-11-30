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

class EigenGenerator(PolymerVibrations,DataOrganizer):
    """This class generates the eigenvalues and eigenvectors
        needed for the specified system in the params file.  This
        class should be initialized once at the beginning of any calculation 
        run, and the generate and save methods should both be called once."""
    def __init__(self,parameter_file_path,*,check_convergence=True,
                 qr_flag=False,mask_by_occupation_num=True):
        parameter_file = os.path.join(parameter_file_path,'params.yaml')
        super().__init__(parameter_file,qr_flag=qr_flag,
                         mask_by_occupation_num=mask_by_occupation_num)
        self.base_path = parameter_file_path
        self.load_params()

        self.set_diagrams_and_manifolds()

        self.check_convergence = check_convergence
            
        try:
            self.load_eigenvalues_and_vectors()
        except (FileNotFoundError, KeyError):
            self.generate_eigenvalues_and_vectors_with_tolerance()
            self.save_eigenvalues_and_vectors()

    def set_diagrams_and_manifolds(self):
        if len(self.energies) == 2:
            self.diagrams = ['GSB', 'SE']
            self.manifolds = ['GSM','SEM']
            self.diagrams_by_manifold = {'GSM':['GSB'],'SEM':['SE']}
        elif len(self.energies) == 3:
            self.diagrams = ['GSB','SE','ESA']
            self.manifolds = ['GSM','SEM','DEM']
            self.diagrams_by_manifold = {'GSM':['GSB'],'SEM':['SE','ESA']}

    def recalculate_eigenvalues_and_vectors(self):
        self.truncation_size = self.params['initial truncation size']
        self.generate_eigenvalues_and_vectors_with_tolerance()
        self.save_eigenvalues_and_vectors()

    def generate_eigenvalues_and_vectors_with_tolerance(self,*,save_intermediate = False):
        eig_tol = self.params['eigenvalue precision']
        rel_err = 1 if self.check_convergence else 0
        N = self.params['number eigenvalues']
        if N == 'full':
            # If using eigh instead of eigsh
            rel_err = 0
        max_iter = 20
        self.generate_eigenvalues_and_vectors()
        
        old_eigvals = np.hstack((self.eigenvalues[0],self.eigenvalues[1]))
        if 'DEM' in self.manifolds:
            old_eigvals = np.hstack((old_eigvals,self.eigenvalues[2]))
        iter_count = 0
        while rel_err > eig_tol:
            if save_intermediate:
                eigval_save_name = os.path.join(self.base_path,'eigenvalues_eigs_truncation{}.npz'.format(self.truncation_size))
                eigval_dict = {key:arr for key,arr in zip(self.manifolds,self.eigenvalues)}
                np.savez(eigval_save_name,**eigval_dict)
            self.truncation_size += 1
            print(self.truncation_size)
            self.set_vibrations()
            self.generate_eigenvalues_and_vectors()
            new_eigvals = np.hstack((self.eigenvalues[0],self.eigenvalues[1]))
            if 'DEM' in self.manifolds:
                new_eigvals = np.hstack((new_eigvals,self.eigenvalues[2]))
            
            # There will always be eigenvalues with value 0, hence the plus 1
            rel_err = np.max(np.abs((new_eigvals - old_eigvals)/(new_eigvals+1)))
            old_eigvals = new_eigvals
            iter_count += 1
            if iter_count > max_iter:
                warnings.warn('Eigenvalues did not converge to requested tolerance')
                break

    def generate_eigenvalues_and_vectors_with_tolerance_eig(self,*,save_intermediate=False):
        eig_tol = self.params['eigenvalue precision']
        rel_err = 1 if self.check_convergence else 0
        N = self.params['number eigenvalues']
        max_iter = 20
        self.generate_eigenvalues_and_vectors_eig()
        
        old_eigvals = np.hstack((self.eigenvalues[0],self.eigenvalues[1]))
        if 'DEM' in self.manifolds:
            old_eigvals = np.hstack((old_eigvals,self.eigenvalues[2]))
        
        iter_count = 0
        while rel_err > eig_tol:
            if save_intermediate:
                eigval_save_name = os.path.join(self.base_path,'eigenvalues_eig_truncation{}.npz'.format(self.truncation_size))
                eigval_dict = {key:arr for key,arr in zip(self.manifolds,self.eigenvalues)}
                np.savez(eigval_save_name,**eigval_dict)
            self.truncation_size += 2
            self.set_vibrations()
            self.generate_eigenvalues_and_vectors_eig()
            new_eigvals = np.hstack((self.eigenvalues[0],self.eigenvalues[1]))
            
            if 'DEM' in self.manifolds:
                new_eigvals = np.hstack((new_eigvals,self.eigenvalues[2]))
            
            # There will always be eigenvalues with value 0, hence the plus 1
            rel_err = np.max(np.abs((new_eigvals - old_eigvals)/(new_eigvals+1)))
            old_eigvals = new_eigvals
            iter_count += 1
            if iter_count > max_iter:
                warnings.warn('Eigenvalues did not converge to requested tolerance')
                break

    def generate_eigenvalues_and_vectors(self):
        """Use the methods from PolymerVibrations to construct the 
            eigenvalues and eigenmatrices for the GSM, SEM, and if 
            applicable, the DEM
"""
        self.eigenvalues = []
        self.eigenvectors = []
        self.time_to_find_eigensystem = []

        N = self.params['number eigenvalues']

        for i in range(len(self.energies)):
            t0 = time.time()
            if N == 'full':
                eigvals, eigvecs = self.eig(i)
            else:
                eigvals, eigvecs = self.eigs(N,i)
            t1 = time.time()
            self.eigenvalues.append(eigvals)
            self.eigenvectors.append(eigvecs)
            self.time_to_find_eigensystem.append(t1-t0)

        # Subtract out the zero point energy
        self.ground_ZPE = np.min(self.eigenvalues[0])
        for i in range(len(self.eigenvalues)):
            self.eigenvalues[i] -= self.ground_ZPE
            
        # The RWA, using the ground electronic ground vibrational to the
        # lowest singly-excited vibronic transition
        self.ground_to_excited_transition = np.min(self.eigenvalues[1])
        for i in range(1,len(self.energies)):
            self.eigenvalues[i] -= i * self.ground_to_excited_transition

    def generate_eigenvalues_and_vectors_eig(self):
        """Use the methods from PolymerVibrations to construct the 
            eigenvalues and eigenmatrices for the GSM, SEM, and if 
            applicable, the DEM
"""
        self.eigenvalues = []
        self.eigenvectors = []

        N = self.params['number eigenvalues']

        for i in range(len(self.energies)):
            cut_off = N*len(self.energies[i])
            eigvals, eigvecs = self.eig(i)
            eigvals = eigvals[:cut_off]
            
            eigvecs = eigvecs[:,:cut_off]
            self.eigenvalues.append(eigvals)
            self.eigenvectors.append(eigvecs)

        # Subtract out the zero point energy
        self.ground_ZPE = np.min(self.eigenvalues[0])
        for i in range(len(self.eigenvalues)):
            self.eigenvalues[i] -= self.ground_ZPE
            
        # The RWA, using the ground electronic ground vibrational to the
        # lowest singly-excited vibronic transition
        self.ground_to_excited_transition = np.min(self.eigenvalues[1])
        for i in range(1,len(self.energies)):
            self.eigenvalues[i] -= i * self.ground_to_excited_transition

    def save_eigenvalues_and_vectors(self):
        eigval_save_name = os.path.join(self.base_path,'eigenvalues.npz')
        eigvec_save_name = os.path.join(self.base_path,'eigenvectors.npz')
        eigval_dict = {key:arr for key,arr in zip(self.manifolds,self.eigenvalues)}
        eigvec_dict = {key:arr for key,arr in zip(self.manifolds,self.eigenvectors)}
        np.savez(eigval_save_name,**eigval_dict)
        np.savez_compressed(eigvec_save_name,**eigvec_dict)
        eigen_params = {'final truncation size':self.truncation_size,
                        'ground to excited transition':float(self.ground_to_excited_transition),
                        'ground zero point energy':float(self.ground_ZPE),
                        'diagonalization time':self.time_to_find_eigensystem}
        with open(os.path.join(self.base_path,'eigen_params.yaml'),'w+') as yamlstream:
            yamlstream.write( yaml.dump(eigen_params,
                                        default_flow_style=False))

    def load_eigenvalues_and_vectors(self):
        eigval_save_name = os.path.join(self.base_path,'eigenvalues.npz')
        eigvec_save_name = os.path.join(self.base_path,'eigenvectors.npz')
        eigval_archive = np.load(eigval_save_name)
        eigvec_archive = np.load(eigvec_save_name)
        self.eigenvalues = [eigval_archive[key] for key in self.manifolds]
        self.eigenvectors = [eigvec_archive[key] for key in self.manifolds]

        with open(os.path.join(self.base_path,'eigen_params.yaml'),'r') as yamlstream:
            eigen_params = yaml.load(yamlstream,Loader=yaml.SafeLoader)
            self.truncation_size = eigen_params['final truncation size']
            self.ground_ZPE = eigen_params['ground zero point energy']
            self.ground_to_excited_transition = eigen_params['ground to excited transition']
            try:
                self.time_to_find_eigensystem = eigen_params['diagonalization time']
            except KeyError:
                pass

    def electronic_trace(self,manifold_num,eignum):
        """Traces out the vibrational space, and returns the partial density matrix
in the electronic site basis"""
        num_sites = len(self.energies[manifold_num])
        vibration_space_size = self.truncation_size**self.num_vibrations
        # eigvec = self.eigenvectors[manifold_num][:,eignum].toarray().reshape((num_sites,vibration_space_size))
        eigvec = self.eigenvectors[manifold_num][:,eignum].reshape((num_sites,vibration_space_size))
        
        partial_trace = np.zeros((vibration_space_size,vibration_space_size),dtype='complex')
        num_sites = len(self.energies[manifold_num])
        for i in range(num_sites):
            vec = eigvec[i,...]
            partial_trace += np.outer(vec,vec)
        return partial_trace

    def vibrational_trace(self,manifold_num,eignum):
        """Traces out the vibrational space, and returns the partial density matrix
in the electronic site basis"""
        num_sites = len(self.energies[manifold_num])
        vibration_space_size = self.truncation_size**self.num_vibrations
        # eigvec = self.eigenvectors[manifold_num][:,eignum].toarray().reshape((num_sites,vibration_space_size))
        eigvec = self.eigenvectors[manifold_num][:,eignum].reshape((num_sites,vibration_space_size))
        
        partial_trace = np.zeros((num_sites,num_sites),dtype='complex')
        for i in range(num_sites):
            for j in range(num_sites):
                vec_i = eigvec[i,...]
                vec_j = eigvec[j,...]
                rho_ij = np.outer(vec_i,vec_j)
                partial_trace[i,j] = rho_ij.trace()
        return partial_trace
    
    def trace_norm_diff(self,eignum1,eignum2,*,manifold_num=1,partial_trace = 'vibrational'):
        if partial_trace == 'vibrational':
            rho1 = self.vibrational_trace(manifold_num,eignum1)
            rho2 = self.vibrational_trace(manifold_num,eignum2)
        elif partial_trace == 'electronic':
            rho1 = self.electronic_trace(manifold_num,eignum1)
            rho2 = self.electronic_trace(manifold_num,eignum2)
        diff = rho1 - rho2
        sing_vals = np.linalg.svd(diff,compute_uv=False)
        return np.sum(sing_vals)/2

    def dodonov_distance(self,eignum1,eignum2,*,manifold_num=1,partial_trace = 'vibrational'):
        if partial_trace == 'vibrational':
            rho1 = self.vibrational_trace(manifold_num,eignum1)
            rho2 = self.vibrational_trace(manifold_num,eignum2)
        elif partial_trace == 'electronic':
            rho1 = self.electronic_trace(manifold_num,eignum1)
            rho2 = self.electronic_trace(manifold_num,eignum2)
        purity1 = np.dot(rho1,rho1).trace()
        purity2 = np.dot(rho2,rho2).trace()
        overlap = np.dot(rho1,rho2).trace()/np.sqrt(purity1*purity2)
        return 1 - np.abs(overlap)
