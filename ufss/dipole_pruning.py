"""
Module dipole_pruning
================
This module supplies one class, DipolePruning, that is used to find the
minimum number of elements required to resolve the dipole operator, to the
requested tolerance.
"""

import os

import numpy as np

class DipolePruning(object):
    r"""This class takes the dipole operator, mu, which must be expressed in
the eigenbasis of the system hamiltonian and uses Bessel's
inequality to determine the smallest number of states needed to
correctly resolve mu, to the given tolerance.  It expects a file
mu.npz in the folder 'file_path', with archive keys 'mu_GSM_to_SEM' and, 
optionally, 'mu_SEM_to_DEM'.  Each key must return a 3d numpy array with 
indices [i,j,k], where i and j are indices of the eigenvalues of the system 
hamiltonian, and k is an index 0,1,2 cooresponding to cartesian coordinates 
x,y,z.
"""
    def __init__(self,file_path):
        """Initialize object with
Args:
    file_path (str): file path to folder containing mu.npz
"""
        self.base_path = file_path
        self.load_mu()

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
be stored as a .npz file, and must contain a up to two arrays, each with 
three indices: (upper manifold eigenfunction, lower manifold eigenfunction, 
cartesian coordinate).  Keys: 'GSM_to_SEM' connects the ground state and 
singly excited manifolds, 'SEM_to_DEM' connects the singly and doubly excited
manifolds."""
        file_name = os.path.join(self.base_path,'mu.npz')
        mu = np.load(file_name)
        mu_keys = mu.keys()
        self.mu = {mu_key:mu[mu_key] for mu_key in mu_keys}

    def calculate_boolean_mu(self,overlap_matrix,*,rel_tol=1E-3):
        """Uses Bessel's inequality to find the minimum number of dipole
matrix elements needed to correctly resolve the dipole operator to the
given tolerance.

        Args:
            overlap_matrix (np.ndarray) : 3d-array of dipole matrix elements 
                [i,j,k] where i,j are eigenstates and k is a cartesian 
                coordinate.

            rel_tol (float) : relative tolerance for resolving the dipole
                operator mu.

"""
        dim0, dim1 = overlap_matrix.shape[:2]
        bool_mat = np.zeros((dim0,dim1),dtype=bool)
        
        # Inner product over cartesian coordinates: mu_ij dot mu_ij for each i,j pair
        # where mu_ij is a cartesian vector 
        prob_matrix = np.sum(overlap_matrix**2,axis=(2))

        # Sum over all lower manifold states
        probabilities = np.sum(prob_matrix,axis=1)

        # For each state n in the higher manifold
        for n in range(dim0):
            prob_tot = probabilities[n]
            # All lower states that connect to state n
            prob_list = prob_matrix[n,:]
            # Sort lower states by magnitude of mu_nj dot mu_nj
            prob_sort_ind = prob_list.argsort()[::-1]
            prob_sorted = prob_list[prob_sort_ind]
            prob = 0
            # Bessel's inequality
            for j in range(prob_sorted.size):
                prob += prob_sorted[j]
                if np.abs((prob_tot - prob)/prob_tot) < rel_tol:
                    # If the relative tolerance is attained, break out of loop
                    break
            #Keep only the states needed to satisfy the specified rel_tol
            non_zero_ind = prob_sort_ind[:j+1]

            #Set the states needed as True in a boolean array
            bool_mat[n,non_zero_ind] = True
        return bool_mat

    def save_boolean_mu(self,*,rel_tol = 1E-3):
        """Create and save the boolean masks for the dipole matrices
at the given tolerance. Files created by this function are mu_pruned.npz
and mu_boolean.npz.

        Args:
            rel_tol (float) : relative tolerance for resolving the dipole
                operator mu.  Default value of 0.001 has been found to work 
                well with vibronic systems to give convergence of the 
                Transient Absorption signal of better than 1%.
"""
        file_name_pruned = os.path.join(self.base_path,'mu_pruned.npz')
        file_name_boolean = os.path.join(self.base_path,'mu_boolean.npz')
        mu_GSM_to_SEM_boolean = self.calculate_boolean_mu(self.mu['GSM_to_SEM'],rel_tol=rel_tol)
        mu_GSM_to_SEM_pruned = self.mu['GSM_to_SEM'] * mu_GSM_to_SEM_boolean[:,:,np.newaxis]
        mu_boolean_dict = {'GSM_to_SEM':mu_GSM_to_SEM_boolean}
        mu_pruned_dict = {'GSM_to_SEM':mu_GSM_to_SEM_pruned}

        if 'SEM_to_DEM' in self.mu.keys():
            mu_SEM_to_DEM_boolean = self.calculate_boolean_mu(self.mu['SEM_to_DEM'],rel_tol=rel_tol)
            mu_SEM_to_DEM_pruned = self.mu['SEM_to_DEM'] * mu_SEM_to_DEM_boolean[:,:,np.newaxis]
            mu_boolean_dict['SEM_to_DEM'] = mu_SEM_to_DEM_boolean
            mu_pruned_dict['SEM_to_DEM'] = mu_SEM_to_DEM_pruned

        np.savez(file_name_pruned,**mu_pruned_dict)
        np.savez(file_name_boolean,**mu_boolean_dict)
