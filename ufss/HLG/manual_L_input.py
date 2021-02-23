import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import eigs, eigsh
import itertools
from scipy.linalg import block_diag, eig, expm, eigh
from scipy.sparse import save_npz, load_npz, csr_matrix, csc_matrix
import yaml
import copy
import warnings
import os

class ManualL:
    def __init__(self,L,mu_ket_up,*,output='uf2',savedir=''):
        self.L = L
        self.mu_ket_up = mu_ket_up
        self.output = output
        if savedir=='':
            savedir = os.getcwd()
        self.base_path = os.path.join(savedir,output)
        os.makedirs(self.base_path,exist_ok=True)
        if output == 'uf2':
            self.eigfun(self.L)
            self.save_eigensystem(self.base_path)
            if len(mu_ket_up.shape) == 2:
                self.save_RWA_mu(self.base_path)
            elif len(mu_ket_up.shape) == 3:
                self.save_RWA_mu3D(self.base_path)
        elif output == 'RKE':
            self.save_L(self.base_path)
            self.save_RWA_mu_site_basis(self.base_path)

    def save_L(self,dirname):
        save_npz(os.path.join(dirname,'L.npz'),csr_matrix(self.L))
            

    def eigfun(self,L,*,check_eigenvectors = True,invert = True,populations_only = False):
        eigvals, eigvecs = np.linalg.eig(L)

        eigvals = np.round(eigvals,12)
        sort_indices = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:,sort_indices]
        for i in range(eigvals.size):
            max_index = np.argmax(np.abs(eigvecs[:,i]))
            if np.real(eigvecs[max_index,i]) < 0:
                eigvecs[:,i] *= -1
            if eigvals[i] == 0:
                # eigenvalues of 0 correspond to thermal distributions,
                # which should have unit trace in the Hamiltonian space
                if populations_only:
                    trace_norm = eigvecs[:,i].sum()
                    eigvecs[:,i] = eigvecs[:,i] / trace_norm
                else:
                    shape = int(np.sqrt(eigvals.size))
                    trace_norm = eigvecs[:,i].reshape(shape,shape).trace()
                    eigvecs[:,i] = eigvecs[:,i] / trace_norm

        if invert:
            eigvecs_left = np.linalg.pinv(eigvecs)
        else:
            eigvals_left, eigvecs_left = np.linalg.eig(L.T)

            eigvals_left = np.round(eigvals_left,12)
            sort_indices_left = eigvals_left.argsort()
            eigvals_left.sort()
            eigvecs_left = eigvecs_left[:,sort_indices_left]
            eigvecs_left = eigvecs_left.T
            for i in range(eigvals_left.size):
                    norm = np.dot(eigvecs_left[i,:],eigvecs[:,i])
                    eigvecs_left[i,:] *= 1/norm

        if check_eigenvectors:
            LV = L.dot(eigvecs)
            D = eigvecs_left.dot(LV)
            if np.allclose(D,np.diag(eigvals),rtol=1E-10,atol=1E-10):
                pass
            else:
                warnings.warn('Using eigenvectors to diagonalize Liouvillian does not result in the expected diagonal matrix to tolerance, largest deviation is {}'.format(np.max(np.abs(D - np.diag(eigvals)))))

        self.eigenvalues = eigvals
        self.eigenvectors = {'left':eigvecs_left,'right':eigvecs}

        return eigvals, eigvecs, eigvecs_left

    def save_eigensystem(self,dirname):
        np.savez(os.path.join(dirname,'right_eigenvectors.npz'),all_manifolds = self.eigenvectors['right'])
        np.savez(os.path.join(dirname,'left_eigenvectors.npz'),all_manifolds = self.eigenvectors['left'])
        np.savez(os.path.join(dirname,'eigenvalues.npz'),all_manifolds = self.eigenvalues)

    def mu3D_eigentransform(self,mu):
        evl = self.eigenvectors['left']
        ev = self.eigenvectors['right']
        
        mu_t = np.zeros(mu.shape,dtype='complex')
        for i in range(3):
            mu_t[:,:,i] = np.dot(np.dot(evl,mu[:,:,i]),ev)
        return mu_t

    def mask_mu3D(self,mu):
        mu_mask_tol = 10
        
        mu_mask = np.zeros(mu.shape[:2],dtype='bool')
        mu_abs = np.sqrt(np.sum(np.abs(mu)**2,axis=2))
        mu_mask[:,:] = np.round(mu_abs,mu_mask_tol)[:,:]
        mu_masked = mu * mu_mask[:,:,np.newaxis]

        return mu_mask, mu_masked

    def save_RWA_mu3D(self,dirname,*,mask=True):

        H_size = self.mu_ket_up.shape[0]
        mu_dtype= self.mu_ket_up.dtype
        L_size = H_size**2
        II = np.eye(H_size)
        mu_ket_up = np.zeros((L_size,L_size,3),dtype=mu_dtype)
        mu_ket_down = np.zeros((L_size,L_size,3),dtype=mu_dtype)
        mu_bra_up = np.zeros((L_size,L_size,3),dtype=mu_dtype)
        mu_bra_down = np.zeros((L_size,L_size,3),dtype=mu_dtype)
        
        for i in range(3):
            mu_ket_up[:,:,i] = np.kron(self.mu_ket_up[:,:,i],II.T)
            mu_ket_down[:,:,i] = np.kron(np.conjugate(self.mu_ket_up[:,:,i].T),II.T)
            mu_bra_up[:,:,i] = np.kron(II,np.conjugate(self.mu_ket_up[:,:,i]))
            mu_bra_down[:,:,i] = np.kron(II,self.mu_ket_up[:,:,i].T)

        mu_ket_up_t = self.mu3D_eigentransform(mu_ket_up)
        mu_ket_down_t = self.mu3D_eigentransform(mu_ket_down)
        mu_bra_up_t = self.mu3D_eigentransform(mu_bra_up)
        mu_bra_down_t = self.mu3D_eigentransform(mu_bra_down)

        np.savez(os.path.join(dirname,'mu.npz'),ket_up=mu_ket_up_t,bra_up=mu_bra_up_t,
                 ket_down=mu_ket_down_t,bra_down=mu_bra_down_t)

        if mask:
            ket_up_t_mask, mu_ket_up_t_masked = self.mask_mu3D(mu_ket_up_t)
            ket_down_t_mask, mu_ket_down_t_masked = self.mask_mu3D(mu_ket_down_t)
            bra_up_t_mask, mu_bra_up_t_masked = self.mask_mu3D(mu_bra_up_t)
            bra_down_t_mask, mu_bra_down_t_masked = self.mask_mu3D(mu_bra_down_t)

            np.savez(os.path.join(dirname,'mu_boolean.npz'),ket_up=ket_up_t_mask,bra_up=bra_up_t_mask,
                     ket_down=ket_down_t_mask,bra_down=bra_down_t_mask)
            np.savez(os.path.join(dirname,'mu_pruned.npz'),ket_up=mu_ket_up_t_masked,
                     bra_up=mu_bra_up_t_masked,ket_down=mu_ket_down_t_masked,
                     bra_down=mu_bra_down_t_masked)

    def save_RWA_mu(self,dirname,*,mask=True):
        evl = self.eigenvectors['left']
        ev = self.eigenvectors['right']
        
        II = np.eye(self.mu_ket_up.shape[0])

        mu_ket_up = np.kron(self.mu_ket_up,II.T)
        mu_ket_down = np.kron(np.conjugate(self.mu_ket_up.T),II.T)
        mu_bra_up = np.kron(II,np.conjugate(self.mu_ket_up))
        mu_bra_down = np.kron(II,self.mu_ket_up.T)

        mu_mask_tol = 10
        
        mu_ket_up_t = np.dot(np.dot(evl,mu_ket_up),ev)
        mu_ket_up_3d = np.zeros((mu_ket_up_t.shape[0],mu_ket_up_t.shape[0],3),dtype='complex')
        mu_ket_up_3d[:,:,0] = mu_ket_up_t

        mu_bra_up_t = np.dot(np.dot(evl,mu_bra_up),ev)
        mu_bra_up_3d = np.zeros((mu_bra_up_t.shape[0],mu_bra_up_t.shape[0],3),dtype='complex')
        mu_bra_up_3d[:,:,0] = mu_bra_up_t

        mu_ket_down_t = np.dot(np.dot(evl,mu_ket_down),ev)
        mu_ket_down_3d = np.zeros((mu_ket_down_t.shape[0],mu_ket_down_t.shape[0],3),dtype='complex')
        mu_ket_down_3d[:,:,0] = mu_ket_down_t

        mu_bra_down_t = np.dot(np.dot(evl,mu_bra_down),ev)
        mu_bra_down_3d = np.zeros((mu_bra_down_t.shape[0],mu_bra_down_t.shape[0],3),dtype='complex')
        mu_bra_down_3d[:,:,0] = mu_bra_down_t

        if mask:
            ket_up_mask = np.zeros(mu_ket_up_t.shape,dtype='bool')
            ket_up_mask[:,:] = np.round(mu_ket_up_t,mu_mask_tol)[:,:]
            mu_ket_up_t_masked = mu_ket_up_t * ket_up_mask
            mu_ket_up_3d_masked = np.zeros((mu_ket_up_t.shape[0],mu_ket_up_t.shape[0],3),dtype='complex')
            mu_ket_up_3d_masked[:,:,0] = mu_ket_up_t_masked

            bra_up_mask = np.zeros(mu_bra_up_t.shape,dtype='bool')
            bra_up_mask[:,:] = np.round(mu_bra_up_t,mu_mask_tol)[:,:]
            mu_bra_up_t_masked = mu_bra_up_t * bra_up_mask
            mu_bra_up_3d_masked = np.zeros((mu_ket_up_t.shape[0],mu_ket_up_t.shape[0],3),dtype='complex')
            mu_bra_up_3d_masked[:,:,0] = mu_bra_up_t_masked

            ket_down_mask = np.zeros(mu_ket_down_t.shape,dtype='bool')
            ket_down_mask[:,:] = np.round(mu_ket_down_t,mu_mask_tol)[:,:]
            mu_ket_down_t_masked = mu_ket_down_t * ket_down_mask
            mu_ket_down_3d_masked = np.zeros((mu_ket_down_t.shape[0],mu_ket_down_t.shape[0],3),dtype='complex')
            mu_ket_down_3d_masked[:,:,0] = mu_ket_down_t_masked

            bra_down_mask = np.zeros(mu_bra_down_t.shape,dtype='bool')
            bra_down_mask[:,:] = np.round(mu_bra_down_t,mu_mask_tol)[:,:]
            mu_bra_down_t_masked = mu_bra_down_t * bra_down_mask
            mu_bra_down_3d_masked = np.zeros((mu_ket_down_t.shape[0],mu_ket_down_t.shape[0],3),dtype='complex')
            mu_bra_down_3d_masked[:,:,0] = mu_bra_down_t_masked

            np.savez(os.path.join(dirname,'mu.npz'),ket_up=mu_ket_up_3d,bra_up=mu_bra_up_3d,
                     ket_down=mu_ket_down_3d,bra_down=mu_bra_down_3d)
            # np.savez(os.path.join(dirname,'eigenvalues.npz'),all_manifolds=self.eigenvalues)
            # np.savez(os.path.join(dirname,'right_eigenvectors.npz'),all_manifolds=ev)
            # np.savez(os.path.join(dirname,'left_eigenvectors.npz'),all_manifolds=evl)
            np.savez(os.path.join(dirname,'mu_boolean.npz'),ket_up=ket_up_mask,bra_up=bra_up_mask,
                     ket_down=ket_down_mask,bra_down=bra_down_mask)
            np.savez(os.path.join(dirname,'mu_pruned.npz'),ket_up=mu_ket_up_3d_masked,
                     bra_up=mu_bra_up_3d_masked,ket_down=mu_ket_down_3d_masked,
                     bra_down=mu_bra_down_3d_masked)

        else:
            np.savez(os.path.join(dirname,'mu.npz'),ket_up=mu_ket_up_3d,bra_up=mu_bra_up_3d,
                     ket_down=mu_ket_down_3d,bra_down=mu_bra_down_3d)
            # np.savez(os.path.join(dirname,'eigenvalues.npz'),all_manifolds=self.eigenvalues)
            # np.savez(os.path.join(dirname,'right_eigenvectors.npz'),all_manifolds=ev)
            # np.savez(os.path.join(dirname,'left_eigenvectors.npz'),all_manifolds=evl)

    def save_RWA_mu_site_basis(self,dirname):
        
        II = np.eye(self.mu_ket_up.shape[0])
        mu_ket_up = np.kron(self.mu_ket_up,II.T)
        mu_ket_down = np.kron(self.mu_ket_up.T,II.T)
        mu_bra_up = np.kron(II,self.mu_ket_up)
        mu_bra_down = np.kron(II,self.mu_ket_up.T)

        mu_mask_tol = 10
        
        mu_ket_up_3d = np.zeros((mu_ket_up.shape[0],mu_ket_up.shape[0],3),dtype='complex')
        mu_ket_up_3d[:,:,0] = mu_ket_up

        mu_bra_up_3d = np.zeros((mu_bra_up.shape[0],mu_bra_up.shape[0],3),dtype='complex')
        mu_bra_up_3d[:,:,0] = mu_bra_up

        mu_ket_down_3d = np.zeros((mu_ket_down.shape[0],mu_ket_down.shape[0],3),dtype='complex')
        mu_ket_down_3d[:,:,0] = mu_ket_down

        mu_bra_down_3d = np.zeros((mu_bra_down.shape[0],mu_bra_down.shape[0],3),dtype='complex')
        mu_bra_down_3d[:,:,0] = mu_bra_down

        np.savez(os.path.join(dirname,'mu_site_basis.npz'),ket_up=mu_ket_up_3d,bra_up=mu_bra_up_3d,
                     ket_down=mu_ket_down_3d,bra_down=mu_bra_down_3d)
