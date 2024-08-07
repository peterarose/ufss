"""
This doesn't work unless there are four pulses. Thus for things like TA, one 
must divide both the pump and probe pulses in two, so that there are 4 effective
pulses. The two copies of the pump should be multiplied by a factor of 2, while 
the two copies of the probe pulse should
"""

#Standard python libraries
import time
import itertools

#Dependencies
import numpy as np

"""The following definitions of kdelvec and kdelvec6 are 
based upon the formulas given in Appendix B of Molecular Quantum 
Electrodynamics, by Akbar Salam

I4_mat and I6_mat are created using the formulae provided in 
"On three‐dimensional rotational averages"
by Andrews, D.L. and Thirunamachandran, T., doi:10.1063/1.434725

Unittest is available to check I4_mat and I6_mat against the matrices that were
explicitly written out in Salam's book.
"""

def kdel(x,y):
    """Kronecker Delta"""
    if x == y:
        return 1
    else:
        return 0

def kdel2(a,b,c,d):
    """Product of 2 Kronecker Deltas"""
    return kdel(a,b)*kdel(c,d)

def kdel3(a,b,c,d,e,f):
    """Product of 2 Kronecker Deltas"""
    return kdel(a,b)*kdel(c,d)*kdel(e,f)

def kdelvec4(i,j,k,l):
    """Length 3 vector of Kronecker Delta products, as defined in """
    vec = [kdel2(i,j,k,l),
           kdel2(i,k,j,l),
           kdel2(i,l,j,k)]
    return np.array(vec)

def kdelvec6(i,j,k,l,m,n):
    """Length 3 vector of Kronecker Delta products, as defined in """
    vec = [kdel3(i,j,k,l,m,n),
           kdel3(i,j,k,m,n,l),
           kdel3(i,j,k,n,l,m),
           kdel3(i,k,j,l,m,n),
           kdel3(i,k,j,m,n,l),
           kdel3(i,k,j,n,l,m),
           kdel3(i,l,j,k,m,n),
           kdel3(i,l,j,m,k,n),
           kdel3(i,l,j,n,k,m),
           kdel3(i,m,j,k,n,l),
           kdel3(i,m,j,l,k,n),
           kdel3(i,m,j,n,k,l),
           kdel3(i,n,j,k,l,m),
           kdel3(i,n,j,l,k,m),
           kdel3(i,n,j,m,k,l)]
    return np.array(vec)

def kdelvec(indices):
    if len(indices) == 4:
        return kdelvec4(*indices)
    elif len(indices) == 6:
        return kdelvec6(*indices)
    else:
        raise Exception('Isotropic averaging is only implemented for rank 4 and rank 6 tensors')

S4 = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                vec1 = kdelvec4(i,j,k,l)
                vec2 = kdelvec4(i,j,k,l)
                S4 += np.outer(vec1,vec2)
I4_mat = np.linalg.inv(S4)

S6 = np.zeros((15,15))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                for m in range(3):
                    for n in range(3):
                        vec1 = kdelvec6(i,j,k,l,m,n)
                        vec2 = kdelvec6(i,j,k,l,m,n)
                        S6 += np.outer(vec1,vec2)
I6_mat = np.linalg.inv(S6)

def get_In_mat(indices):
    if len(indices) == 4:
        return I4_mat
    elif len(indices) == 6:
        return I6_mat
    else:
        raise Exception('Isotropic averaging is only implemented for rank 4 and rank 6 tensors')

class IsotropicAverage(object):
    """This class performs the isotropic average of the 4th order tensor
        which is the material response produced by 4-wave mixing process"""

    def __init__(self,spectra_calculator,*,diagrams='all'):
        """Takes as input a ufss object that calculates 4-wave mixing spectra,
        and calculates the isotropically averaged signal, given a lab-frame polarization"""
        self.sc = spectra_calculator
        self.diagrams = diagrams
        self.signal_dict = {}

    def molecular_frame_signal(self,mol_polarization):
        self.sc.set_polarization_sequence(mol_polarization)
        if self.diagrams == 'all':
            signal = self.sc.calculate_signal_all_delays(composite_diagrams=False)
        else:
            signal = self.sc.calculate_diagrams_all_delays(self.diagrams)
        return signal
    
    def get_polarization_options(self):
        """Determines which polarizations are necessary, given the cartesian 
        structure of the dipole operator"""
        xyz = ['x','y','z']

        pol_options = []
        for i in range(3):
            # Check to see if the dipole operator has any non-zero components along the given
            # molecular frame axis, if the dipole exists only in the x-y plane, for example,
            # then we can avoid doing quite a few unnecessary calculations!
            try:
                random_mu_key = list(self.sc.mu.keys())[0]
                test_mu = self.sc.mu[random_mu_key]
            except AttributeError:
                random_mu_key = list(self.sc.H_mu.keys())[0]
                test_mu = self.sc.H_mu[random_mu_key]
            if type(test_mu) is np.ndarray:
                if not np.allclose(test_mu[:,:,i],0):
                    pol_options.append(xyz[i])
            elif type(test_mu) is list:
                if not np.allclose(test_mu[i][:,:],0):
                    pol_options.append(xyz[i])

        return pol_options
    
    def get_signal_key(self,polarization_sequence):
        return ''.join(polarization_sequence)

    def averaged_signal(self,lab_polarizations,*,return_signal=False,reset=False):
        t0 = time.time()
        self.pol = lab_polarizations
        if reset:
            self.signal_dict = {}

        left_vec = kdelvec(self.pol)

        num_pol = len(self.pol)

        pol_options = self.get_polarization_options()

        pol_options_N = [pol_options]*num_pol

        pol_options_iter = itertools.product(*pol_options_N)

        In_mat = get_In_mat(self.pol)

        new_signal = True
        for pol in pol_options_iter:
            # generate the vector of kronecker delta products
            right_vec = kdelvec(pol)
            if np.allclose(right_vec,0):
                # If the vector is 0, don't bother!
                pass
            else:
                # If not, set the polarization sequence, do the calculation, and
                # add the weight given by the isotropic weight matrix, In_mat
                # Note the the polarization sequences are not the lab frame
                # polarization sequence of the pulses.
                key = self.get_signal_key(pol)
                try:
                    mol_frame_signal = self.signal_dict[key]
                except KeyError:
                    mol_frame_signal = self.molecular_frame_signal(pol)
                    self.signal_dict[key] = mol_frame_signal
                weight = In_mat.dot(right_vec)
                weight = np.dot(left_vec,weight)
                if new_signal:
                    signal = weight * mol_frame_signal
                    new_signal = False
                else:
                    signal += weight * mol_frame_signal

        self.signal = signal

        self.calculation_time = time.time() - t0
        if return_signal:
            return signal

    def save(self,save_name,pulse_delay_names=[],*,use_base_path=True):
        if save_name == 'auto':
            save_name = 'IsotropicAverage_'+''.join(self.pol)
        self.sc.signal = self.signal
        self.sc.calculation_time = self.calculation_time
        self.sc.save(save_name,pulse_delay_names,use_base_path=use_base_path)

class FWMIsotropicAverage(IsotropicAverage):
    """For backwards compatibility"""

    def __init__(self,spectra_calculator,lab_polarization,*,diagrams='all'):
        """Takes as input a ufss object that calculates 4-wave mixing spectra,
        and calculates the isotropically averaged signal, given a lab-frame polarization"""
        self.sc = spectra_calculator
        self.pol = lab_polarization
        self.diagrams = diagrams
        self.signal_dict = {}

    def averaged_signal(self,*,return_signal=False):
        sig = super().averaged_signal(self.pol,return_signal=return_signal)
        if return_signal:
            return sig

    def save(self,save_name,pulse_delay_names=[],*,use_base_path=True):
        if save_name == 'auto':
            save_name = 'FWM_IsotropicAverage_'+''.join(self.pol)
        super().save(save_name,pulse_delay_names=pulse_delay_names,
                     use_bath_path=use_base_path)   