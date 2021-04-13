import numpy as np
import scipy.sparse as sp

class LindbladConstructor:

    @staticmethod
    def make_Lindblad_instructions(gamma,O):
        """O must be square
"""
        II = np.eye(O.shape[0])
        Od = np.conjugate(O.T)
        leftright = gamma * (-np.dot(Od,O)/2)
        return [(gamma*O,Od),(leftright,II),(II,leftright)]

    @staticmethod
    def make_Lindblad_instructions2(gamma,Oket,Obra):
        IIket = np.eye(Oket.shape[0])
        IIbra = np.eye(Obra.shape[0])
        Oketd = np.conjugate(Oket.T)
        Obrad = np.conjugate(Obra.T)
        left = gamma * (-np.dot(Oketd,Oket)/2)
        right = gamma * (-np.dot(Obrad,Obra)/2)
        return [(gamma*Oket,Obrad),(left,IIbra),(IIket,right)]

    @staticmethod
    def make_Lindblad_instructions2_Obra0(gamma,Oket,Obra):
        IIbra = np.eye(Obra.shape[0])
        Oketd = np.conjugate(Oket.T)
        left = gamma * (-np.dot(Oketd,Oket)/2)
        return [(left,IIbra)]

    @staticmethod
    def make_Lindblad_instructions2_Oket0(gamma,Oket,Obra):
        IIket = np.eye(Oket.shape[0])
        Obrad = np.conjugate(Obra.T)
        right = gamma * (-np.dot(Obrad,Obra)/2)
        return [(IIket,right)]

class LiouvillianConstructor(LindbladConstructor):

    @staticmethod
    def make_commutator_instructions(O):
        """O must be square
"""
        II = np.eye(O.shape[0])
        return [(O,II),(II,-O)]

    @staticmethod
    def make_commutator_instructions2(Oket,Obra):
        """
"""
        IIket = np.eye(Oket.shape[0])
        IIbra = np.eye(Obra.shape[0])
        return [(Oket,IIbra),(IIket,-Obra)]

    @staticmethod
    def make_Liouvillian(instruction_list,*,sparse=False):
        if sparse:
            kron = sp.kron
        else:
            kron = np.kron
        left, right = instruction_list[0]
        L = kron(left,right.T)
        for left,right in instruction_list[1:]:
            L = L + kron(left,right.T)
        return L
