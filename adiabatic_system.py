import numpy as np
from abc import ABC, abstractmethod

# Classical Only
class AdiabaticSystem(ABC):

    @abstractmethod
    def __init__(self, beta, sysparam):
        self.beta = beta
        self.sysparam = sysparam

    # All Classical Nuclear Potential Surfaces

    @abstractmethod
    def V0(self, R):
        pass

    @abstractmethod
    def V1(self, R):
        pass

    @abstractmethod
    def K(self, R):
        pass

    def Vg(self, R):
        return 0.5 * (self.V0(R) + self.V1(R)) - 0.5 * np.sqrt((self.V0(R) - self.V1(R))**2 + 4 * self.K(R)**2)
    
    # All Classical Nuclear Forces

    @abstractmethod
    def F0(self, R):
        pass

    @abstractmethod
    def F1(self, R):
        pass

    @abstractmethod
    def FK(self, R):
        pass
    
    def Fg(self, R):
        return 0.5 * (self.F0(R) + self.F1(R)) - ((self.V0(R) - self.V1(R)) * (self.F0(R) - self.F1(R)) + 4 * self.K(R) * self.FK(R)) / (2 * np.sqrt((self.V0(R) - self.V1(R))**2 + 4 * self.K(R)**2))
    
    # Additional Initialization Functions

    @abstractmethod
    def set_dnuclei(self):
        pass

    @abstractmethod
    def set_mR(self):
        pass

    @abstractmethod
    def kinked_pair_R(self):
        pass

##############################################################
# KC-RPMD - NVT momentum sampling and kinetic energy

    def sample_vR(self):
        return np.random.normal(scale = 1. / np.sqrt(self.beta * self.mR))

    def TKC(self, vR):
        return np.sum(0.5 * self.mR * vR**2)
    
##############################################################

class SystemB(AdiabaticSystem):
    def __init__(self, beta, sysparam):
        super().__init__(beta, sysparam)
        self.ms = sysparam[0]
        self.omegas = sysparam[1]
        self.s0 = sysparam[2]
        self.s1 = sysparam[3]
        self.epsilon = sysparam[4]
        self.nbath = int(sysparam[5])

        self.mq= sysparam[6]
        self.omegaq = sysparam[7]
        self.Dq = sysparam[8]
        self.K0= sysparam[9]
        self.bq= sysparam[10]
    
        self.set_dnuclei()
        self.set_mR()

    def set_dnuclei(self):
        if self.nbath == 0:
            self.dnuclei = 2
        else:
            self.dnuclei = 2 + self.nbath
            self.M = self.ms
            self.omegac = self.omegas
            self.gamma = 1.0 * self.M * self.omegac
            self.omegaj = -self.omegac * np.log((np.arange(self.nbath) + 0.5) / self.nbath)
            self.cj = self.omegaj * np.sqrt(2 * self.gamma * self.M * self.omegac / (self.nbath * np.pi))

    def set_mR(self):
        mR = np.ones(self.dnuclei)
        if self.nbath == 0:
            mR[0] = self.ms
            mR[1] = self.mq
        else:
            mR *= self.M
            mR[0] = self.ms
            mR[self.nbath + 1] = self.mq
        self.mR = mR

    def Vq(self, q):
        if (q > 0.):
            return 0.5 * self.mq * self.omegaq**2 * q**2
        else:
            return self.Dq * (1 - np.exp(-np.sqrt(0.5 * self.mq * self.omegaq**2 / self.Dq) * q))**2

    def Fq(self, q):
        if (q > 0.):
            return -self.mq * self.omegaq**2 * q
        else:
            return -2 * self.Dq * np.sqrt(0.5 * self.mq * self.omegaq**2 / self.Dq) * np.exp(-np.sqrt(0.5 * self.mq * self.omegaq**2 / self.Dq) * q) * (1 - np.exp(-np.sqrt(0.5 * self.mq * self.omegaq**2 / self.Dq) * q))

    def V0(self, R):
        if self.nbath == 0:
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s0)**2 + self.Vq(R[1])
        else:
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s0)**2 + np.sum(0.5 * self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M* self.omegaj**2))**2) + self.Vq(R[1 + self.nbath])

    def V1(self, R):
        if self.nbath == 0:    
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s1)**2 + self.epsilon + 0.5 * self.mq * self.omegaq**2 * (R[1])**2
        else:
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s1)**2 + self.epsilon + np.sum(0.5 * self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M* self.omegaj**2))**2) + self.Vq(R[1 + self.nbath])

    def K(self, R):
        if self.nbath == 0:
            return self.K0 * np.exp(-self.bq * R[1])
        else:
            return self.K0 * np.exp(-self.bq * R[1 + self.nbath])

    def F0(self, R):
        F = np.zeros(self.dnuclei)
        if self.nbath == 0:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s0)
            F[1] = self.Fq(R[1])
        else:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s0) + np.sum(self.cj * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2)))
            F[1:1 + self.nbath] = -self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2))
            F[1 + self.nbath] = self.Fq(R[1 + self.nbath])
        return F

    def F1(self, R):
        F = np.zeros(self.dnuclei)
        if self.nbath == 0:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s1)
            F[1] = self.Fq(R[1])
        else:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s1) + np.sum(self.cj * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2)))
            F[1:1 + self.nbath] = -self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2))
            F[1 + self.nbath] = self.Fq(R[1 + self.nbath])
        return F
    
    def FK(self, R):
        F = np.zeros(self.dnuclei)
        if self.nbath == 0:    
            F[1] = self.bq * self.K0 * np.exp(-self.bq * R[1])
        else:
            F[1 + self.nbath] = self.bq * self.K0 * np.exp(-self.bq * R[1 + self.nbath])
        return F

    def kinked_pair_R(self):
        sdagger = 0.5 * (self.s0 + self.s1) - self.epsilon / (self.ms * self.omegas**2 * (self.s0 - self.s1))
        R = np.zeros(self.dnuclei)
        if self.nbath == 0:
            R[0] = sdagger
        else:
            R[0] = sdagger
            R[1:1 + self.nbath] = self.cj * sdagger / (self.M * self.omegaj**2)
        return R

##############################################################

class SystemC(KcrpmdSystem):
    def __init__(self, beta, sysparam):
        super().__init__(beta, sysparam)
        self.ms = sysparam[0]
        self.omegas = sysparam[1]
        self.s0 = sysparam[2]
        self.s1 = sysparam[3]
        self.epsilon = sysparam[4]
        self.nbath = int(sysparam[5])

        self.mq= sysparam[6]
        self.Aq = sysparam[7]
        self.Bq = sysparam[8]
        self.Cq = sysparam[9]
        self.K0= sysparam[10]
        self.bq= sysparam[11]
 
        self.set_dnuclei()
        self.set_mR()

    def set_dnuclei(self):
        if self.nbath == 0:
            self.dnuclei = 2
        else:
            self.dnuclei = 2 + self.nbath
            self.M = self.ms
            self.omegac = self.omegas
            self.gamma = 1.0 * self.M * self.omegac
            self.omegaj = -self.omegac * np.log((np.arange(self.nbath) + 0.5) / self.nbath)
            self.cj = self.omegaj * np.sqrt(2 * self.gamma * self.M * self.omegac / (self.nbath * np.pi))

    def set_mR(self):
        mR = np.ones(self.dnuclei)
        if self.nbath == 0:
            mR[0] = self.ms
            mR[1] = self.mq
        else:
            mR *= self.M
            mR[0] = self.ms
            mR[self.nbath + 1] = self.mq
        self.mR = mR

    def Vq(self, q):
        return self.Aq * q**4 - self.Bq * q**3 + self.Cq * q**2

    def Fq(self, q):
        return -4 * self.Aq * q**3 + 3 * self.Bq * q**2 - 2 * self.Cq * q

    def V0(self, R):
        if self.nbath == 0:
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s0)**2 + self.Vq(R[1])
        else:
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s0)**2 + np.sum(0.5 * self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M* self.omegaj**2))**2) + self.Vq(R[1 + self.nbath])

    def V1(self, R):
        if self.nbath == 0:    
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s1)**2 + self.epsilon + 0.5 * self.mq * self.omegaq**2 * (R[1])**2
        else:
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s1)**2 + self.epsilon + np.sum(0.5 * self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M* self.omegaj**2))**2) + self.Vq(R[1 + self.nbath])

    def K(self, R):
        if self.nbath == 0:
            return self.K0 * np.exp(-self.bq * R[1])
        else:
            return self.K0 * np.exp(-self.bq * R[1 + self.nbath])

    def F0(self, R):
        F = np.zeros(self.dnuclei)
        if self.nbath == 0:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s0)
            F[1] = self.Fq(R[1])
        else:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s0) + np.sum(self.cj * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2)))
            F[1:1 + self.nbath] = -self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2))
            F[1 + self.nbath] = self.Fq(R[1 + self.nbath])
        return F

    def F1(self, R):
        F = np.zeros(self.dnuclei)
        if self.nbath == 0:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s1)
            F[1] = self.Fq(R[1])
        else:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s1) + np.sum(self.cj * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2)))
            F[1:1 + self.nbath] = -self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2))
            F[1 + self.nbath] = self.Fq(R[1 + self.nbath])
        return F
    
    def FK(self, R):
        F = np.zeros(self.dnuclei)
        if self.nbath == 0:    
            F[1] = self.bq * self.K0 * np.exp(-self.bq * R[1])
        else:
            F[1 + self.nbath] = self.bq * self.K0 * np.exp(-self.bq * R[1 + self.nbath])
        return F

    def kinked_pair_R(self):
        sdagger = 0.5 * (self.s0 + self.s1) - self.epsilon / (self.ms * self.omegas**2 * (self.s0 - self.s1))
        R = np.zeros(self.dnuclei)
        if self.nbath == 0:
            R[0] = sdagger
        else:
            R[0] = sdagger
            R[1:1 + self.nbath] = self.cj * sdagger / (self.M * self.omegaj**2)
        return R

