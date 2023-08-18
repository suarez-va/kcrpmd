import numpy as np
from abc import ABC, abstractmethod

# Classical Only
class KcrpmdSystem(ABC):

    @abstractmethod
    def __init__(self, beta, a0, b, c, eta, my, sysparam):
        self.beta = beta
        self.a0 = a0
        self.b = b
        self.c = c
        self.eta = eta
        self.my = my
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
    
    def Ve(self, R):
        return 0.5 * (self.V0(R) + self.V1(R)) + 0.5 * np.sqrt((self.V0(R) - self.V1(R))**2 + 4 * self.K(R)**2)

    def VMF(self, R):
        return self.Vg(R) - 1 / self.beta * np.log(1 + np.exp(self.beta * (self.Vg(R) - self.Ve(R))))
    
    def VKP(self, R):
        return self.VMF(R) - 1 / self.beta * np.log(1 - np.exp(self.beta * (self.VMF(R) - self.V0(R))) - np.exp(self.beta * (self.VMF(R) - self.V1(R))))
    
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
    
    def Fe(self, R):
        return 0.5 * (self.F0(R) + self.F1(R)) + ((self.V0(R) - self.V1(R)) * (self.F0(R) - self.F1(R)) + 4 * self.K(R) * self.FK(R)) / (2 * np.sqrt((self.V0(R) - self.V1(R))**2 + 4 * self.K(R)**2))

    def FMF(self, R):
        return self.Fg(R) - 0.5 * (self.Fg(R) - self.Fe(R)) * (1 + np.tanh(0.5 * self.beta * (self.Vg(R) - self.Ve(R))))
    
    def FKP(self, R):
        return (self.FMF(R) - self.F0(R) * np.exp(self.beta * (self.VMF(R) - self.V0(R))) - self.F1(R) * np.exp(self.beta * (self.VMF(R) - self.V1(R)))) / (1 - np.exp(self.beta * (self.VMF(R) - self.V0(R))) - np.exp(self.beta * (self.VMF(R) - self.V1(R))))

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

    def sample_vy(self):
        return np.random.normal(scale = 1. / np.sqrt(self.beta * self.my))    

    def sample_vR(self):
        return np.random.normal(scale = 1. / np.sqrt(self.beta * self.mR))

    def TKC(self, vy, vR):
        return 0.5 * self.my * vy**2 + np.sum(0.5 * self.mR * vR**2)
    
##############################################################
# KC-RPMD - potential energy functions

    def Vr(self, y, theta):
        if abs(y - theta) < 0.5:
            return -1 / self.beta * np.log(1 / (1 + np.exp(self.b * (2 * abs(y - theta) - 1))))
        else:
            return -1 / self.beta * np.log(1 / (1 + np.exp(-self.b * (2 * abs(y - theta) - 1)))) + self.b * (2 * abs(y - theta) - 1) / self.beta
        
    def w(self, R):
        return (self.V0(R) - self.V1(R)) / self.K(R)
        
    def a(self, R):
        return self.a0 / 2 * (1 + np.tanh(-self.c * (self.beta * abs(self.K(R)) - 1)))
    
    def C(self, R):
        return self.eta * np.sqrt(self.a(R) / np.pi) + (1 - self.eta * np.sqrt(self.a(R) / np.pi)) * 0.5 * (1 + np.tanh(self.c * (self.beta * abs(self.K(R)) - 1)))

    def VKC(self, y, R):
        VKP_arg = -self.beta * self.Vr(y, 0) + np.log(self.C(R)) - self.a(R) * self.w(R)**2 - self.beta * self.VKP(R)
        V0_arg = -self.beta * self.Vr(y, -1) - self.beta * self.V0(R)
        V1_arg = -self.beta * self.Vr(y, 1) - self.beta * self.V1(R)
        exp_shift = np.maximum(np.maximum(V0_arg, V1_arg), VKP_arg)
        return -1 / self.beta * np.log(np.exp(VKP_arg - exp_shift) + np.exp(V0_arg - exp_shift) + np.exp(V1_arg - exp_shift)) - exp_shift / self.beta

##############################################################
# KC-RPMD - force functions

    def Fr(self, y, theta):
        return -self.b / self.beta * np.sign(y - theta) * (1 + np.tanh(self.b * (abs(y - theta) - 0.5)))

    def dw(self, R):
        return -1 / self.K(R) * (self.F0(R) - self.F1(R) - self.w(R) * self.FK(R))

    def da(self, R):
        hyperbolic_arg = self.c * (self.beta * abs(self.K(R)) - 1)
        if abs(hyperbolic_arg) > 250.:
            return 0.
        else:
            return 0.5 * self.a0 * self.c * self.beta * np.sign(self.K(R)) / (np.cosh(hyperbolic_arg)**2) * self.FK(R)
        
    def dC(self, R):
        return 1 / self.a0 * (1.5 * self.eta * np.sqrt(self.a(R) / np.pi) - 1) * self.da(R)

    def FKC_y(self, y, R):
        VKP_arg = -self.beta * self.Vr(y, 0) + np.log(self.C(R)) - self.a(R) * self.w(R)**2 - self.beta * self.VKP(R)
        V0_arg = -self.beta * self.Vr(y, -1) - self.beta * self.V0(R)
        V1_arg = -self.beta * self.Vr(y, 1) - self.beta * self.V1(R)
        exp_shift = np.maximum(np.maximum(V0_arg, V1_arg), VKP_arg)
        return (self.Fr(y, 0) * np.exp(VKP_arg - exp_shift) + self.Fr(y, -1) * np.exp(V0_arg - exp_shift) + self.Fr(y, 1) * np.exp(V1_arg - exp_shift)) / (np.exp(VKP_arg - exp_shift) + np.exp(V0_arg - exp_shift) + np.exp(V1_arg - exp_shift))

    def FKC_R(self, y, R):
        VKP_arg = -self.beta * self.Vr(y, 0) + np.log(self.C(R)) - self.a(R) * self.w(R)**2 - self.beta * self.VKP(R)
        V0_arg = -self.beta * self.Vr(y, -1) - self.beta * self.V0(R)
        V1_arg = -self.beta * self.Vr(y, 1) - self.beta * self.V1(R)
        exp_shift = np.maximum(np.maximum(V0_arg, V1_arg), VKP_arg)
        #(1 / (self.beta * self.C(R)) * self.dC(R) - self.w(R)**2 / self.beta * self.da(R) - 2 * self.a(R) * self.w(R) / self.beta * self.dw(R) + self.FKP(R))
        return ((1 / (self.beta * self.C(R)) * self.dC(R) - self.w(R)**2 / self.beta * self.da(R) - 2 * self.a(R) * self.w(R) / self.beta * self.dw(R) + self.FKP(R)) * np.exp(VKP_arg - exp_shift) + self.F0(R) * np.exp(V0_arg - exp_shift) + self.F1(R) * np.exp(V1_arg - exp_shift)) / (np.exp(VKP_arg - exp_shift) + np.exp(V0_arg - exp_shift) + np.exp(V1_arg - exp_shift))


##############################################################

class SystemB(KcrpmdSystem):
    def __init__(self, beta, a0, b, c, eta, my, sysparam):
        super().__init__(beta, a0, b, c, eta, my, sysparam)
        self.ms = sysparam[0]
        self.omegas = sysparam[1]
        self.s0 = sysparam[2]
        self.s1 = sysparam[3]
        self.epsilon = sysparam[4]
        self.nbath = int(sysparam[5])

        self.mq= sysparam[6]
        self.omegaq = sysparam[7]
        self.K0= sysparam[8]
        self.bq= sysparam[9]
    
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
            self.cj = self.omegaj * np.sqrt(2 * self.gamma * self.M* self.omegac / (self.nbath * np.pi))

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

    def V0(self, R):
        if self.nbath == 0:
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s0)**2 + 0.5 * self.mq * self.omegaq**2 * (R[1])**2
        else:
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s0)**2 + np.sum(0.5 * self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M* self.omegaj**2))**2) + 0.5 * self.mq * self.omegaq**2 * (R[1 + self.nbath])**2

    def V1(self, R):
        if self.nbath == 0:    
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s1)**2 + self.epsilon + 0.5 * self.mq * self.omegaq**2 * (R[1])**2
        else:
            return 0.5 * self.ms * self.omegas**2 * (R[0] - self.s1)**2 + self.epsilon + np.sum(0.5 * self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M* self.omegaj**2))**2) + 0.5 * self.mq * self.omegaq**2 * (R[1 + self.nbath])**2

    def K(self, R):
        if self.nbath == 0:
            return self.K0 * np.exp(-self.bq * R[1])
        else:
            return self.K0 * np.exp(-self.bq * R[1 + self.nbath])

    def F0(self, R):
        F = np.zeros(self.dnuclei)
        if self.nbath == 0:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s0)
            F[1] = -self.mq * self.omegaq**2 * R[1]
        else:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s0) + np.sum(self.cj * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2)))
            F[1:1 + self.nbath] = -self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2))
            F[1 + self.nbath] = -self.mq * self.omegaq**2 * R[1 + self.nbath]
        return F
    

    def F1(self, R):
        F = np.zeros(self.dnuclei)
        if self.nbath == 0:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s1)
            F[1] = -self.mq * self.omegaq**2 * R[1]
        else:
            F[0] = -self.ms * self.omegas**2 * (R[0] - self.s1) + np.sum(self.cj * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2)))
            F[1:1 + self.nbath] = -self.M * self.omegaj**2 * (R[1:1 + self.nbath] - self.cj * R[0] / (self.M * self.omegaj**2))
            F[1 + self.nbath] = -self.mq * self.omegaq**2 * R[1 + self.nbath]
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


