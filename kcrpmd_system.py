import numpy as np
from abc import ABC, abstractmethod

# Classical + Quantum Nuclear Build
class KcrpmdSystem(ABC):

    @abstractmethod
    def __init__(self, nbeads, beta, a, b, L, eta, my, sysparam):
        self.set_nbeads(nbeads)
        self.beta = beta
        self.a = a
        self.b = b
        self.L = L
        self.eta = eta
        self.my = my
        self.sysparam = sysparam

    def set_nbeads(self, nbeads):
        self.classical_limit = False
        self.nbeads = nbeads
        if self.nbeads == 1:
            self.classical_limit = True

    def set_dnuclei(self, dnuclei):
        self.dnuclei = dnuclei

    def set_Vref(self, Vref):
        self.Vref = Vref

    @abstractmethod
    def set_mR(self):
        pass

    @abstractmethod
    def V0(self, R):
        pass

    @abstractmethod
    def V1(self, R):
        pass

    @abstractmethod
    def K(self, R):
        pass

    @abstractmethod
    def F0(self, R):
        pass

    @abstractmethod
    def F1(self, R):
        pass

    @abstractmethod
    def FK(self, R):
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

    def TKCRPMD(self, vy, vR):
        return 0.5 * self.my * vy**2 + np.sum(0.5 * self.mR * vR**2)
    
##############################################################
# KC-RPMD - potential energy functions

    def Uint(self, R):
        if self.classical_limit:
            return 0.
        else:
            return np.sum(1 / 2 * self.mR * (self.nbeads / self.beta)**2 * (R - np.concatenate((R[:,1:], R[:,:1]), axis=1))**2)

    def w(self, R):
        if self.classical_limit:
            w = (self.V0(R) - self.V1(R)) / self.K(R)
        else:
            Rbar = np.average(R, axis=1)
            w = (self.V0(Rbar) - self.V1(Rbar)) / self.K(Rbar)
        return w

    def f(self, y, theta):
        if theta == 0:
            ltheta = self.L
        else:
            ltheta = 2 - self.L
        hyperbolic_arg = self.b * (abs(y - theta) - ltheta / 2)
        if hyperbolic_arg < 0.:
            return 1 / (ltheta * (1 + np.exp(2 * hyperbolic_arg)))
        else:
            return np.exp(-2 * hyperbolic_arg) / (ltheta * (1 + np.exp(-2 * hyperbolic_arg)))

    def M(self, R):
        return np.array([[np.exp(-self.beta / self.nbeads * self.V0(R)), -self.beta / self.nbeads * self.K(R) * np.exp(-self.beta / self.nbeads * self.V0(R))], [-self.beta / self.nbeads * self.K(R) * np.exp(-self.beta / self.nbeads * self.V1(R)), np.exp(-self.beta / self.nbeads * self.V1(R))]])

    def Gamma(self, R):
        if self.classical_limit:
            Gamma = 1 / 2 * (self.V0(R) + self.V1(R)) - 1 / self.beta * np.log(2 * np.cosh(self.beta / 2 * np.sqrt((self.V0(R) - self.V1(R))**2 + 4 * self.K(R)**2)))
        else:
            PiM = self.M(R[:,0])
            for i in range(1, self.nbeads):
                PiM = np.matmul(PiM, self.M(R[:,i]))
            Gamma = -1 / self.beta * np.log(np.trace(PiM))
        return Gamma

    def U0(self, R):
        if self.classical_limit:
            U0 = self.V0(R)
        else:
            U0 = np.sum(1 / self.nbeads * np.array([self.V0(R[:, i]) for i in range(self.nbeads)]))
        return U0

    def U1(self, R):
        if self.classical_limit:
            U1 = self.V1(R)
        else:
            U1 = np.sum(1 / self.nbeads * np.array([self.V1(R[:, i]) for i in range(self.nbeads)]))
        return U1

    def UKP(self, R):
        return -1 / self.beta * np.log(self.eta * np.sqrt(self.a / np.pi) * (np.exp(-self.beta * self.Gamma(R)) - np.exp(-self.beta * self.U0(R)) - np.exp(-self.beta * self.U1(R)))) + self.a / self.beta * self.w(R)**2

    def rhoKC(self, y, R):
        return self.f(y, 0) * np.exp(-self.beta * self.UKP(R)) + self.f(y, -1) * np.exp(-self.beta * self.U0(R)) + self.f(y, 1) * np.exp(-self.beta * self.U1(R))
    
    def VKCRPMD(self, y, R):
        return -self.Vref + self.Uint(R) -1 / self.beta * np.log(self.f(y, 0) * np.exp(-self.beta * self.UKP(R)) + self.f(y, -1) * np.exp(-self.beta * self.U0(R)) + self.f(y, 1) * np.exp(-self.beta * self.U1(R)))

##############################################################
# KC-RPMD - force functions

    def dUint(self, R):
        if self.classical_limit:
            return 0.
        else:
            return self.mR * (self.nbeads / self.beta)**2 * (2 * R - np.concatenate((R[:,-1:], R[:,:-1]), axis=1) - np.concatenate((R[:,1:], R[:,:1]), axis=1))

    def dw(self, R):
        if self.classical_limit:
            dw = -1 / self.K(R) * (self.F0(R) - self.F1(R) - self.w(R) * self.FK(R))
        else:
            Rbar = np.average(R, axis=1)
            dw = np.tile(-1 / (self.nbeads * self.K(Rbar)) * (self.F0(Rbar) - self.F1(Rbar) - self.w(R) * self.FK(Rbar)), (self.nbeads, 1)).T
        return dw

    def df(self, y, theta):
        if theta == 0:
            ltheta = self.L
        else:
            ltheta = 2. - self.L 
        hyperbolic_arg = self.b * (abs(y - theta) - ltheta / 2)
        if abs(hyperbolic_arg) > 250.:
            return 0.
        elif y > theta:
            return -self.b / (2 * ltheta * np.cosh(hyperbolic_arg)**2)
        else:
            return self.b / (2 * ltheta * np.cosh(hyperbolic_arg)**2)

    def dM(self, R):
        return np.array([[self.beta / self.nbeads * self.F0(R) * np.exp(-self.beta / self.nbeads * self.V0(R)), -self.beta / self.nbeads * (self.beta / self.nbeads * self.K(R) * self.F0(R) - self.FK(R)) * np.exp(-self.beta / self.nbeads * self.V0(R))], [-self.beta / self.nbeads * (self.beta / self.nbeads * self.K(R) * self.F1(R) - self.FK(R)) * np.exp(-self.beta / self.nbeads * self.V1(R)), self.beta / self.nbeads * self.F1(R) * np.exp(-self.beta / self.nbeads * self.V1(R))]])
    
    def dGamma(self, R):
        if self.classical_limit:
            dGamma = -1 / 2 * (self.F0(R) + self.F1(R)) + (((self.V0(R) - self.V1(R)) * (self.F0(R) - self.F1(R)) + 4 * self.K(R) * self.FK(R)) / (2 * np.sqrt((self.V0(R) - self.V1(R))**2 + 4 * self.K(R)**2))) * np.tanh(self.beta / 2 * np.sqrt((self.V0(R) - self.V1(R))**2 + 4 * self.K(R)**2))
        else:
            Fbell = np.zeros((self.nbeads, 2, 2)); Fbell[0] = np.eye(2)
            Gbell = np.zeros((self.nbeads, 2, 2)); Gbell[-1] = np.eye(2)
            for i in range(1, self.nbeads):
                Fbell[i] = np.matmul(Fbell[i - 1], self.M(R[:, i - 1]))
                Gbell[self.nbeads - 1 - i] = np.matmul(self.M(R[:, self.nbeads - i]), Gbell[self.nbeads - i])
            dGamma = np.zeros((self.dnuclei, self.nbeads))
            PiM = np.matmul(self.M(R[:, 0]), Gbell[0])
            for i in range(self.nbeads):
                dGamma[:,i] = -1 / self.beta * np.trace(np.einsum('ijk,jl->ilk', self.dM(R[:, i]), np.matmul(Gbell[i], Fbell[i]))) / np.trace(PiM)
        return dGamma

    def dU0(self, R):
        if self.classical_limit:
            dU0 = -self.F0(R)
        else:
            dU0 = -1 / self.nbeads * np.array([self.F0(R[:, i]) for i in range(self.nbeads)]).T
        return dU0
    
    def dU1(self, R):
        if self.classical_limit:
            dU1 = -self.F1(R)
        else:
            dU1 = -1 / self.nbeads * np.array([self.F1(R[:, i]) for i in range(self.nbeads)]).T
        return dU1

    def dUKP(self, R):
        return (self.dGamma(R) * np.exp(-self.beta * self.Gamma(R)) - self.dU0(R) * np.exp(-self.beta * self.U0(R)) - self.dU1(R) * np.exp(-self.beta * self.U1(R))) / (np.exp(-self.beta * self.Gamma(R)) - np.exp(-self.beta * self.U0(R)) - np.exp(-self.beta * self.U1(R))) + 2 * self.a / self.beta * self.w(R) * self.dw(R)

    def FKCRPMD_y(self, y, R):
        return (self.df(y, 0) * np.exp(-self.beta * self.UKP(R)) + self.df(y, -1) * np.exp(-self.beta * self.U0(R)) + self.df(y, 1) * np.exp(-self.beta * self.U1(R))) / (self.beta * (self.f(y, 0) * np.exp(-self.beta * self.UKP(R)) + self.f(y, -1) * np.exp(-self.beta * self.U0(R)) + self.f(y, 1) * np.exp(-self.beta * self.U1(R))))

    def FKCRPMD_R(self, y, R):
        return -self.dUint(R) - (self.f(y, 0) * self.dUKP(R) * np.exp(-self.beta * self.UKP(R)) + self.f(y, -1) * self.dU0(R) * np.exp(-self.beta * self.U0(R)) + self.f(y, 1) * self.dU1(R) * np.exp(-self.beta * self.U1(R))) / (self.f(y, 0) * np.exp(-self.beta * self.UKP(R)) + self.f(y, -1) * np.exp(-self.beta * self.U0(R)) + self.f(y, 1) * np.exp(-self.beta * self.U1(R)))

##############################################################

class SystemA(KcrpmdSystem):
    def __init__(self, nbeads, beta, a, b, L, eta, my, sysparam):
        super().__init__(nbeads, beta, a, b, L, eta, my, sysparam)

        self.ms = sysparam[0]
        self.omegas = sysparam[1]
        self.s0 = sysparam[2]
        self.s1 = sysparam[3]
        self.epsilon = sysparam[4]
        self.Delta = sysparam[5]
        self.mx= sysparam[6]
        self.omegac = sysparam[7]
        self.gamma = sysparam[8]
        self.fmodes = int(sysparam[9])
        
        self.set_dnuclei(1 + self.fmodes)
        self.set_mR()
        
        if self.fmodes > 0:
            self.omegaj = -self.omegac * np.log((np.arange(self.fmodes) + 0.5) / self.fmodes)
            self.cj = self.omegaj * np.sqrt(2 * self.gamma * self.mx* self.omegac / (self.fmodes * np.pi))
    
    
    def set_mR(self):
        mR = np.ones(self.dnuclei); mR *= self.mx / self.nbeads; mR[-1] = self.ms / self.nbeads
        if self.classical_limit:
            self.mR = mR
        else:
            self.mR = np.tile(mR, (self.nbeads, 1)).T

    def V0(self, R):
        if self.fmodes > 0:
            return self.Vref + 0.5 * self.ms * self.omegas**2 * (R[-1] - self.s0)**2 + np.sum(0.5 * self.mx* self.omegaj**2 * (R[:-1] - self.cj * R[-1] / (self.mx* self.omegaj**2))**2)
        else:
            return self.Vref + 0.5 * self.ms * self.omegas**2 * (R[-1] - self.s0)**2

    def V1(self, R):
        if self.fmodes > 0:
            return self.Vref + 0.5 * self.ms * self.omegas**2 * (R[-1] - self.s1)**2 + self.epsilon + np.sum(0.5 * self.mx* self.omegaj**2 * (R[:-1] - self.cj * R[-1] / (self.mx* self.omegaj**2))**2)
        else:
            return self.Vref + 0.5 * self.ms * self.omegas**2 * (R[-1] - self.s1)**2 + self.epsilon

    def K(self, R):
        return self.Delta

    def F0(self, R):
        F = np.zeros(self.dnuclei)
        if self.fmodes > 0:
            F[:-1] = -self.mx* self.omegaj**2 * (R[:-1] - self.cj * R[-1] / (self.mx* self.omegaj**2))
            F[-1] = -self.ms * self.omegas**2 * (R[-1] - self.s0) + np.sum(self.cj * (R[:-1] - self.cj * R[-1] / (self.mx* self.omegaj**2)))
        else:
            F[-1] = -self.ms * self.omegas**2 * (R[-1] - self.s0)
        return F

    def F1(self, R):
        F = np.zeros(self.dnuclei)
        if self.fmodes > 0:
            F[:-1] = -self.mx* self.omegaj**2 * (R[:-1] - self.cj * R[-1] / (self.mx* self.omegaj**2))
            F[-1] = -self.ms * self.omegas**2 * (R[-1] - self.s1) + np.sum(self.cj * (R[:-1] - self.cj * R[-1] / (self.mx* self.omegaj**2)))
        else:
            F[-1] = -self.ms * self.omegas**2 * (R[-1] - self.s1)
        return F

    def FK(self, R):
        return np.zeros(self.dnuclei)

    def kinked_pair_R(self):
        sdagger = 0.5 * (self.s0 + self.s1) - self.epsilon / (self.ms * self.omegas**2 * (self.s0 - self.s1))
        R = np.zeros(self.dnuclei)
        R[-1] = sdagger
        if self.fmodes > 0:
            R[:-1] = self.cj * sdagger / (self.mx* self.omegaj**2)
        if self.classical_limit:
            return R
        else:
            return np.tile(R, (self.nbeads, 1)).T
