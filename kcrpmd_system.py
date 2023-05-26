import numpy as np
from abc import ABC, abstractmethod
from scipy.integrate import quad, dblquad

# Classical Lmimit Build
class KcrpmdSystem(ABC):

    @abstractmethod
    def __init__(self, beta, a, b, L, eta, my, sysparam):
        self.beta = beta
        self.a = a
        self.b = b
        self.L = L
        self.eta = eta
        self.my = my
        self.sysparam = sysparam

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

    @abstractmethod
    def TST_Rate(self):
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

    def f(self, y, theta):
        if theta == -1 or theta == 1:
            ltheta = 2. - self.L
        elif theta == 0:
            ltheta = self.L
        else:
            print("ERROR: theta should be -1, 0, or 1 !!! >:( ")
            exit()
        exp_arg = self.b * (2 * abs(y - theta) - ltheta)
        if exp_arg > 0.:
            return np.exp(-exp_arg) / (ltheta * (1 + np.exp(-exp_arg)))
        else:
            return 1 / (ltheta * (1 + np.exp(exp_arg)))

    def w(self, R):
        return (self.V0(R) - self.V1(R)) / self.K(R)

    def g(self, R):
        return np.sqrt(self.a / np.pi) * self.eta * np.exp(-self.a * self.w(R)**2)

    def phi(self, R):
        return 2 * np.cosh(self.beta / 2 * self.K(R) * np.sqrt(self.w(R)**2 + 4)) * np.exp(-self.beta / 2 * (self.V0(R) + self.V1(R))) - np.exp(-self.beta * self.V0(R)) - np.exp(-self.beta * self.V1(R))

    def VKC(self, y, R):
        return -1 / self.beta * np.log(self.f(y, 0) * self.g(R) * self.phi(R) + self.f(y, -1) * np.exp(-self.beta * self.V0(R)) + self.f(y, 1) * np.exp(-self.beta * self.V1(R)))

##############################################################
# KC-RPMD - force functions

    def df(self, y, theta):
        if theta == -1 or theta == 1:
            ltheta = 2. - self.L
        elif theta == 0:
            ltheta = self.L
        else:
            print("ERROR: theta should be -1, 0, or 1 !!! >:( ")
            exit() 
        if abs(self.b * (abs(y - theta) - ltheta / 2.)) > 200.:
            return 0.
        elif y > theta:
            return -self.b / (2 * ltheta * np.cosh(self.b * (abs(y - theta) - ltheta / 2.))**2)
        else:
            return self.b / (2 * ltheta * np.cosh(self.b * (abs(y - theta) - ltheta / 2.))**2)

    def dg(self, R):
        return 2 * self.a * self.g(R) * self.w(R) * (self.F0(R) - self.F1(R) - self.w(R) * self.FK(R)) / self.K(R)

    def dphi(self, R):
        return self.beta * ((self.F0(R) + self.F1(R)) * np.cosh(self.beta / 2 * self.K(R) * np.sqrt(self.w(R)**2 + 4)) - (self.w(R) * (self.F0(R) - self.F1(R)) + 4 * self.FK(R)) / np.sqrt(self.w(R)**2 + 4) * np.sinh(self.beta / 2 * self.K(R) * np.sqrt(self.w(R)**2 + 4))) * np.exp(-self.beta / 2 * (self.V0(R) + self.V1(R))) - self.beta * self.F0(R) * np.exp(-self.beta * self.V0(R)) - self.beta * self.F1(R) * np.exp(-self.beta * self.V1(R))

    def FKCy(self, y, R):
        return (self.g(R) * self.phi(R) * self.df(y, 0) + np.exp(-self.beta * self.V0(R)) * self.df(y, -1) + np.exp(-self.beta * self.V1(R)) * self.df(y, 1)) / (self.beta * (self.f(y, 0) * self.g(R) * self.phi(R) + self.f(y, -1) * np.exp(-self.beta * self.V0(R)) + self.f(y, 1) * np.exp(-self.beta * self.V1(R))))

    def FKCR(self, y, R):
        return (self.f(y, 0) * (self.phi(R) * self.dg(R) + self.g(R) * self.dphi(R)) + self.beta * self.f(y, -1) * self.F0(R) * np.exp(-self.beta * self.V0(R)) + self.beta * self.f(y, 1) * self.F1(R) * np.exp(-self.beta * self.V1(R))) / (self.beta * (self.f(y, 0) * self.g(R) * self.phi(R) + self.f(y, -1) * np.exp(-self.beta * self.V0(R)) + self.f(y, 1) * np.exp(-self.beta * self.V1(R))))

##############################################################

class SystemA(KcrpmdSystem):
    def __init__(self, beta, a, b, L, eta, my, sysparam):
        super().__init__(beta, a, b, L, eta, my, sysparam)

        self.ms = sysparam[0]
        self.omegas = sysparam[1]
        self.s0 = sysparam[2]
        self.s1 = sysparam[3]
        self.epsilon = sysparam[4]
        self.Delta = sysparam[5]
        self.M = sysparam[6]
        self.omegac = sysparam[7]
        self.gamma = sysparam[8]
        self.fmodes = int(sysparam[9])

        self.omegaj = -self.omegac * np.log((np.arange(self.fmodes) + 0.5) / self.fmodes)
        self.cj = self.omegaj * np.sqrt(2 * self.gamma * self.M * self.omegac / (self.fmodes * np.pi))
        self.set_mR()

    def set_mR(self):
            mR = np.ones(self.fmodes + 1); mR *= self.M; mR[-1] = self.ms
            self.mR = mR
    
    def V0(self, R):
        return 0.5 * self.ms * self.omegas**2 * (R[-1] - self.s0)**2 + np.sum(0.5 * self.M * self.omegaj**2 * (R[:-1] - self.cj * R[-1] / (self.M * self.omegaj**2))**2)

    def V1(self, R):
        return 0.5 * self.ms * self.omegas**2 * (R[-1] - self.s1)**2 + self.epsilon + np.sum(0.5 * self.M * self.omegaj**2 * (R[:-1] - self.cj * R[-1] / (self.M * self.omegaj**2))**2)

    def K(self, R):
        return self.Delta

    def F0(self, R):
        F = np.zeros(self.fmodes + 1)
        F[:-1] = -self.M * self.omegaj**2 * (R[:-1] - self.cj * R[-1] / (self.M * self.omegaj**2))
        F[-1] = -self.ms * self.omegas**2 * (R[-1] - self.s0) + np.sum(self.cj * (R[:-1] - self.cj * R[-1] / (self.M * self.omegaj**2)))
        return F

    def F1(self, R):
        F = np.zeros(self.fmodes + 1)
        F[:-1] = -self.M * self.omegaj**2 * (R[:-1] - self.cj * R[-1] / (self.M * self.omegaj**2))
        F[-1] = -self.ms * self.omegas**2 * (R[-1] - self.s1) + np.sum(self.cj * (R[:-1] - self.cj * R[-1] / (self.M * self.omegaj**2)))
        return F

    def FK(self, R):
        return np.zeros(self.fmodes + 1)

    def kinked_pair_R(self):
        sdagger = 0.5 * (self.s0 + self.s1) - self.epsilon / (self.ms * self.omegas**2 * (self.s0 - self.s1))
        R = np.zeros(self.fmodes + 1)
        R[:-1] = self.cj * sdagger / (self.M * self.omegaj**2)
        R[-1] = sdagger
        return R

    def V0s(self,s):
        return 0.5 * self.ms * self.omegas**2 * (s - self.s0)**2

    def V1s(self,s):
        return 0.5 * self.ms * self.omegas**2 * (s - self.s1)**2 + self.epsilon

    def phis(self,s):
        return 2 * np.cosh(self.beta / 2 * np.sqrt((self.V0s(s) - self.V1s(s))**2 + 4 * self.Delta**2)) * np.exp(-self.beta / 2 * (self.V0s(s) + self.V1s(s))) - np.exp(-self.beta * self.V0s(s)) - np.exp(-self.beta * self.V1s(s))

    def Psy(self, s, y):
        return np.sqrt(self.a / np.pi) * self.eta * self.f(y, 0) * np.exp(-self.a * ((self.V0s(s) - self.V1s(s))/self.Delta)**2) * self.phis(s) + self.f(y, -1) * np.exp(-self.beta * self.V0s(s)) + self.f(y, 1) * np.exp(-self.beta * self.V1s(s))
    
    def TST_Rate(self):
        nsigma = 5.
        s_low = self.s0 - nsigma / np.sqrt(self.beta * self.ms * self.omegas**2); s_high = self.s1 + nsigma / np.sqrt(self.beta * self.ms * self.omegas**2)
        P = quad(self.Psy, s_low, s_high, args=(0.))[0]
        Q = dblquad(self.Psy, -3., 0., s_low, s_high)[0]
        return 1 / np.sqrt(2 * np.pi * self.beta * self.my) * P / Q

