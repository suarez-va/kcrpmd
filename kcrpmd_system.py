import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod

# Classical Only
class KcrpmdSystem(ABC):

    @abstractmethod
    def __init__(self, beta, a0, b, c, d, sysparam):
        self.beta = beta
        self.a0 = a0
        self.b = b
        self.c = c
        self.d = d
        self.sysparam = sysparam

    @abstractmethod
    def set_eta_my_gammay(self, *args):
        pass

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
        return self.Vg(R) - 1 / self.beta * np.log(1 + np.exp(-self.beta * (self.Ve(R) - self.Vg(R))))
    
    def VKP(self, R):
        V0 = self.V0(R); V1 = self.V1(R); K = self.K(R)
        if (V0 == V1):
            return V0 - 1 / self.beta * np.log(4 * np.sinh(0.5 * self.beta * K)**2)
        elif (self.beta * K > 1e-3):
            return self.VMF(R) - 1 / self.beta * np.log(1 - np.exp(-self.beta * (V0 - self.VMF(R))) - np.exp(-self.beta * (V1 - self.VMF(R))))
        else:
            return 0.5 * (self.V0(R) + self.V1(R)) - 1 / self.beta * np.log((self.beta * K)**2 * np.sinh(0.5 * self.beta * (V0 - V1)) / (0.5 * self.beta * (V0 - V1)))

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
        return self.Fg(R) + 0.5 * (self.Fe(R) - self.Fg(R)) * (1 + np.tanh(-0.5 * self.beta * (self.Ve(R) - self.Vg(R))))
    
    def FKP(self, R):
        V0 = self.V0(R); V1 = self.V1(R); K = self.K(R)
        if (V0 == V1):
            return self.F0(R) - self.FK(R) * np.cosh(0.5 * self.beta * K) / np.sinh(0.5 * self.beta * K)
        elif (self.beta * K > 1e-3):
            return (self.FMF(R) - self.F0(R) * np.exp(-self.beta * (V0 - self.VMF(R))) - self.F1(R) * np.exp(-self.beta * (V1 - self.VMF(R)))) / (1 - np.exp(-self.beta * (V0 - self.VMF(R))) - np.exp(-self.beta * (V1 - self.VMF(R))))
        else:
            return 0.5 * (self.F0(R) + self.F1(R)) + (self.F0(R) - self.F1(R)) * (1 / (self.beta * (V0 - V1)) - 0.5 * np.cosh(0.5 * self.beta * (V0 - V1)) / np.sinh(0.5 * self.beta * (V0 - V1)))  - 2 * self.FK(R) / (self.beta * K)

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
        return 1 + 0.5 * (np.sqrt(self.a(R) / np.pi) * self.eta - 1) * (1 + np.tanh(-self.d * (self.beta * abs(self.K(R)) - 1)))

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
        hyperbolic_arg = -self.d * (self.beta * abs(self.K(R)) - 1)
        if abs(hyperbolic_arg) > 250.:
            return self.eta / (4 * np.sqrt(np.pi)) * (1 + np.tanh(-self.d * (self.beta * abs(self.K(R)) - 1))) * self.da(R) / np.sqrt(self.a(R))
        else:
            return self.eta / (4 * np.sqrt(np.pi)) * (1 + np.tanh(-self.d * (self.beta * abs(self.K(R)) - 1))) * self.da(R) / np.sqrt(self.a(R)) + 0.5 * self.d * self.beta * np.sign(self.K(R)) * (np.sqrt(self.a(R) / np.pi) * self.eta - 1) / (np.cosh(hyperbolic_arg)**2) * self.FK(R) 

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
        return ((1 / (self.beta * self.C(R)) * self.dC(R) - self.w(R)**2 / self.beta * self.da(R) - 2 * self.a(R) * self.w(R) / self.beta * self.dw(R) + self.FKP(R)) * np.exp(VKP_arg - exp_shift) + self.F0(R) * np.exp(V0_arg - exp_shift) + self.F1(R) * np.exp(V1_arg - exp_shift)) / (np.exp(VKP_arg - exp_shift) + np.exp(V0_arg - exp_shift) + np.exp(V1_arg - exp_shift))

##############################################################

class SystemAB(KcrpmdSystem):
    def __init__(self, beta, a0, b, c, d, sysparam):
        super().__init__(beta, a0, b, c, d, sysparam)
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

    def set_eta_my_gammay(self, q_array):
        Vq = lambda q: np.piecewise(q, [q >= 0., q < 0.], [lambda q: 0.5 * self.mq * self.omegaq**2 * q**2, lambda q: self.Dq * (1 - np.exp(-np.sqrt(0.5 * self.mq * self.omegaq**2 / self.Dq) * q))**2])
        Kq = lambda q: self.K0 * np.exp(-self.bq * q)
        exp_arg = -self.beta * (Vq(q_array))
        exp_shift = np.max(exp_arg) - 500.
        Pq_array = np.exp(exp_arg - exp_shift) / np.trapz(np.exp(exp_arg - exp_shift), q_array)
        Kq_array = Kq(q_array)
        self.eta = 2 * np.pi * np.trapz(abs(Kq_array) * Pq_array, q_array) * np.trapz(abs(Kq_array)**2 * Pq_array, q_array) / np.trapz(abs(Kq_array)**3 * Pq_array, q_array)  
        self.my = self.beta**3 * self.eta**2 / ((2*np.pi)**3) * (np.trapz(abs(Kq_array)**3 * Pq_array, q_array) / np.trapz(abs(Kq_array)**2 * Pq_array, q_array))**2
        self.gammay = 0.5 * np.sqrt((1 + np.abs(-2 * np.log(np.sqrt(self.a0 / np.pi) * self.eta * self.beta**2) - 4 * np.trapz(np.log(abs(Kq_array)) * Pq_array, q_array))) / (self.beta * self.my)) 
        return None

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
    def __init__(self, beta, a0, b, c, d, sysparam):
        super().__init__(beta, a0, b, c, d, sysparam)
        self.ms = float(sysparam[0])
        self.omegas = float(sysparam[1])
        self.s0 = float(sysparam[2])
        self.s1 = float(sysparam[3])
        self.epsilon = float(sysparam[4])
        self.nbath = int(sysparam[5])

        self.mq= float(sysparam[6])
        self.q0 = float(sysparam[7])
        self.lepsilon = float(sysparam[8])
        self.Ea = float(sysparam[9])
        self.Dq = float(sysparam[10])
        self.K0= float(sysparam[11])
        self.bq= float(sysparam[12]) 
        self.set_ABC()

        self.HW = sysparam[13]
        self.q_star = float(sysparam[14])
        self.k_star = float(sysparam[15])
        self.set_HW()

        self.set_dnuclei()
        self.set_mR()

    def set_ABC(self):
        def minimize_me(x, q0, lepsilon, Ea):
            [A, B, C] = x
            return (q0 - (3 * np.abs(B) + np.sqrt(9 * B**2 - 32 * np.abs(A) * C)) / (8 * np.abs(A)))**2 + (lepsilon - (np.abs(A) * ((3 * np.abs(B) + np.sqrt(9 * B**2 - 32 * np.abs(A) * C)) / (8 * np.abs(A)))**4 - np.abs(B) * ((3 * np.abs(B) + np.sqrt(9 * B**2 - 32 * np.abs(A) * C)) / (8 * np.abs(A)))**3 + C * ((3 * np.abs(B) + np.sqrt(9 * B**2 - 32 * np.abs(A) * C)) / (8 * np.abs(A)))**2))**2 + (Ea - (np.abs(A) * ((3 * np.abs(B) - np.sqrt(9 * B**2 - 32 * np.abs(A) * C)) / (8 * np.abs(A)))**4 - np.abs(B) * ((3 * np.abs(B) - np.sqrt(9 * B**2 - 32 * np.abs(A) * C)) / (8 * np.abs(A)))**3 + C * ((3 * np.abs(B) - np.sqrt(9 * B**2 - 32 * np.abs(A) * C)) / (8 * np.abs(A)))**2))**2
        x_guess = np.array([16 * self.Ea / (self.q0**4), 32 * self.Ea / (self.q0**3), 16 * self.Ea / (self.q0**2)])
        for i in range(7):
            x_guess = minimize(minimize_me, x_guess, args = (self.q0, self.lepsilon, self.Ea), method='Nelder-Mead', tol=1e-11).x
        self.Aq = np.abs(x_guess[0])
        self.Bq = np.abs(x_guess[1])
        self.Cq = x_guess[2]
        return None

    def VC(self, q):
        return np.piecewise(q, [q >= 0., q < 0.], [lambda q: self.Aq * q**4 - self.Bq * q**3 + self.Cq * q**2, lambda q: self.Dq * (1 - np.exp(-np.sqrt(self.Cq / self.Dq) * q))**2])

    def FC(self, q):
        return np.piecewise(q, [q >= 0., q < 0.], [lambda q: -4 * self.Aq * q**3 + 3 * self.Bq * q**2 - 2 * self.Cq * q, lambda q: -2 * self.Dq * np.sqrt(self.Cq / self.Dq) * np.exp(-np.sqrt(self.Cq / self.Dq) * q) * (1 - np.exp(-np.sqrt(self.Cq / self.Dq) * q))])

    def set_HW(self):
        if self.HW == 'ad':
            self.VHW = lambda q: np.piecewise(q, [q <= self.q_star, q > self.q_star], [lambda q: 0., lambda q: self.k_star * (q - self.q_star)**6])
            self.FHW = lambda q: np.piecewise(q, [q <= self.q_star, q > self.q_star], [lambda q: 0., lambda q: -6 * self.k_star * (q - self.q_star)**5])
        elif self.HW == 'nad':
            self.VHW = lambda q: np.piecewise(q, [q >= self.q_star, q < self.q_star], [lambda q: 0., lambda q: self.k_star * (q - self.q_star)**6])
            self.FHW = lambda q: np.piecewise(q, [q >= self.q_star, q < self.q_star], [lambda q: 0., lambda q: -6 * self.k_star * (q - self.q_star)**5])
        else:
            self.VHW = lambda q: 0.
            self.FHW = lambda q: 0.

    def Vq(self, q):
        return self.VC(q) + self.VHW(q)

    def Fq(self, q):
        return self.FC(q) + self.FHW(q)

    def set_eta_my_gammay(self, q_array):
        Kq = lambda q: self.K0 * np.exp(-self.bq * q)
        exp_arg = -self.beta * (self.Vq(q_array))
        exp_shift = np.max(exp_arg) - 500.
        Pq_array = np.exp(exp_arg - exp_shift) / np.trapz(np.exp(exp_arg - exp_shift), q_array)
        Kq_array = Kq(q_array)
        self.eta = 2 * np.pi * np.trapz(abs(Kq_array) * Pq_array, q_array) * np.trapz(abs(Kq_array)**2 * Pq_array, q_array) / np.trapz(abs(Kq_array)**3 * Pq_array, q_array)  
        self.my = self.beta**3 * self.eta**2 / ((2*np.pi)**3) * (np.trapz(abs(Kq_array)**3 * Pq_array, q_array) / np.trapz(abs(Kq_array)**2 * Pq_array, q_array))**2
        self.gammay = 0.5 * np.sqrt((1 + np.abs(-2 * np.log(np.sqrt(self.a0 / np.pi) * self.eta * self.beta**2) - 4 * np.trapz(np.log(abs(Kq_array)) * Pq_array, q_array))) / (self.beta * self.my)) 
        return None

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
            if self.HW == 'nad':
                R[1] = self.q0
        else:
            R[0] = sdagger
            R[1:1 + self.nbath] = self.cj * sdagger / (self.M * self.omegaj**2)
            if self.HW == 'nad':
                R[1 + self.nbath] = self.q0
        return R

