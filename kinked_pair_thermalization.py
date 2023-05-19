import numpy as np
import sys
import os
sys.path.append('/storage/home/hcoda1/8/vsuarez6/kcrpmd_5.18.2023')
import kcrpmd_system as KcrpmdSystem
from kcrpmd import Kcrpmd

epsilon_array = np.array([0., 0.04, 0.07, 0.118, 0.18, 0.236])

ms = 1836.
omegas = 2.28e-3
s0 = -2.40
s1 = 2.40
epsilon = -epsilon_array[aaa - 1]
K = 6.67e-7
M = 1836.
omegac = 2.28e-3
gamma = 1. * M * omegac
fmodes = 12
sysparam = np.array([ms, omegas, s0, s1, epsilon, K, M, omegac, gamma, fmodes])

beta = 1 / (300 * 3.16681156345e-6)
a = 5e-6
b = 400.
L = 0.1
eta = 2 * np.pi
my = K**2 * beta**3 / (2 * np.pi * L**2)
system = KcrpmdSystem.SystemA(beta, a, b, L, eta, my, sysparam)

dt = 2.5
Nstep = 40000000
Nprint = 2500
gammay = 1 / (2 * (2 - L)) * np.sqrt((1 - 2 * np.log((2 - L) / L * np.sqrt(a / np.pi) * eta * beta**2) - 4 * np.log(K)) / (beta * my))
Ntemp = 25000
set_y = 0.
kcrpmd = Kcrpmd(system, dt, Nstep, Nprint, langevin_nve=False, gammay=gammay, resample_vel=True, Ntemp=Ntemp, fix_y=True, set_y=set_y)

kcrpmd.kernel()

