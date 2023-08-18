import numpy as np
import sys
import os
sys.path.append('/storage/home/hcoda1/8/vsuarez6/p-jkretchmer3-0/repos/kcrpmd')
import kcrpmd_system as KcrpmdSystem
from kcrpmd import Kcrpmd

beta = 1 / (300 * 3.16681156345e-6)
a0 = 1.
b = 5000.
c = 1000.

ms = 1836.
omegas = 2.28e-3
s0 = -2.40
s1 = 2.40
epsilon = 0.
nbath = 7

mq = 50000.
omegaq = 1e-3
K0 =  0.1 / beta
bq = 0.

sysparam = np.array([ms, omegas, s0, s1, epsilon, nbath, mq, omegaq, K0, bq])

eta = 2 * np.pi * np.exp(-2 * bq**2 / (beta * mq * omegaq**2))
my = beta**3 * K0**2 / (2 * np.pi) * np.exp(bq**2 / (beta * mq * omegaq**2))
gammay = 0.5 * np.sqrt((1 - 2 * np.log(np.sqrt(a0 / np.pi) * eta * abs(beta * K0)**2)) / (beta * my))

system = KcrpmdSystem.SystemB(beta, a0, b, c, eta, my, sysparam)

dt = 2.5
Nstep = 40000000
Nprint = 2500
Ntemp = 25000
set_y = 0.

kcrpmd = Kcrpmd(system, dt, Nstep, Nprint, langevin_nve=False, gammay=gammay, resample_vel=True, Ntemp=Ntemp, fix_y=True, set_y=set_y, fix_s=False)

kcrpmd.kernel()
