import numpy as np

# Classical Lmimit Build
class Kcrpmd():

    def __init__(self, sys, delt, Nstep, Nprint, langevin_nve=False, gammay=0., resample_vel=False, Ntemp=100, fix_y=False, set_y=-1., fix_s=False):
        self.sys = sys
        self.delt = delt
        self.Nstep = Nstep
        self.Nprint = Nprint
        self.langevin_nve = langevin_nve
        self.gammay = gammay; self.sigma = np.sqrt(2 * self.gammay / (self.sys.beta * self.sys.my)); self.theta = 0.; self.xi = 0.
        self.resample_vel = resample_vel
        self.Ntemp = Ntemp
        self.fix_y = fix_y
        self.fix_s = fix_s
        self.y = set_y; self.vy = 0.; self.Fy = 0.
        self.R = self.sys.kinked_pair_R(); self.vR = np.zeros(self.R.shape[0]); self.FR = np.zeros(self.R.shape[0])
        self.sdagger = self.R[0]

        self.file_output = open('output.dat', 'w')
        self.file_positions = open('positions.dat', 'w')
        self.file_initialization = open('initialization.dat', 'w')

###############################################################

    def get_y(self):
        if (self.langevin_nve):
            self.y += self.vy * self.delt + self.sigma / (2 * np.sqrt(3)) * self.theta * np.sqrt(self.delt**3)
        else:
            self.y += self.vy * self.delt

    def get_vy(self):
        if (self.langevin_nve):
            self.vy += 0.5 * self.delt * self.Fy / self.sys.my - 0.5 * self.gammay * self.vy * self.delt + 0.5 * self.sigma * self.xi * np.sqrt(self.delt) - 0.125 * self.gammay * self.delt**2 * (self.Fy / self.sys.my - self.gammay * self.vy) - 0.25 * self.gammay * self.sigma * np.sqrt(self.delt**3) * (0.5 * self.xi + 1/np.sqrt(3) * self.theta)
        else:
            self.vy += 0.5 * self.delt * self.Fy / self.sys.my

    def get_Fy(self):
        self.Fy = self.sys.FKC_y(self.y, self.R)

    def resample_vy(self):
        self.vy = self.sys.sample_vy()

###############################################################

    def get_R(self):
        self.R += self.vR * self.delt

    def get_vR(self):
        self.vR += 0.5 * self.delt * self.FR / self.sys.mR

    def get_FR(self):
        self.FR = self.sys.FKC_R(self.y, self.R)

    def resample_vR(self):
        self.vR = self.sys.sample_vR()

###############################################################

    def get_pe(self):
        return self.sys.VKC(self.y, self.R) 

    def get_ke(self):
        return self.sys.TKC(self.vy, self.vR)

    def print_data(self, currtime):
        fmt_str = '%20.8e'
        # Output Data
        engpe = self.get_pe()
        engke = self.get_ke()
        etot = engpe + engke
        output = np.zeros(4)
        output[0] = currtime
        output[1] = etot
        output[2] = engpe
        output[3] = engke
        np.savetxt(self.file_output, output.reshape(1, output.shape[0]), fmt_str)
        self.file_output.flush()
        # Positions Data
        positions = np.zeros(self.R.shape[0] + 1)
        positions[0] = self.y
        positions[1:] = self.R
        np.savetxt(self.file_positions, positions.reshape(1, positions.shape[0]), fmt_str)
        self.file_positions.flush()

###############################################################

    def integrate(self):
        if (self.langevin_nve):
            self.theta = np.random.normal()
            self.xi = np.random.normal()
        if (self.fix_y and self.fix_s):
            print("WARNING: YOU'RE TRYING TO FIX BOTH y AND s!!!")        
        elif (self.fix_y):
            self.vy = 0.; self.Fy = 0.
            self.get_vR()
            self.get_R()
            self.get_FR()
            self.get_vR()
        elif (self.fix_s):
            self.R[0] = self.sdagger; self.vR[0] = 0.; self.FR[0] = 0.
            self.get_vy(); self.get_vR()
            self.get_y(); self.get_R()
            self.get_Fy(); self.get_FR()
            self.R[0] = self.sdagger; self.vR[0] = 0.; self.FR[0] = 0.
            self.get_vy(); self.get_vR()
        else:
            self.get_vy(); self.get_vR()
            self.get_y(); self.get_R()
            self.get_Fy(); self.get_FR()
            self.get_vy(); self.get_vR()

    def kernel(self):
        initialization = np.zeros(4)
        initialization[0] = self.y
        initialization[1] = self.vy
        initialization[2] = self.R[0]
        initialization[3] = self.vR[0]
        np.savetxt(self.file_initialization, initialization.reshape(1, initialization.shape[0]), '%20.8e')
        self.file_initialization.flush()
        for step in range(self.Nstep):
            currtime = step * self.delt
            if(self.resample_vel and np.mod(step, self.Ntemp) == 0):
                self.resample_vR()
                self.resample_vy()
            if(np.mod(step, self.Nprint) == 0):
                print(f"PRINTING!!! --> t = {currtime}")
                self.print_data(currtime)
            self.integrate()

###############################################################

