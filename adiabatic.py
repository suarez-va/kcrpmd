import numpy as np

# Classical Lmimit Build
class Adiabatic():

    def __init__(self, sys, delt, Nstep, Nprint, resample_vel=False, Ntemp=100, fix_s=False):
        self.sys = sys
        self.delt = delt
        self.Nstep = Nstep
        self.Nprint = Nprint
        self.resample_vel = resample_vel
        self.Ntemp = Ntemp
        self.fix_s = fix_s
        self.R = self.sys.kinked_pair_R(); self.vR = np.zeros(self.R.shape[0]); self.FR = np.zeros(self.R.shape[0])
        self.sdagger = self.R[0]

        self.file_output = open('output.dat', 'w')
        self.file_positions = open('positions.dat', 'w')
        self.file_initialization = open('initialization.dat', 'w')

###############################################################

    def get_R(self):
        self.R += self.vR * self.delt

    def get_vR(self):
        self.vR += 0.5 * self.delt * self.FR / self.sys.mR

    def get_FR(self):
        self.FR = self.sys.Fg(self.R)

    def resample_vR(self):
        self.vR = self.sys.sample_vR()

###############################################################

    def get_pe(self):
        return self.sys.Vg(self.R) 

    def get_ke(self):
        return self.sys.TKC(self.vR)

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
        positions = self.R
        np.savetxt(self.file_positions, positions.reshape(1, positions.shape[0]), fmt_str)
        self.file_positions.flush()

###############################################################

    def integrate(self):        
        if (self.fix_s):
            self.R[0] = self.sdagger; self.vR[0] = 0.; self.FR[0] = 0.
            self.get_vR()
            self.get_R()
            self.get_FR()
            self.R[0] = self.sdagger; self.vR[0] = 0.; self.FR[0] = 0.
            self.get_vR()
        else:
            self.get_vR()
            self.get_R()
            self.get_FR()
            self.get_vR()

    def kernel(self):
        initialization = np.zeros(2)
        initialization[0] = self.R[0]
        initialization[1] = self.vR[0]
        np.savetxt(self.file_initialization, initialization.reshape(1, initialization.shape[0]), '%20.8e')
        self.file_initialization.flush()
        for step in range(self.Nstep):
            currtime = step * self.delt
            if(self.resample_vel and np.mod(step, self.Ntemp) == 0):
                self.resample_vR()
            if(np.mod(step, self.Nprint) == 0):
                print(f"PRINTING!!! --> t = {currtime}")
                self.print_data(currtime)
            self.integrate()

###############################################################

