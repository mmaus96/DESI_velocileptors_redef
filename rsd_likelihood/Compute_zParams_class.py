import numpy as np
import json
from taylor_approximation import taylor_approximate

class Compute_zParams():
    """
    Computes sigma8(0), D(z), f(z) given a Taylor series in w,h,Om,logA.
    """
    
    def __init__(self, emu_filename):
    
        json_file = open(emu_filename, 'r')
        emu = json.load( json_file )
        json_file.close()
        
        self.x0s = emu['x0']
        self.derivs_s8 = [np.array(ll) for ll in emu['derivs_s8']]
        self.derivs_Dz = [np.array(ll) for ll in emu['derivs_Dz']]
        self.derivs_fz = [np.array(ll) for ll in emu['derivs_fz']]
        
        del emu
        
    def compute_sig8(self, pars,order):
        
        s8_emu = taylor_approximate(pars, self.x0s, self.derivs_s8, order=order)[0]
        
        return s8_emu
    
    def compute_Dz(self, pars,order):
        
        Dz_emu = taylor_approximate(pars, self.x0s, self.derivs_Dz, order=order)[0]
        
        return Dz_emu
    
    def compute_fz(self, pars,order):
        
        fz_emu = taylor_approximate(pars, self.x0s, self.derivs_fz, order=order)[0]
        
        return fz_emu