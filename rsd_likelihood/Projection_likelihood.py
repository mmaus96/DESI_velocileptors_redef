import numpy as np
# from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
# from make_pkclass import make_pkclass
# from EH98_funcs import*
from getdist.mcsamples    import loadMCSamples
from classy import Class
from scipy import interpolate, linalg
from copy import deepcopy
import scipy.constants as conts

class trial_likelihood(Likelihood):
    
    sig1: float
    sig2: float
    sig3: float
    mu1: float
    mu2: float
    mu3: float
    dist: int
    
    def get_requirements(self):
        
        req = {'x1': None,\
               'x2': None,\
               'x3': None,\
               }
        return(req)
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        pp   = self.provider
        x1 = pp.get_param('x1')
        x2 = pp.get_param('x2')
        x3 = pp.get_param('x3')
        
        if self.dist == 1:
            Prob = np.exp(-((x1-self.mu1)**2 + (x2-self.mu2)**2)/self.sig1**2)
            Prob += np.exp(-(x1 - self.mu3)**2 / self.sig2**2  - (x2/self.sig3)**2 )
        if self.dist == 2:
            Prob = np.exp(-(x1-x2)**2/self.sig2**2)
        if self.dist == 3:
            Prob = np.exp(-(x1-self.mu1)**2/self.sig1**2-(x2-self.mu2)**2/self.sig2**2)
            
        if self.dist == 4:
            Ros = (self.mu1 - x1)**2 + self.sig1*(x2 - x1**2)**2
            return -0.5* Ros
        if self.dist == 5:
            Prob = 0
            Prob = np.exp(-(x1-self.mu1)**2/self.sig1**2 - (x2-self.mu2)**2/self.sig2**2)
            Prob += np.exp(- 0.5*((self.mu3 - x1)**2 + self.sig3*(x2 - x1**2)**2))
            # return Prob
            
        if self.dist == 6:
            dx1 = x1-self.mu1
            dx2 = x2-self.mu2
            dx3 = x3-self.mu3
            
            Prob = np.exp(-(dx1)**2/self.sig1**2 - (dx2)**2/self.sig2**2 - (dx3)**2/self.sig3**2-(dx3-2*dx2)**2/0.1**2)
        
        if self.dist == 7:
            dx1 = x1-self.mu1
            dx2 = x2-self.mu2
            dx3 = x3-self.mu3
            
            Prob = np.exp(-(dx1)**2/self.sig1**2 - (dx2)**2/self.sig2**2 - (dx3)**2/self.sig3**2-(dx1-2*dx2)**2/0.1**2)
            # Prob = np.exp(-(x3-x1 -self.mu3 + self.mu1)**2/0.1**2 - (x1-self.mu1)**2/self.sig1**2 - (x2-self.mu2)**2/self.sig2**2  - (x3-self.mu3)**2/self.sig3**2)
            # Prob = np.exp(-((x3-self.mu3)*(x1-self.mu1)**3)**2/0.1**2 - (x1-self.mu1)**2/self.sig1**2 - (x2-self.mu2)**2/self.sig2**2  - (x3-self.mu3)**2/self.sig3**2)
            # return -0.5* Ros
            
        return(np.log(Prob)) 