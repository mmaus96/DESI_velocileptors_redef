import numpy as np
import time
import json

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval

from scipy.special import spherical_jn
from scipy.integrate import simps

from taylor_approximation import taylor_approximate
# from taylor_approximation import taylor_approximate
from Compute_zParams_class import Compute_zParams
from linear_theory import D_of_a,f_of_a

# Class to have a full-shape likelihood for a bunch of pieces of data from both galactic caps in the same z bin
# Currently assumes all data have the same fiducial cosmology etc.
# If not I suggest chaning the theory class so that instead of being labelled by "zstr" it gets labelled by sample name.
# And each sample name indexing the fiducial cosmology numbers (chi, Hz etc) in dictionaries. For another time...

class PkLikelihood(Likelihood):
    
    zfid: str
    
    basedir: str
    
    fs_sample_names: list
 
    fs_datfns: list

    covfn: str
    inv_cov_fac: float
    cov_fac: float
    
    fs_kmins: list
    fs_mmaxs: list
    fs_qmaxs: list
    fs_hmaxs: list
    fs_matMfns: list
    fs_matWfns: list
    # kmax_spline: list
    
    w_kin_fn: str


    def initialize(self):
        """Sets up the class."""
        # Redshift Label for theory classes
        self.zstr = "%.2f" %(self.zfid)
        print(self.fs_sample_names,self.fs_datfns)
        
        print("We are here!")
        
        self.pconv = {}
        self.xith = {}

#         self.sp_kmax = {}
        
#         for ll, fs_sample_name in enumerate(self.fs_sample_names):
#             self.sp_kmax[fs_sample_name] = self.kmax_spline[ll]
        
        self.loadData()
        #

    def get_requirements(self):
        
        req = {'taylor_pk_ell_mod': None,\
               'zPars': None,\
               'w': None,\
              'H0': None,\
               'sigma8': None,\
               'omegam': None,\
               'omega_b': None,\
               'omega_cdm': None,\
               'logA': None}
        
        for fs_sample_name in self.fs_sample_names:
            req_bias = { \
                   'bsig8_' + fs_sample_name: None,\
                   'b2sig8_' + fs_sample_name: None,\
                   'bssig8_' + fs_sample_name: None,\
                   'b3sig8_' + fs_sample_name: None,\
                   'alpha0_' + fs_sample_name: None,\
                   'alpha2_' + fs_sample_name: None,\
                   'alpha4_' + fs_sample_name: None,\
                   'SN0_' + fs_sample_name: None,\
                   'SN2_' + fs_sample_name: None,\
                   'SN4_' + fs_sample_name: None,\
                   }
            req = {**req, **req_bias}

        return(req)
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        thy_obs = []
        
        for fs_sample_name in self.fs_sample_names:
            fs_thy  = self.fs_predict(fs_sample_name)
            fs_obs  = self.fs_observe(fs_thy, fs_sample_name)
            thy_obs = np.concatenate( (thy_obs,fs_obs) )

        diff = self.dd - thy_obs
        
        chi2 = np.dot(diff,np.dot(self.cinv,diff))
        #print('diff', self.sample_name, diff[:20])
        #
        return(-0.5*chi2)
        #
        
    def loadData(self):
        """
        Loads the required data.
        
        Do this in two steps... first load full shape data then xirecon, concatenate after.
        
        The covariance is assumed to already be joint in the concatenated format.
        
        """
        # First load the data
        
        self.kdats = {}
        self.p0dats = {}
        self.p2dats = {}
        self.p4dats = {}
        self.fitiis = {}
        
        for ii, fs_datfn in enumerate(self.fs_datfns):
            fs_sample_name = self.fs_sample_names[ii]
            fs_dat = np.loadtxt(self.basedir+fs_datfn)
            self.kdats[fs_sample_name] = fs_dat[:,0]
            self.p0dats[fs_sample_name] = fs_dat[:,1]
            self.p2dats[fs_sample_name] = fs_dat[:,2]
            self.p4dats[fs_sample_name] = fs_dat[:,3]
            
            # Make a list of indices for the monopole and quadrupole only in Fourier space
            # This is specified to each sample in case the k's are different.
            yeses = self.kdats[fs_sample_name] > 0
            nos   = self.kdats[fs_sample_name] < 0
            self.fitiis[fs_sample_name] = np.concatenate( (yeses, nos, yeses, nos, yeses ) )
        

        
        # Join the data vectors together
        self.dd = []        
        for fs_sample_name in self.fs_sample_names:
            self.dd = np.concatenate( (self.dd, self.p0dats[fs_sample_name], self.p2dats[fs_sample_name], self.p4dats[fs_sample_name]) )

        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.basedir+self.covfn)/self.cov_fac
        
        # We're only going to want some of the entries in computing chi^2.
        
        # this is going to tell us how many indices to skip to get to the nth multipole
        startii = 0
        
        for ss, fs_sample_name in enumerate(self.fs_sample_names):
            
            kcut = (self.kdats[fs_sample_name] > self.fs_mmaxs[ss])\
                          | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:     # FS Monopole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
            kcut = (self.kdats[fs_sample_name] > self.fs_qmaxs[ss])\
                       | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
            kcut = (self.kdats[fs_sample_name] > self.fs_hmaxs[ss])\
                       | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:       # FS Hexadecapole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
        
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov) * self.inv_cov_fac
        #print(self.sample_name, np.diag(self.cinv)[:10])
        
        # Finally load the window function matrix.
        self.matMs = {}
        self.matWs = {}
        for ii, fs_sample_name in enumerate(self.fs_sample_names):
            self.matMs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matMfns[ii])
            self.matWs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matWfns[ii])
        
        self.w_kin = np.loadtxt(self.w_kin_fn)
        
    def combine_bias_terms_pkell(self,bvec, p0ktable, p2ktable, p4ktable):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        
        return p0, p2, p4
    
    def fs_predict(self, fs_sample_name):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        
        taylorPTs = pp.get_result('taylor_pk_ell_mod')
        kv, p0ktable, p2ktable, p4ktable = taylorPTs[self.zstr]

        #
        sig8 = pp.get_param('sigma8')
        zPars = pp.get_result('zPars')
        D_z = zPars[self.zstr][0]
        f_z = zPars[self.zstr][1]
        sig8_z = sig8*D_z
        
        #sig8 = pp.get_result('sigma8')
        b1   = pp.get_param('bsig8_' + fs_sample_name)/sig8_z - 1
        # b1   = pp.get_param('b1_' + fs_sample_name)
        b2   = pp.get_param('b2sig8_' + fs_sample_name)/(sig8_z**2)
        bs   = pp.get_param('bssig8_' + fs_sample_name)/(sig8_z**2)
        b3   = pp.get_param('b3sig8_' + fs_sample_name)/(sig8_z**3)
        
        alp0_tilde = pp.get_param('alpha0_' + fs_sample_name)
        alp2_tilde = pp.get_param('alpha2_' + fs_sample_name)
        alp4_tilde = pp.get_param('alpha4_' + fs_sample_name)
        
        alp0 = (1+b1)**2 * alp0_tilde
        alp2 = f_z*(1+b1)*(alp0_tilde+alp2_tilde)
        alp4 = f_z*(f_z*alp2_tilde+(1+b1)*alp4_tilde)
        alp6 = f_z**2*alp4_tilde
        
        
        
        sn0  = pp.get_param('SN0_' + fs_sample_name)
        sn2  = pp.get_param('SN2_' + fs_sample_name)
        sn4  = pp.get_param('SN4_' + fs_sample_name)
        
        
        
        bias = [b1, b2, bs, b3]
        cterm = [alp0,alp2,alp4,alp6]
        stoch = [sn0, sn2, sn4]
        bvec = bias + cterm + stoch
        
        #print(self.zstr, b1, sig8)
        
        p0, p2, p4 = self.combine_bias_terms_pkell(bvec, p0ktable, p2ktable, p4ktable)
        
        #np.savetxt('pells_' + self.zstr + '_' + self.sample_name + '.txt',[kv,p0,p2,p4])
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],kv),np.append([0.0,],p0)
        p2 = np.append([0.,],p2)
        p4 = np.append([0.0,],p4)
        tt    = np.array([kv,p0,p2,p4]).T
        
        if np.any(np.isnan(tt)):
            print("NaN's encountered. Parameter values are: ", str(hub,sig8,OmM))
        
        return(tt)
        #

    def fs_observe(self,tt,fs_sample_name):
        """Apply the window function matrix to get the binned prediction."""
        
        
        # print(fs_sample_name)
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        # kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        # maxk = self.sp_kmax[fs_sample_name]
        # kv  = np.linspace(0.0,maxk,int(maxk/0.001),endpoint=False) + 0.0005
        kv  = self.w_kin
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,1],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        
        if np.any(np.isnan(thy)) or np.max(thy) > 1e8:
            hub = self.provider.get_param('H0') / 100.
            sig8 = self.provider.get_param('sigma8')
            OmM = self.provider.get_param('omegam')
            print("NaN's encountered. Parameter values are: ", str(hub,sig8,OmM))
        
        # wide angle
        # expanded_model = np.matmul(self.matMs[fs_sample_name], thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        # Multiply by ad-hoc factor
        # convolved_model = np.matmul(self.matWs[fs_sample_name], expanded_model )
        convolved_model = np.matmul(self.matWs[fs_sample_name], thy )
        
        # keep only the monopole and quadrupole
        # convolved_model = convolved_model[self.fitiis[fs_sample_name]]
        
        # Save the model:
        self.pconv[fs_sample_name] = convolved_model
    
        return convolved_model
    

class Taylor_pk_theory_zs(Theory):
    """
    A class to return a set of derivatives for the Taylor series of Pkell.
    """
    zfids: list
    pk_filenames: list
    s8_filenames: list
    basedir: str
    omega_nu: float
    
    def initialize(self):
        """Sets up the class by loading the derivative matrices."""
        
        print("Loading Taylor series.")
        
        # Load sigma8
        # self.compute_sigma8 = Compute_Sigma8(self.basedir + self.s8_filename)
        
        # self.s8_emu = Compute_zParams(self.basedir + self.s8_filename)
        
        # Load clustering
        self.taylors_pk = {}
        self.s8_emus = {}
        
        for zfid, pk_filename,s8_filename in zip(self.zfids, self.pk_filenames, self.s8_filenames):
            zstr = "%.2f"%(zfid)
            taylors_pk = {}
            
            # Load the power spectrum derivatives
            json_file = open(self.basedir+pk_filename, 'r')
            emu = json.load( json_file )
            json_file.close()
            
            x0s = emu['x0']
            kvec = emu['kvec']
            derivs_p0 = [np.array(ll) for ll in emu['derivs0']]
            derivs_p2 = [np.array(ll) for ll in emu['derivs2']]
            derivs_p4 = [np.array(ll) for ll in emu['derivs4']]
            
            taylors_pk['x0'] = np.array(x0s)
            taylors_pk['kvec'] = np.array(kvec)
            taylors_pk['derivs_p0'] = derivs_p0
            taylors_pk['derivs_p2'] = derivs_p2
            taylors_pk['derivs_p4'] = derivs_p4


            self.taylors_pk[zstr] = taylors_pk
            self.s8_emus[zstr] = Compute_zParams(self.basedir + s8_filename)
            
            del emu
    
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zmax = max(self.zfids)
        zg  = np.linspace(0,zmax,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {\
               'w': None,\
               'omega_b': None,\
               'omega_cdm': None,\
               'H0': None,\
               'logA': None,\
              }
        
        return(req)
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        return ['taylor_pk_ell_mod','zPars']
    
    def get_can_provide_params(self):
        return ['sigma8','omegam']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider
        
        w = pp.get_param('w')
        hub = pp.get_param('H0') / 100.
        logA = pp.get_param('logA')
        omega_b = pp.get_param('omega_b')
        omega_cdm = pp.get_param('omega_cdm')
        OmM = (omega_cdm + omega_b + self.omega_nu)/hub**2
        #sig8 = pp.get_param('sigma8')
        # OmM = pp.get_param('omegam')
        # sig8 = self.compute_sigma8.compute_sigma8(OmM,hub,logA)
        
        
        cosmopars = [w,omega_b, omega_cdm, hub, logA]
        
        ptables = {}
        zPars = {}
        for zfid in self.zfids:
            zstr = "%.2f" %(zfid)
            
            # Load pktables
            x0s = self.taylors_pk[zstr]['x0']
            derivs0 = self.taylors_pk[zstr]['derivs_p0']
            derivs2 = self.taylors_pk[zstr]['derivs_p2']
            derivs4 = self.taylors_pk[zstr]['derivs_p4']
            
            kv = self.taylors_pk[zstr]['kvec']
            p0ktable = taylor_approximate(cosmopars, x0s, derivs0, order=4)
            p2ktable = taylor_approximate(cosmopars, x0s, derivs2, order=4)
            p4ktable = taylor_approximate(cosmopars, x0s, derivs4, order=4)
            
            ptables[zstr] = (kv, p0ktable, p2ktable, p4ktable)
            
            compute_zpars = self.s8_emus[zstr]
            sig8_emu = compute_zpars.compute_sig8(cosmopars,order = 5)
            Dz_emu = compute_zpars.compute_Dz(cosmopars,order = 5)
            fz_emu = compute_zpars.compute_fz(cosmopars,order = 5)
            sig8z_emu =  sig8_emu*Dz_emu
            zPars[zstr] = [Dz_emu,fz_emu]
            
        #state['sigma8'] = sig8
        state['derived'] = {'sigma8': sig8_emu,'omegam':OmM}
        state['taylor_pk_ell_mod'] = ptables
        state['zPars'] = zPars
