import numpy as np

from classy import Class
from linear_theory import f_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

from scipy.special import spherical_jn
from scipy.integrate import simps
from scipy.interpolate import interp1d
from linear_theory import*
from pnw_dst import pnw_dst

#from zeldovich_rsd_recon_fftw import Zeldovich_Recon
from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform as SBT

# k vector to use:
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )

# Reference Cosmology:
# z = 0.59
# Omega_M = 0.31
# fb = 0.1571
# h = 0.6766
# ns = 0.9665
# speed_of_light = 2.99792458e5


# pkparams = {
#     'output': 'mPk',
#     'P_k_max_h/Mpc': 20.,
#     'z_pk': '0.0,10',
#     'A_s': np.exp(3.040)*1e-10,
#     'n_s': 0.9665,
#     'h': h,
#     'N_ur': 3.046,
#     'N_ncdm': 0,#1,
#     #'m_ncdm': 0,
#     'tau_reio': 0.0568,
#     'omega_b': h**2 * fb * Omega_M,
#     'omega_cdm': h**2 * (1-fb) * Omega_M}

# import time
# t1 = time.time()
# pkclass = Class()
# pkclass.set(pkparams)
# pkclass.compute()

# Hz_fid = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
# chiz_fid = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 

# print(Hz_fid, chiz_fid)

kint = np.logspace(-3, 2, 2000)
sphr = SBT(kint,L=5,fourier=True,low_ring=False)

def compute_bao_pkmu(mu_obs, B1, F, klin, plin, pnw, f0, apar, aperp, R, sigmas):
    '''
    Helper function to get P(k,mu) post-recon in RecSym.
        
    This is turned into Pkell and then Hankel transformed in the bao_predict funciton.
    '''

        
    sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds = sigmas
    pw = plin - pnw
    

    Sk = np.exp(-0.5*(klin*R)**2)
        
    # Our philosophy here is to take kobs = klin
    # Then we predict P(ktrue) on the klin grid, so that the final answer is
    # Pobs(kobs,mu_obs) ~ Ptrue(ktrue, mu_true) = interp( klin, Ptrue(klin, mu_true) )(ktrue)
    # (Up to a normalization that we drop.)
    # Interpolating this way avoids having to interpolate Pw and Pnw separately.
        
    F_AP = apar/aperp
    AP_fac = np.sqrt(1 + mu_obs**2 *(1./F_AP**2 - 1) )
    mu = mu_obs / F_AP / AP_fac
    ktrue = klin/aperp*AP_fac
        
    # First construct P_{dd,ss,ds} individually
    #New changes: same damp fac for P_{dd,ss,ds}
    dampfac_dd = np.exp( -0.5 * klin**2 * sigmadd * (1 + f0*(2+f0)*mu**2) )
    pw_damp = dampfac_dd * pw
    pdd = ( (1 + F*mu**2)*(1-Sk) + B1 )**2 * (pw_damp + pnw)
        
    # then Pss
    # dampfac_ss = np.exp( -0.5 * klin**2 * sigmass * (1 + f0*(2+f0)*mu**2) )
    pss = Sk**2 * (1 + F*mu**2)**2 * (pw_damp + pnw)
        
    # Finally Pds
    # dampfac_ds = np.exp(-0.5 * klin**2 * ( 0.5*sigmads_dd*(1+f0*(2+f0)*mu**2)\
    #                                          + 0.5*sigmads_ss \
    #                                          + (1+f0*mu**2)*sigmads_ds) )
    # dampfac_ds = np.exp(-0.5 * klin**2 * ( 0.5*sigmads_dd + 0.5*sigmads_ss+ sigmads_ds) * (1 + f0*(2+f0)*mu**2) )
    linfac = - Sk * (1 + F*mu**2) *  ( (1+F*mu**2)*(1-Sk) + B1 )
    pds = linfac * (pw_damp + pnw)
        
    # Sum it all up and interpolate?
    ptrue = pdd + pss - 2*pds
    pmodel = interp1d(klin, ptrue, kind='cubic', fill_value=0,bounds_error=False)(ktrue)
    
    return pmodel

def compute_xiells(rout, B1, F, klin, plin, pnw, f0, apar, aperp, R, sigmas):
        

    # Generate the sampling
    ngauss = 4
    nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
    nus_calc = nus[0:ngauss]
        
    L0 = np.polynomial.legendre.Legendre((1))(nus)
    L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
    
    pknutable = np.zeros((len(nus),len(klin)))
    
    for ii, nu in enumerate(nus_calc):
        pknutable[ii,:] = compute_bao_pkmu(nu, B1, F, klin, plin, pnw, f0, apar, aperp, R, sigmas)
 
    pknutable[ngauss:,:] = np.flip(pknutable[0:ngauss],axis=0)
        
    p0 = 0.5 * np.sum((ws*L0)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[m0,m1,m2,m3,m4,m5]) / klin
    p2 = 2.5 * np.sum((ws*L2)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[q0,q1,q2,q3,q4,q5]) / klin

    p0t = interp1d(klin,p0, kind='cubic', bounds_error=False, fill_value=0)(kint)
    p2t = interp1d(klin,p2, kind='cubic', bounds_error=False, fill_value=0)(kint)

    damping = np.exp(-(kint/10)**2)
    rr0, xi0t = sphr.sph(0,p0t * damping)
    rr2, xi2t = sphr.sph(2,p2t * damping); xi2t *= -1

    return interp1d(rr0,xi0t,kind='cubic')(rout), interp1d(rr0,xi2t,kind='cubic')(rout)


def compute_xiell_tables(pars, z=0.61, fid_dists= (None,None), R=15., rmin=50, rmax=160, dr=0.1 ):
    
    w,omega_b,omega_cdm, h, logA = pars
    Hzfid, chizfid = fid_dists
    speed_of_light = 2.99792458e5
    
    print(pars)

    # omega_b = 0.02242
    w0=w
    wa = 0.

    As =  np.exp(logA)*1e-10
    ns = 0.9649

    nnu = 1
    nur = 2.0328
    # mnu = 0.06
    omega_nu = 0.0006442 #0.0106 * mnu
    # mnu = omega_nu / 0.0106
        
    # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
    OmegaM = (omega_cdm + omega_b + omega_nu) / h**2

    pkparams = {
        'output': 'mPk',
        'P_k_max_h/Mpc': 20.,
        'z_pk': '0.0,10',
        'A_s': As,
        'n_s': ns,
        'h': h,
        'N_ur': nur,
        'N_ncdm': nnu,
        'omega_ncdm': omega_nu,
        # 'm_ncdm': mnu,
        'z_reio': 7.,
        # 'tau_reio': 0.0544,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'Omega_Lambda': 0.,
        'w0_fld': w0,
        'wa_fld': wa}

    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()
    
    print(pars, 'CLASS computed')

    # Caluclate AP parameters
    Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
    chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
    apar, aperp = Hzfid / Hz, chiz / chizfid
    
    # Calculate growth rate
    # fnu = pkclass.Omega_nu / pkclass.Omega_m()
    f0   = pkclass.scale_independent_growth_factor_f(z) #f_of_a(1/(1.+z), OmegaM=OmegaM) * (1 - 0.6 * fnu)

    # Calculate and renormalize power spectrum
    ki = np.logspace(-3.0,1.0,200)
    pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] )
    # pi = (sigma8/pkclass.sigma8())**2 * pi

    # Do the Zeldovich reconstruction predictions

    knw, pnw = pnw_dst(ki, pi)
    pw = pi - pnw
            
    qbao   = pkclass.rs_drag() * h # want this in Mpc/h units

    j0 = spherical_jn(0,ki*qbao)
    Sk = np.exp(-0.5*(ki*15)**2)

    sigmadd = simps( 2./3 * pi * (1-Sk)**2 * (1-j0), x = ki) / (2*np.pi**2)
    sigmass = simps( 2./3 * pi * (-Sk)**2 * (1-j0), x = ki) / (2*np.pi**2)

    sigmads_dd = simps( 2./3 * pi * (1-Sk)**2, x = ki) / (2*np.pi**2)
    sigmads_ss = simps( 2./3 * pi * (-Sk)**2, x = ki) / (2*np.pi**2)
    sigmads_ds = -simps( 2./3 * pi * (1-Sk)*(-Sk)*j0, x = ki) / (2*np.pi**2) # this minus sign is because we subtract the cross term
    
    sigmas = (sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds)
    
    # Now make the multipoles!
    klin, plin = ki, pi
    routs = np.arange(rmin, rmax, dr)
    
    # this is 1
    xi0_00,xi2_00  = compute_xiells(routs, 0, 0, klin, plin, pnw, f0, apar, aperp, R, sigmas )

    # this is 1 + B1 + B1^2
    # and 1 + 2 B1 + 4 B1^2
    xi0_10,xi2_10  = compute_xiells(routs,1, 0, klin, plin, pnw, f0, apar, aperp, R, sigmas )
    xi0_20,xi2_20  = compute_xiells(routs,2, 0, klin, plin, pnw, f0, apar, aperp, R, sigmas )
    
    # this is 1 + F + F^2
    # and 1 + 2 F + 4 F^2
    xi0_01,xi2_01  = compute_xiells(routs,0, 1, klin, plin, pnw, f0, apar, aperp, R, sigmas )
    xi0_02,xi2_02  = compute_xiells(routs,0, 2, klin, plin, pnw, f0, apar, aperp, R, sigmas )
    
    # and 1 + B1 + F + B1^2 + F^2 + BF
    xi0_11,xi2_11 = compute_xiells(routs,1, 1, klin, plin, pnw, f0, apar, aperp, R, sigmas )
    
    xi0table, xi2table = np.zeros( (len(routs),6) ), np.zeros( (len(routs),6) )
    
    # Form combinations:
    xi0_B1 = 0.5 * (4 * xi0_10 - xi0_20 - 3*xi0_00)
    xi0_B1sq = xi0_10 - xi0_B1 - xi0_00

    xi0_F = 0.5 * (4 * xi0_01 - xi0_02 - 3*xi0_00)
    xi0_Fsq = xi0_01 - xi0_F - xi0_00

    xi0_BF = xi0_11 - xi0_B1 - xi0_F - xi0_B1sq - xi0_Fsq - xi0_00
    
    xi2_B1 = 0.5 * (4 * xi2_10 - xi2_20 - 3*xi2_00)
    xi2_B1sq = xi2_10 - xi2_B1 - xi2_00

    xi2_F = 0.5 * (4 * xi2_01 - xi2_02 - 3*xi2_00)
    xi2_Fsq = xi2_01 - xi2_F - xi2_00

    xi2_BF = xi2_11 - xi2_B1 - xi2_F - xi2_B1sq - xi2_Fsq - xi2_00
    
    # Load
    xi0table[:,0] = xi0_00
    
    xi0table[:,1] = xi0_B1
    xi0table[:,2] = xi0_F

    xi0table[:,3] = xi0_B1sq
    xi0table[:,4]= xi0_Fsq

    xi0table[:,5] = xi0_BF
    
    xi2table[:,0] = xi2_00
    
    xi2table[:,1] = xi2_B1
    xi2table[:,2] = xi2_F

    xi2table[:,3] = xi2_B1sq
    xi2table[:,4]= xi2_Fsq

    xi2table[:,5] = xi2_BF

    return xi0table, xi2table