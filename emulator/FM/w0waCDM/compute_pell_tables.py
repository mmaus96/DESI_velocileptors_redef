import numpy as np

from classy import Class
from linear_theory import f_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

# k vector to use:
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )



def compute_pell_tables(pars, z=0.8, fid_dists= (None,None), ap_off=False ):
    
    w0,wa, omega_b,omega_cdm, h, logA = pars
    Hzfid, chizfid = fid_dists
    speed_of_light = 2.99792458e5

    # omega_b = 0.02242
    w0 = w0
    wa = wa
    As =  np.exp(logA)*1e-10#2.0830e-9
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
        # 'tau_reio': 0.0568,
        'z_reio': 7.,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'Omega_Lambda': 0.,
        'w0_fld': w0,
        'wa_fld': wa}

    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()

    # Caluclate AP parameters
    Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
    chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
    apar, aperp = Hzfid / Hz, chiz / chizfid
    
    if ap_off:
        apar, aperp = 1.0, 1.0
    
    # Calculate growth rate
    fnu = pkclass.Omega_nu / pkclass.Omega_m()
    # f   = f_of_a(1/(1.+z), OmegaM=OmegaM) * (1 - 0.6 * fnu)
    f = pkclass.scale_independent_growth_factor_f(z)

    # Calculate and renormalize power spectrum
    ki = np.logspace(-3.0,1.0,200)
    pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] )
    # pi = (sigma8/pkclass.sigma8())**2 * pi
    
    # Now do the RSD
    modPT = LPT_RSD(ki, pi, kIR=0.2,use_Pzel = False,\
                cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
    modPT.make_pltable(f, kv=kvec, apar=apar, aperp=aperp, ngauss=3)
    
    return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable
    