import numpy as np
from classy import Class

def make_pkclass_dists(z):
# Reference Cosmology:
    omega_b = 0.02237
    omega_cdm = 0.1200
    As =  2.0830e-9
    ns = 0.9649
    h = 0.6736
    speed_of_light = 2.99792458e5

    nnu = 1
    nur = 2.0328
    # mnu = 0.06
    omega_nu = 0.0006442 #0.0106 * mnu
    # mnu = omega_nu / 0.0106

    # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2

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
        'tau_reio': 0.0568,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm}

    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()
    
    Hz_fid = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
    chiz_fid = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
    fid_dists = (Hz_fid, chiz_fid)

    return pkclass,fid_dists

def make_pkclass(z,omegaM=0.31,h=0.6766):
# Reference Cosmology:
    OmegaM = omegaM
    omega_b = 0.02242
    h = h
    lnAs =  3.047
    ns = 0.9665

    nnu = 1
    nur = 2.033
    # mnu = 0.06
    # omega_nu = 0.0106 * mnu

    omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2

    pkparams = {
        'output': 'mPk',
        'P_k_max_h/Mpc': 20.,
        'z_pk': '0.0,10',
        'A_s': np.exp(lnAs)*1e-10,
        'n_s': ns,
        'h': h,
        'N_ur': nur,
        'N_ncdm': nnu,
        'm_ncdm': mnu,
        'tau_reio': 0.0568,
        'omega_b': omega_b,
        'omega_cdm': omega_c}

    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()

    return pkclass