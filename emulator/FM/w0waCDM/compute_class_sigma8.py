import numpy as np

from classy import Class

def compute_sigma8(pars, z):
    
    w0,wa, omega_b,omega_cdm, h, logA = pars

    # omega_b = 0.02242
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
    
    sigma8_0 = pkclass.sigma8()
    f_z = pkclass.scale_independent_growth_factor_f(z)
    D_z = pkclass.scale_independent_growth_factor(z)
    sigma8_z = pkclass.sigma(8./h,z)

    return sigma8_0, D_z, f_z, sigma8_z
    
