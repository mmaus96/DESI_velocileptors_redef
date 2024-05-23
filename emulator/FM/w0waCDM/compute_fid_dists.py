import numpy as np
import sys

from classy import Class

# Reference Cosmology:

def compute_fid_dists(z, OmegaM, h=0.6766):

    fb = 0.1571
    h = h
    ns = 0.9665
    speed_of_light = 2.99792458e5


    pkparams = {
        'output': 'mPk',
    'P_k_max_h/Mpc': 20.,
    'z_pk': '0.0,10',
    'A_s': np.exp(3.040)*1e-10,
    'n_s': 0.9665,
    'h': h,
    'N_ur': 3.046,
    'N_ncdm': 0,#1,
    #'m_ncdm': 0,
    'tau_reio': 0.0568,
    'omega_b': h**2 * fb * OmegaM,
    'omega_cdm': h**2 * (1-fb) * OmegaM}

#     OmegaM = OmegaM
#     omega_b = 0.02242
#     h = H0/100.
#     lnAs =  3.047
#     ns = 0.9665

#     nnu = 1
#     nur = 2.033
#     mnu = 0.06
#     omega_nu = 0.0106 * mnu

#     omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2

#     pkparams = {
#         'output': 'mPk',
#         'P_k_max_h/Mpc': 20.,
#         'z_pk': '0.0,10',
#         'A_s': np.exp(lnAs)*1e-10,
#         'n_s': ns,
#         'h': h,
#         'N_ur': nur,
#         'N_ncdm': nnu,
#         'm_ncdm': mnu,
#         'tau_reio': 0.0568,
#         'omega_b': omega_b,
#         'omega_cdm': omega_c}

    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()

    Hz_fid = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
    chiz_fid = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
    
    return Hz_fid, chiz_fid

if __name__ == '__main__':
    
    z = float(sys.argv[1])
    
    print("Computing fid. dists. for z = ", z)

    np.savetxt('fid_dists_z_%.2f.txt'%(z), compute_fid_dists(z,0.31) )