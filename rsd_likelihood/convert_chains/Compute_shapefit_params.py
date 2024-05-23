import numpy as np
from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
# from make_pkclass import make_pkclass_dists
from EH98_funcs import*
from getdist.mcsamples    import loadMCSamples
from classy import Class
from scipy import interpolate, linalg
from copy import deepcopy
import scipy.constants as conts
import numpy as np
import json
import os
import sys
from mpi4py import MPI
import time

chaindir = sys.argv[1] +'/'
z= float(sys.argv[4])

i_start = int(sys.argv[2])
i_end = int(sys.argv[3])

tic = time.perf_counter()

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
if mpi_rank==0:
    print("Compute Dists running on {:d} processes.".format(mpi_size))
    
samples = loadMCSamples(chaindir + '/pk',no_cache=True, \
                            settings={'ignore_rows':0.3, 'contours': [0.68, 0.95],\
                                     })

p = samples.getParams().__dict__


#Fiducial Parameters
kmpiv = 0.03
speed_of_light = 2.99792458e5
kvec = np.logspace(-2.0, 0.0, 300)
zfid = z
As_fid = np.exp(3.0364)/1e10
fid_class = Class()

fid_class.set({
"A_s": As_fid,
"n_s": float(0.9649),
"H0": 100.0*float(0.6736),
"omega_b": float(0.02237),
"omega_cdm": float(0.1200),
"N_ur": float(2.0328),
"N_ncdm": int(1),
"m_ncdm": float(0.06),
"Omega_k": float(0.0),
"output": "mPk",
"P_k_max_1/Mpc": float(1.0),
"z_max_pk": zfid + 0.5
 })


fid_class.compute()


# fid_class = make_pkclass(zfid)
h_fid = fid_class.h()
Hz_fid = fid_class.Hubble(zfid) * conts.c/1000.0
Chiz_fid = fid_class.angular_distance(zfid) * (1.+zfid)
rd_fid = fid_class.rs_drag()
pi_fid = np.array( [fid_class.pk_cb(k*h_fid, zfid ) * h_fid**3 for k in kvec] )

# fAmp_fid = fid_class.scale_independent_growth_factor_f(zfid)*np.sqrt(fid_class.pk_lin(kmpiv*h_fid,zfid)*h_fid**3)
# Amp_fid = fid_class.scale_independent_growth_factor_f(zfid)*np.sqrt(fid_class.pk_lin(kmpiv*h_fid,zfid)*h_fid**3)
Amp_fid = np.sqrt(fid_class.pk_lin(kmpiv*h_fid,zfid)*h_fid**3)

transfer_fid = EH98(kvec, zfid, 1.0, cosmo=fid_class)#*h_fid**3
# transfer_fid = EH98_transfer(kvec, zfid, 1.0, cosmo = fid_class)
# P_r_fid = Primordial(kvec*h_fid, As_fid, 0.9649, k_p = 0.05)
# _,P_nw_fid = pnw_dst(kvec,pi_fid)
# transfer_fid = P_nw_fid/P_r_fid
sigma8_fid = fid_class.sigma(8.0/h_fid, zfid)

# fsigma8_fid = fid_class.scale_independent_growth_factor_f(zfid)*fid_class.sigma(8.0/fid_class.h(), zfid)
# Pk_ratio_fid = EHpk_fid
print(zfid, sigma8_fid)

fiducial_all = [h_fid, Hz_fid, rd_fid, Chiz_fid, Amp_fid, transfer_fid, sigma8_fid]

theo_fsig8_all = np.zeros(len(p['H0']))
theo_apara_all = np.zeros(len(p['H0']))
theo_aperp_all = np.zeros(len(p['H0']))
theo_mslope_all = np.zeros(len(p['H0']))
weights_all = np.zeros(len(p['H0']))
logLs_all = np.zeros(len(p['H0']))

theo_fsig8_is = np.zeros(len(p['H0']))
theo_apara_is = np.zeros(len(p['H0']))
theo_aperp_is = np.zeros(len(p['H0']))
theo_mslope_is = np.zeros(len(p['H0']))
weights_is = np.zeros(len(p['H0']))
logLs_is = np.zeros(len(p['H0']))
# theo_fAmp*fsigma8_fid, theo_apara, theo_aperp, theo_mslope

outfile = chaindir + 'computed_SF_pars_{}.json'.format(i_start)

for i,x in enumerate(p['H0'][i_start:i_end]):
    i+= i_start
    if i%mpi_size == mpi_rank:
        h = p['H0'][i] / 100.
        logA = p['logA'][i]
        omega_b = p['omega_b'][i]
        omega_cdm = p['omega_cdm'][i]
        kmpiv = 0.03
        kvec = np.logspace(-2.0, 0.0, 300)
        
        wt = samples.weights[i]
        logL = samples.loglikes[i]
        
        As =  np.exp(logA)*1e-10
        ns = 0.9649

        nnu = 1
        nur = 2.0328
        mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106

        # omega_c = (OmM - omega_b/h**2 - omega_nu/h**2) * h**2
        OmegaM = (omega_cdm + omega_b + omega_nu) / h**2

        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 1.,
            'z_max_pk': z + 0.5,
            # 'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            # 'omega_ncdm': omega_nu,
            'm_ncdm': mnu,
            'tau_reio': 0.0544,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_k': float(0.0)}

        try:
            pkclass = Class()
            pkclass.set(pkparams)
            pkclass.compute()
        except:
            print('CLASS error')
            theo_fsig8_is[i] = float("nan") #theo_fsig8_is[i-1]
            theo_apara_is[i] = float("nan") #theo_apara_is[i-1]
            theo_aperp_is[i] = float("nan") #theo_aperp_is[i-1]
            theo_mslope_is[i] = float("nan") #theo_mslope_is[i-1]
            weights_is[i] = float("nan")
            logLs_is[i] = float("nan")
            continue
        f = pkclass.scale_independent_growth_factor_f(z)
        sigma8 = pkclass.sigma(8.0 / pkclass.h(), z)
        fsigma8_unsc = f*sigma8

        # Caluclate AP parameters
        Hz = pkclass.Hubble(z) *conts.c/1000.0
        Chiz = pkclass.angular_distance(z) * (1.+z)
         
        rd = pkclass.rs_drag()
        pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in kvec] )
        
        #Compute the ratio of fAmp in the Shapefit paper in order to find the corresponding fsigma8 parameter. 
        # theo_fAmp = pkclass.scale_independent_growth_factor_f(z)*np.sqrt(pkclass.pk_lin(kmpiv*h_fid*rd_fid/(rd),z)*(h*rd_fid/(rd))**3.)/fAmp_fid
        theo_fAmp = pkclass.scale_independent_growth_factor_f(z)*np.sqrt(pkclass.pk_lin(kmpiv*h_fid*rd_fid/(rd),z)*(h_fid*rd_fid/(rd))**3.)/Amp_fid
        theo_aperp = (Chiz) / Chiz_fid / rd * rd_fid
        theo_apara = Hz_fid/ (Hz) / rd * rd_fid
        
        #Compute the transfer function with the EH98 formula. 
        transfer_new = EH98(kvec*h_fid*rd_fid/(rd*h), z ,1.0, cosmo=pkclass) #*(rd_fid/rd)**3

        # P_r = Primordial(kvec*h*(rd_fid/rd), As, ns, k_p = 0.05)
        # _,P_nw = pnw_dst(kvec*h_fid*rd_fid/(rd*h),pi)
        # transfer_new = P_nw/P_r

        #Find the slope at the pivot scale.  
        ratio_transfer = slope_at_x(np.log(kvec), np.log(transfer_new/transfer_fid))
        theo_mslope = interpolate.interp1d(kvec, ratio_transfer, kind='cubic')(kmpiv)
        
        theo_fsig8_is[i] = theo_fAmp*sigma8_fid
        theo_apara_is[i] = theo_apara 
        theo_aperp_is[i] = theo_aperp
        theo_mslope_is[i] = theo_mslope
        weights_is[i] = wt
        logLs_is[i] = logL
        
    if (i%1000 ==0) and (mpi_rank == 0):
        toc = time.perf_counter()
        print('{}th process (out of {}) at {} mins'.format(i+1,len(p['H0'][i_start:i_end]),(toc-tic)/60))
        
comm.Allreduce(theo_fsig8_is, theo_fsig8_all, op=MPI.SUM)
comm.Allreduce(theo_apara_is, theo_apara_all, op=MPI.SUM)
comm.Allreduce(theo_aperp_is, theo_aperp_all, op=MPI.SUM)
comm.Allreduce(theo_mslope_is, theo_mslope_all, op=MPI.SUM)
comm.Allreduce(weights_is, weights_all, op=MPI.SUM)
comm.Allreduce(logLs_is, logLs_all, op=MPI.SUM)

del(theo_fsig8_is,theo_apara_is,theo_aperp_is,theo_mslope_is,weights_is,logLs_is)
comm.Barrier()

outdict = {'z':z, \
           'fsigma8':theo_fsig8_all.tolist(), \
           'apar':theo_apara_all.tolist(), \
           'aperp': theo_aperp_all.tolist(),\
           'm':theo_mslope_all.tolist(), \
           'weights': weights_all.tolist(),\
           'logLs': logLs_all.tolist(),\
           'i_start':i_start,'i_final': i}

if mpi_rank == 0:
    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()