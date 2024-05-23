import numpy as np
import json
import sys
import os
from mpi4py import MPI

# from compute_fid_dists import compute_fid_dists
from compute_xiell_tables_recsym import compute_xiell_tables
from taylor_approximation_mpi import compute_derivatives
from make_pkclass import make_pkclass_dists


comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

basedir = sys.argv[1] + '/'
z = float(sys.argv[2])
Omfid = float(sys.argv[3])
Rs = float(sys.argv[4])

# Set r grid:
rmin, rmax, dr = 60, 160, 0.5

# Compute fiducial distances
_,fid_dists = make_pkclass_dists(z=z)

# Remake the data grid:
order = 3
# these are w,omega_b,omega_cdm, h, logA
x0s = [-1.0,0.02237, 0.1200,0.68, 3.05]; Nparams = len(x0s) # these are chosen to be roughly at the BOSS best fit value
dxs = [0.05,0.0001, 0.01,0.01, 0.01]

# Set output shape
rr = np.arange(rmin, rmax, dr)
output_shape = (len(rr),6) # this is for 1, B1, F, B1*F, B1^2, F^2

# Make parameter grid:
template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

# Compute the grid!
X0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
X2grid = np.zeros( (Npoints,)*Nparams+ output_shape)

X0gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
X2gridii = np.zeros( (Npoints,)*Nparams+ output_shape)

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        try:
            xi0, xi2= compute_xiell_tables(coord,z=z,fid_dists=fid_dists, rmin=rmin, rmax=rmax, dr=dr,R=Rs)
        except:
            errfile = basedir+'err_file_xi.txt'
            bad_coord_iis = [coord,iis]
            np.savetxt(errfile,bad_coord_iis,encoding='utf-8')
            
        X0gridii[iis] = xi0
        X2gridii[iis] = xi2
        
        del xi0,xi2

print(mpi_rank,'grid evaluations finished')
comm.Allreduce(X0gridii, X0grid, op=MPI.SUM)
comm.Allreduce(X2gridii, X2grid, op=MPI.SUM)

del(X0gridii, X2gridii)
            
# Now compute the derivatives

if mpi_rank == 0:
    print('computing derivatives')
derivs0 = compute_derivatives(X0grid, dxs, center_ii, 4)
derivs2 = compute_derivatives(X2grid, dxs, center_ii, 4)

if mpi_rank == 0:
    print('finished computing derivatives')
    # derivs0 = compute_derivatives(X0grid, dxs, center_ii, 4)
    # derivs2 = compute_derivatives(X2grid, dxs, center_ii, 4)

    # Make the emulator (emu) directory if it
    # doesn't already exist.
    fb = basedir+'emu'
    if not os.path.isdir(fb):
        print("Making directory ",fb)
        os.mkdir(fb)
    else:
        print("Found directory ",fb)
    # Now save:
    outfile = basedir + 'emu/abacus_z_{:.2f}_xiells_recsym_sm{:2d}.json'.format(z,round(Rs))

    list0 = [ dd.tolist() for dd in derivs0 ]
    list2 = [ dd.tolist() for dd in derivs2 ]

    outdict = {'params': ['w','omega_b','omega_cdm', 'h', 'logA'],\
           'x0': x0s,\
           'rvec': rr.tolist(),\
           'derivs0': list0,\
           'derivs2': list2,}

    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()