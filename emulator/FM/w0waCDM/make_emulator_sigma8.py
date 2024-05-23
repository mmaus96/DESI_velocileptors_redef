import numpy as np
import json
import sys
import os
from mpi4py import MPI

from taylor_approximation_mpi import compute_derivatives
from compute_class_sigma8 import compute_sigma8


comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

basedir = sys.argv[1] + '/'
z = float(sys.argv[2])

output_shape = (1,)

# First construct the grid
order = 3
# these are w0, wa,omega_b,omega_cdm, h, logA
x0s = [-1.0,0.0,0.02237, 0.1200,0.6736, 3.036394]; Nparams = len(x0s) # these are chosen to be roughly at the BOSS best fit value
dxs = [0.1,0.2,0.0015, 0.01,0.03, 0.05]

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

s8_grid = np.zeros( (Npoints,)*Nparams + output_shape )
s8_gridii = np.zeros( (Npoints,)*Nparams + output_shape )

Dz_grid = np.zeros( (Npoints,)*Nparams + output_shape )
Dz_gridii = np.zeros( (Npoints,)*Nparams + output_shape )

fz_grid = np.zeros( (Npoints,)*Nparams + output_shape )
fz_gridii = np.zeros( (Npoints,)*Nparams + output_shape )

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        s8_gridii[iis],Dz_gridii[iis],fz_gridii[iis],_ = compute_sigma8(coord,z)

comm.Allreduce(s8_gridii, s8_grid, op=MPI.SUM)
comm.Allreduce(Dz_gridii, Dz_grid, op=MPI.SUM)
comm.Allreduce(fz_gridii, fz_grid, op=MPI.SUM)
del(s8_gridii,Dz_gridii,fz_gridii)


# Now compute the derivatives
derivs_s8 = compute_derivatives(s8_grid, dxs, center_ii, 5)
derivs_Dz = compute_derivatives(Dz_grid, dxs, center_ii, 5)
derivs_fz = compute_derivatives(fz_grid, dxs, center_ii, 5)

# Now save:
if mpi_rank == 0:
    
    # Make the emulator (emu) directory if it
    # doesn't already exist.
    fb = basedir+'emu'
    if not os.path.isdir(fb):
        print("Making directory ",fb)
        os.mkdir(fb)
    else:
        print("Found directory ",fb)
    #
    outfile = basedir+'emu/abacus_s8_z%.2f.json'%(z)

    list0 = [ dd.tolist() for dd in derivs_s8 ]
    list1 = [ dd.tolist() for dd in derivs_Dz ]
    list2 = [ dd.tolist() for dd in derivs_fz ]

    outdict = {'params': ['w0','wa','omega_b','omega_cdm', 'h','logA'],\
           'x0': x0s,\
           # 'lnA0': 3.03639,\
           'derivs_s8': list0,\
           'derivs_Dz': list1,\
           'derivs_fz': list2}

    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()