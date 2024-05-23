import numpy as np
import json
import sys
import os
from mpi4py import MPI

from compute_fid_dists import compute_fid_dists
from taylor_approximation import compute_derivatives

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
if mpi_rank==0:
    print(sys.argv[0]+" running on {:d} processes.".format(mpi_size))
    
basedir = sys.argv[1] +'/'
z = float(sys.argv[2])

output_shape = (1,1)

# First construct the grid

order = 5
# these are OmegaM, h, sigma8
x0s = [0.31, 0.68]; Nparams = len(x0s) # these are chosen to be roughly at the BOSS best fit value
dxs = [0.01, 0.01]

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

Hzgrid = np.zeros( (Npoints,)*Nparams+ output_shape)
Hzgridii = np.zeros( (Npoints,)*Nparams+ output_shape)

Chizgrid = np.zeros( (Npoints,)*Nparams+ output_shape)
Chizgridii = np.zeros( (Npoints,)*Nparams+ output_shape)

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        Hz,Chiz = compute_fid_dists(z=z,OmegaM = coord[0], h=coord[1])
        
        Hzgridii[iis] = Hz
        Chizgridii[iis] = Chiz
        
comm.Allreduce(Hzgridii, Hzgrid, op=MPI.SUM)
comm.Allreduce(Chizgridii, Chizgrid, op=MPI.SUM)

del(Hzgridii, Chizgridii)

derivs0 = compute_derivatives(Hzgrid, dxs, center_ii, 6)
derivs2 = compute_derivatives(Chizgrid, dxs, center_ii, 6)

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
comm.Barrier()

# Now save:
outfile = basedir + 'emu/cosmo_dists_z_%.2f.txt'%(z)

list0 = [ dd.tolist() for dd in derivs0 ]
list2 = [ dd.tolist() for dd in derivs2 ]

outdict = {'params': ['omegam', 'h'],\
           'x0': x0s,\
           'derivs_Hz': list0,\
           'derivs_Chiz': list2
           }