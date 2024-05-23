import numpy as np
import json
import sys
import os
from mpi4py import MPI

from compute_fid_dists import compute_fid_dists
from taylor_approximation_mpi import compute_derivatives
from make_pkclass import make_pkclass_dists


comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
if mpi_rank==0:
    print(sys.argv[0]+" running on {:d} processes.".format(mpi_size))
#print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

basedir = sys.argv[1] +'/'
z = float(sys.argv[2])
Omfid = float(sys.argv[3])

# Compute fiducial distances
_,fid_dists = make_pkclass_dists(z=z)
# h = 0.6766
# speed_of_light = 2.99792458e5
# Hz_fid = fid_dist_class.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
# chiz_fid = fid_dist_class.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
# fid_dists = (Hz_fid, chiz_fid)

# Set up the output k vector:
from compute_pell_tables import compute_pell_tables, kvec

output_shape = (len(kvec),19) # two multipoles and 19 types of terms


# First construct the grid

order = 2
# these are w0,wa,omega_b,omega_cdm, h, logA
x0s = [-1.0,0.0,0.02237, 0.1200,0.6736, 3.036394]; Nparams = len(x0s) # these are chosen to be roughly at the BOSS best fit value
dxs = [0.1,0.3,0.0015, 0.01,0.03, 0.05]

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P2grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P4grid = np.zeros( (Npoints,)*Nparams+ output_shape)

P0gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
P2gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
P4gridii = np.zeros( (Npoints,)*Nparams+ output_shape)

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        try:
            p0, p2, p4 = compute_pell_tables(coord,z=z,fid_dists=fid_dists)
        except:
            errfile = basedir+'err_file.txt'
            bad_coord_iis = [coord,iis]
            np.savetxt(errfile,bad_coord_iis,encoding='utf-8')
        
        P0gridii[iis] = p0
        P2gridii[iis] = p2
        P4gridii[iis] = p4
        
comm.Allreduce(P0gridii, P0grid, op=MPI.SUM)
comm.Allreduce(P2gridii, P2grid, op=MPI.SUM)
comm.Allreduce(P4gridii, P4grid, op=MPI.SUM)

del(P0gridii, P2gridii, P4gridii)

# Now compute the derivatives
derivs0 = compute_derivatives(P0grid, dxs, center_ii, 4)
derivs2 = compute_derivatives(P2grid, dxs, center_ii, 4)
derivs4 = compute_derivatives(P4grid, dxs, center_ii, 4)

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
outfile = basedir + 'emu/abacus_z_%.2f_pkells.json'%(z)

list0 = [ dd.tolist() for dd in derivs0 ]
list2 = [ dd.tolist() for dd in derivs2 ]
list4 = [ dd.tolist() for dd in derivs4 ]

outdict = {'params': ['w0','wa','omega_b','omega_cdm', 'h', 'logA'],\
           'x0': x0s,\
           'kvec': kvec.tolist(),\
           'derivs0': list0,\
           'derivs2': list2,\
           'derivs4': list4}

if mpi_rank == 0:
    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()

