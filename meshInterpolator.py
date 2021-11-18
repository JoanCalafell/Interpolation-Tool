#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
from array import array
import time

import pyAlya

import sys

import mpi4py, numpy as np, struct
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from io_utils import Write_par2seq
from interp_utils import Interpolate

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

reducefun = MPI.Op.Create(lambda xyz1,xyz2,dtype : np.nanmax([xyz1,xyz2],axis=0), commute=True)

if len(sys.argv)<5:
	if mpi_rank==0:
		print("pyAlya interpolator: [source case name] [source dir name] [source file iteration] [target mesh file name]")
	quit()

#S stands for source
#T stands for target

SCASE_NAME = sys.argv[1]
SBASE_DIR  = sys.argv[2]
SDATA_ITE  = int(sys.argv[3])
TFILE_NAME = sys.argv[4]

iniTime = time.time()

#OPENING SOURCE MESH

mpi_comm.Barrier()
if mpi_rank==0:	print('pyAlya interpolator: OPENING SOURCE MESH',flush=True)
smesh = pyAlya.Mesh.read(SCASE_NAME,basedir=SBASE_DIR,read_commu=False,read_massm=False,read_codno=False)


# READING SOURCE DATA FILE

mpi_comm.Barrier()
if mpi_rank==0: print('pyAlya interpolator: READING SOURCE DATA FILE',flush=True)
VARLIST = ['VELOC']
sfields, header = pyAlya.Field.read(SCASE_NAME,VARLIST,SDATA_ITE,smesh.xyz,basedir=SBASE_DIR)


#OPENING TARGET MESH

mpi_comm.Barrier()
if mpi_rank==0: print('pyAlya interpolatro: OPENING TARGET MESH',flush=True)   
header = pyAlya.io.AlyaMPIO_header.read(TFILE_NAME)
tpoints, _ = pyAlya.io.AlyaMPIO_readByChunk(TFILE_NAME,header.npoints,0)


#PARALLEL INTERPOLATION

pyAlya.cr_start("interpolation_loop",0)
ownedIds,tfield = Interpolate(smesh,tpoints,sfields)
pyAlya.cr_stop("interpolation_loop",0)

pyAlya.cr_start("write_file",0)
Write_par2seq(SCASE_NAME+"-XFIEL.00000001.00000001.mpio.bin",len(tpoints),ownedIds,tfield)
mpi_comm.Barrier()
end=time.time()
if mpi_rank==0:	print("pyAlya interpolator: FILE WRITTEN. [Elapsed time:","{:10.4f}".format(end-iniTime),"s]",flush=True)
pyAlya.cr_stop("write_file",0)
	

pyAlya.cr_info()
