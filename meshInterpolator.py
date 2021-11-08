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

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

reducefun = MPI.Op.Create(lambda xyz1,xyz2,dtype : np.nanmax([xyz1,xyz2],axis=0), commute=True)

if len(sys.argv)<6:
	if mpi_rank==0:
		print("pyAlya interpolation tool:[source case name] [source dir name] [source file iteration] [target mesh file name] [number of target partitions]")
	quit()

#S stands for source
#T stands for target

SCASE_NAME = sys.argv[1]
SBASE_DIR  = sys.argv[2]
SDATA_ITE  = int(sys.argv[3])
TFILE_NAME = sys.argv[4]
TN_PART    = int(sys.argv[5])

def Init_file(fname, dime=1):

	binFile = open(fname, 'wb')
	binFile.write((27093).to_bytes(8, 'little'))    # Magic Number
	binFile.close()

	asciiFile = open(fname, 'a')
	asciiFile.write(('{:_<8}'.format('MPIAL00\0'))) # Format
	asciiFile.write(('{:_<8}'.format('V000400\0'))) # Version
	asciiFile.write(('{:_<8}'.format('XFIEL00\0')))      # Object
	asciiFile.write(('{:_<8}'.format('VECTO00\0')))   # Dimension 
	asciiFile.write(('{:_<8}'.format('NPOIN00\0')))   # Results On
	asciiFile.write(('{:_<8}'.format('REAL000\0')))        # Type
	asciiFile.write(('{:_<8}'.format('8BYTE00\0')))    # Size
	asciiFile.write(('{:_<8}'.format('SEQUE00\0'))) # Seq/Paral
	asciiFile.write(('{:_<8}'.format('NOFIL00\0'))) # Filter
	asciiFile.write(('{:_<8}'.format('ASCEN00\0'))) # Sorting
	asciiFile.write(('{:_<8}'.format('NOID000\0'))) # ID
	asciiFile.write(('{:_<8}'.format('0000000\0'))) # Alignment
	asciiFile.close()

	binFile = open(fname, 'ab')
	data = array('q', [dime, 0, 0, 1, 0, 1, -1])
	data.tofile	(binFile)
	time = array('d', [0.0])
	time.tofile	(binFile)
	binFile.close()

	asciiFile = open(fname, 'a')
	asciiFile.write(('{:_<8}'.format('0000000\0'))) # Alignment
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.write(('{:_<8}'.format('NONE000\0'))) # Option
	asciiFile.close()

def Append_to_file(fname, field):

	appdNlines = len(field)

	f = open(fname, 'rb+')
	f.seek(14*8)
	oldNlines = np.fromfile(f,count=1,dtype=np.int64)[0]
	newNlines = appdNlines + oldNlines

	f.seek(14*8)
	newline=array('q',[newNlines])
	newline.tofile(f)
	f.close

	f = open(fname, 'ab')

	for i in range(appdNlines): 
		line = field[i]
		values = array('d',[line[0],line[1],line[2]])
		values.tofile(f)
	f.close

def Interpolate(smesh,tpoints,sfields):

	ball_iter = 1000
	r_incr = 0.01
	radius = 0.005

	tfield_L = []

	mask = smesh.boundingBox.areinside(tpoints)
	owned_points = np.where(mask==True)[0]

	print("rank=",mpi_rank," num smesh points=",len(smesh.xyz)," num owned=",len(owned_points),flush=True)
	mpi_comm.Barrier()

	owned_points_L=len(owned_points)
	owned_points_MAX =mpi_comm.allreduce(owned_points_L,op=MPI.MAX)
	if mpi_rank==0: owned_points_L=1e14
	owned_points_MIN =mpi_comm.allreduce(owned_points_L,op=MPI.MIN)
	if mpi_rank==0: owned_points_L=0
	owned_points_TOT =mpi_comm.allreduce(owned_points_L,op=MPI.SUM)
	smesh_points_TOT =mpi_comm.allreduce(len(smesh.xyz),op=MPI.SUM)
	if mpi_rank==0: print(f"total points={len(tpoints)} | total smesh={smesh_points_TOT} | total owned points= {owned_points_TOT} | owned points MAX={owned_points_MAX} | owned points MIN={owned_points_MIN}",flush=True)


	count=0
	for i in owned_points:
		pyAlya.cr_start("iter_owned",0)

		tpoint = tpoints[i]
	
		mindist = 1e10
		tnodeId = -1
		for ii in range(ball_iter):

			pyAlya.cr_start("ball",0)
			ball = pyAlya.Geom.Ball(pyAlya.Geom.Point.from_array(tpoint),(1.+r_incr*ii)*radius)
			mask = ball.areinside(smesh.xyz)
			ssubset = np.where(mask==True)[0] 
			pyAlya.cr_stop("ball",0)
			

			if ssubset.shape[0] > 0: 
				pyAlya.cr_start("point_finder",0)
				found_L=1
				for nodeId in ssubset:
					node = smesh.xyz[nodeId,:]
					dist= np.linalg.norm(tpoint-node)
					if dist < mindist: 
						mindist = dist
						tnodeId = nodeId
				pyAlya.cr_stop("point_finder",0)
				break

		#if mindist > 1e-8: print(f"rank={mpi_rank} | mindist={mindist}")
		if mindist > 1e-8: count=count+1

		pyAlya.cr_start("data_append",0)
		tfield_L.append(sfields['VELOC'][tnodeId])

		#if np.linalg.norm(smesh.xyz[tnodeId]-sfields['VELOC'][tnodeId]) > 1e-5:
		#	print('check=',np.linalg.norm(smesh.xyz[tnodeId]-sfields['VELOC'][tnodeId]),' rank=',mpi_rank,' punt=',smesh.xyz[tnodeId],' data=',sfields['VELOC'][tnodeId],flush=True)
		pyAlya.cr_stop("data_append",0)
		pyAlya.cr_stop("iter_owned",0)


	print(f"rank={mpi_rank} | count of not 0 dist={count}")

	mpi_comm.Barrier()
	if mpi_rank==0: print('NO NHI HA CAP QUE NO DONI BE',flush=True)

	pyAlya.cr_start("gather",0)
	tfield_G = mpi_comm.gather(tfield_L,root=0)
	owner_list = mpi_comm.gather(owned_points,root=0)
	pyAlya.cr_stop("gather",0)

	tfield =[]

	pyAlya.cr_start("write",0)
	if mpi_rank==0:

		tfield = [0.0]*len(tpoints)

		for ol,tf in zip(owner_list,tfield_G):
			for id,value in zip(ol,tf):
				tfield[id] = value
	pyAlya.cr_stop("write",0)

	return tfield


iniTime = time.time()

#OPENING SOURCE MESH

if mpi_rank==0:
	print('OPENING SOURCE MESH',flush=True)

smesh = pyAlya.Mesh.read(SCASE_NAME,basedir=SBASE_DIR,read_commu=False,read_massm=False,read_codno=False)


# READING SOURCE DATA FILE

if mpi_rank==0:
	print('READING SOURCE DATA FILE',flush=True)

VARLIST = ['VELOC']

sfields, header = pyAlya.Field.read(SCASE_NAME,VARLIST,SDATA_ITE,smesh.xyz,basedir=SBASE_DIR)


#OPENING TARGET MESH

if mpi_rank==0:
	print('OPENING TARGET MESH BY CHUNKS',flush=True)

header = pyAlya.io.AlyaMPIO_header.read(TFILE_NAME)
npartitions = TN_PART

chunk_size = header.npoints//npartitions

if mpi_rank==0:
	Init_file(SCASE_NAME+"-XFIEL.00000001.00000001.mpio.bin", 3)

for i in range(npartitions):
	
	start = time.time()

	rows2skip = chunk_size*i
	rows2read = chunk_size
	if i == npartitions-1:
		rows2read = header.npoints-rows2skip

	if mpi_rank==0:	print(f"opening chunk {i+1}/{npartitions}",flush=True)
	tpoints, _ = pyAlya.io.AlyaMPIO_readByChunk(TFILE_NAME,rows2read,rows2skip)

	if mpi_rank==0:	print(f"interpolating chunk {i+1}/{npartitions}",flush=True)
	tfield = Interpolate(smesh,tpoints,sfields)

	pyAlya.cr_start("append_file",0)
	if mpi_rank==0:
		
		end=time.time()

		Append_to_file(SCASE_NAME+"-XFIEL.00000001.00000001.mpio.bin", tfield)
		print('pyAlya intepolator [partition ',i+1,'/',npartitions,'] [chunk time:',"{:10.4f}".format((end-start)),'s] [Elapsed time:',"{:10.4f}".format(end-iniTime),'s]',flush=True)
	pyAlya.cr_stop("append_file",0)

	pyAlya.cr_info()


if mpi_rank==0:	print('INTERPOLATION DONE')


