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

from interp_utils import Write_par2seq
from interp_utils import Bounding_Box

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

def Interpolate(smesh,tpoints,sfields):

	ball_iter = 100
	#r_incr = 0.1
	#radius = 0.5

	r_incr = 1.0
	radius = np.cbrt(np.nanmin(smesh._vmass))
	#radius = np.cbrt(np.nanmin(smesh._vmass))
	#print(smesh._vmass) DEMANAR A ARANU QUE ESTA DONANT AIXO!!! PQ NO SON ELS VOLUMS DELS ELEMENTS. HAURIEN DE SER TOTS IGUAL I NO HO SON.
	#print(f"rank={mpi_rank} | radius={radius}")

	tfield = []

	mask = smesh.boundingBox.areinside(tpoints)
	old_bounded_points = np.where(mask==True)[0]
	old_bounded_L = len(old_bounded_points)
	old_bounded_MAX = mpi_comm.allreduce(old_bounded_L,op=MPI.MAX)

	if mpi_rank==0: print("generating BOUNDING BOX",flush=True)

	pyAlya.cr_start("new_bound_box",0)
	maxSize_L = np.cbrt(np.nanmax(smesh._vmass))
	box = Bounding_Box(smesh.xyz)
	box.discretize_by_minsize(smesh.xyz,maxSize_L)

	bounded_index = box.areInside(tpoints)
	pyAlya.cr_stop("new_bound_box",0)

	mpi_comm.Barrier()
	if mpi_rank==0: print("finishing BOUNDING BOX",flush=True)

	#print("bounded index=",bounded_index)

	
	#print("rank=",mpi_rank," num smesh points=",len(smesh.xyz)," new bounded=",len(bounded_index)," old bounded=",len(old_bounded_points),flush=True)
	#mpi_comm.Barrier()

	bounded_points_L=len(bounded_index)
	bounded_points_MAX =mpi_comm.allreduce(bounded_points_L,op=MPI.MAX)
	if mpi_rank==0: bounded_points_L=1e14
	bounded_points_MIN =mpi_comm.allreduce(bounded_points_L,op=MPI.MIN)
	if mpi_rank==0: bounded_points_L=0
	bounded_points_TOT =mpi_comm.allreduce(bounded_points_L,op=MPI.SUM)
	smesh_points_TOT =mpi_comm.allreduce(len(smesh.xyz),op=MPI.SUM)

	real_points_L=len(smesh.xyz)
	real_points_MAX = mpi_comm.allreduce(len(smesh.xyz),op=MPI.MAX)
	if mpi_rank==0: real_points_L =1e14 
	real_points_MIN = mpi_comm.allreduce(real_points_L,op=MPI.MIN) 
	if mpi_rank==0: print(f"total points={len(tpoints)} | total smesh={smesh_points_TOT} | total bounded points= {bounded_points_TOT} | real max = {real_points_MAX} | bounded points MAX={bounded_points_MAX} | old bounded MAX={old_bounded_MAX} | diff={old_bounded_MAX-bounded_points_MAX} | real min={real_points_MIN} | bounded points MIN={bounded_points_MIN}",flush=True)

	bounded_points_L=len(bounded_index)
	if bounded_points_MAX==bounded_points_L: print(f"El rank amb mes bounded points es rank={mpi_rank}",flush=True)
	real_points_L=len(smesh.xyz)
	if real_points_MAX== len(smesh.xyz) : print(f"El rank amb mes points reals es rank={mpi_rank}",flush=True)


	ownedIds =[]
	for bnodeId in bounded_index:
		pyAlya.cr_start("iter_owned",0)

		tpoint = tpoints[bnodeId]

		#if mpi_rank==1: print(f"punt target: {tpoint}")
	
		mindist   = 1e10
		mindist_L = 1e10
		tnodeId = -1
		count=0
		found = False
		for ii in range(ball_iter):

			pyAlya.cr_start("ball",0)
			ball = pyAlya.Geom.Ball(pyAlya.Geom.Point.from_array(tpoint),(1.+r_incr*ii)*radius)
			mask = ball.areinside(smesh.xyz)
			ssubset = np.where(mask==True)[0] 
			pyAlya.cr_stop("ball",0)
			
			if ssubset.shape[0] >= 2: 													# I need at least two owned points to evaluate the local mesh charactristic lenght 
				pyAlya.cr_start("point_finder",0)
				found_L=1
				
				#if mpi_rank==1: print(f"number of sphere iterations={ii} | sub-set shape={ssubset.shape[0]}")
				
				for nodeId in ssubset:													#here the nearest owned node is found
					node = smesh.xyz[nodeId,:]
					dist= np.linalg.norm(tpoint-node)
					#if mpi_rank==1: print(f"node subset={node} | dist: {dist}")
					if dist < mindist: 
						mindist = dist
						tnodeId = nodeId
				pyAlya.cr_stop("point_finder",0)
					
				pyAlya.cr_start("char_length",0)
				nodeMin = smesh.xyz[tnodeId,:]
				#if mpi_rank==1: print(f"mindist={mindist} | minnode: {nodeMin}")
				for nodeId in ssubset:													#here, the local mesh characteristic lenght is evaluated
					if tnodeId != nodeId:
						node = smesh.xyz[nodeId,:]
						dist= np.linalg.norm(nodeMin-node)
						if dist < mindist_L: 
							mindist_L = dist

				pyAlya.cr_stop("char_length",0)
				found = True
				break
		
		if mindist/mindist_L < 0.99 and found: 
			tfield.append(sfields['VELOC'][tnodeId])
			ownedIds.append(bnodeId)

		pyAlya.cr_stop("iter_owned",0)
	
	print("rank=",mpi_rank," num smesh points=",len(smesh.xyz)," num bounded=",len(bounded_index)," num owned=",len(ownedIds),' diff=', len(bounded_index,)-len(ownedIds),flush=True)

	return ownedIds,tfield

	'''
	mpi_comm.Barrier()
	pyAlya.cr_start("gather",0)
	tfield_G = mpi_comm.gather(tfield_L,root=0)
	owned_points_G = mpi_comm.gather(owned_points,root=0)
	pyAlya.cr_stop("gather",0)

	tfield =[]

	pyAlya.cr_start("write",0)
	if mpi_rank==0:

		tfield = [0.0]*len(tpoints)

		for op,tf in zip(owned_points_G,tfield_G):
			for id,value in zip(op,tf):
				tfield[id] = value
	pyAlya.cr_stop("write",0)

	return tfield
	'''

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

#if mpi_rank==0:
#	Init_file(SCASE_NAME+"-XFIEL.00000001.00000001.mpio.bin", 3)

for i in range(npartitions):
	
	start = time.time()

	rows2skip = chunk_size*i
	rows2read = chunk_size
	if i == npartitions-1:
		rows2read = header.npoints-rows2skip

	if mpi_rank==0:	print(f"opening chunk {i+1}/{npartitions}",flush=True)
	tpoints, _ = pyAlya.io.AlyaMPIO_readByChunk(TFILE_NAME,rows2read,rows2skip)

	pyAlya.cr_start("interpolation_loop",0)
	mpi_comm.Barrier()
	if mpi_rank==0:	print("INTERPOLATING!!!!!!!!!!!!!!!",flush=True)
	ownedIds,tfield = Interpolate(smesh,tpoints,sfields)
	mpi_comm.Barrier()
	if mpi_rank==0:	print("INTERPOLATION DONE!!!!!!!!!!!!!!",flush=True)
	pyAlya.cr_stop("interpolation_loop",0)

	pyAlya.cr_start("write_file",0)
	if mpi_rank==0:	print("WRITING FILE!!!!!!!!!!!!!!!",flush=True)
	Write_par2seq(SCASE_NAME+"-XFIEL.00000001.00000001.mpio.bin",len(tpoints),ownedIds,tfield)
	mpi_comm.Barrier()
	if mpi_rank==0:	print("FILE WRITTEN!!!!!!!!!!!!!!!",flush=True)
	pyAlya.cr_stop("write_file",0)
	
	mpi_comm.Barrier()
	if mpi_rank==0:
		end=time.time()
		#Append_to_file(SCASE_NAME+"-XFIEL.00000001.00000001.mpio.bin", tfield)
		print('pyAlya intepolator [Elapsed time:',"{:10.4f}".format(end-iniTime),'s]',flush=True)

	pyAlya.cr_info()


if mpi_rank==0:	print('INTERPOLATION DONE')

