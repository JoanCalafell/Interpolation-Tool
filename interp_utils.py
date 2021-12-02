#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import math
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

class Bounding_Box:

	def __init__(self,*args):

		if len(args)==1:
			self._pmin = np.array([1e100,1e100,1e100])
			self._pmax = np.array([-1e100,-1e100,-1e100])

			#ill defined is used basically used for the rank 0 which tries to create a box without contents.
			self._illDefined=False
			self._box_from_points(args[0])

		if len(args)==7:
			self._pmin=np.array([args[0],args[2],args[4]])
			self._pmax=np.array([args[1],args[3],args[5]])

			self._illDefined=False
			if args[0]<=args[1] or args[2]<=args[3] or args[4]<=args[5]: 
				self._illDefined=True

		elif len(args)==2:
			self._pmin = args[0]
			self._pmax = args[1]

			self._illDefined=False
			if args[0][0]<=args[1][0] or args[0][1]<=args[1][1] or args[0][2]<=args[1][2]: 
				self._illDefined=True

		else:
			if mpi_rank==0: (f"Bounding Box: Incorrect number of arguments. Expected 1, 2 or 7 arguments, {len(args)} provided.")


		self._isIn  = []
		self._nParti = 1
		self._nPartj = 1
		self._nPartk = 1

	def __str__(self):
		return f"pmin={self._pmin}\npmax={self._pmax}\npartitions i={self._nParti} j={self._nPartj} k={self._nPartk}"

	@property
	def pmin(self):
		return self._pmin

	@property
	def pmax(self):
		return self._pmax

	@property
	def isIn(self):
		return self._isIn

	def _box_from_points(self,points):

		if points.shape[0]<4: 
			self._illDefined=True
			print("Warning: you are attempting to generate a box from less than four non-coplanar points")
		else:
			for p in points:
				if p[0]<self._pmin[0]: self._pmin[0]=p[0]
				if p[1]<self._pmin[1]: self._pmin[1]=p[1]
				if p[2]<self._pmin[2]: self._pmin[2]=p[2]
				if p[0]>self._pmax[0]: self._pmax[0]=p[0]
				if p[1]>self._pmax[1]: self._pmax[1]=p[1]
				if p[2]>self._pmax[2]: self._pmax[2]=p[2]

	def _getIndex(self,i,j,k):

		return i+self._nParti*j+self._nParti*self._nPartj*k

	#protected method to evaluate whether a point is inside a subbox and to get the it corresponding index for the _isIn vector
	def _isInside(self,point):

		if np.isnan(point[0]) or self._illDefined:
			return False, -1
		else:
			norm_position = np.array([round((point[0]-self._pmin[0])/(self._pmax[0]-self._pmin[0]),8) \
						 ,round((point[1]-self._pmin[1])/(self._pmax[1]-self._pmin[1]),8) \
						 ,round((point[2]-self._pmin[2])/(self._pmax[2]-self._pmin[2]),8)]) #normalizes position with box real dimensions

			if norm_position[0]<0 or norm_position[0]>1 or norm_position[1]<0 or norm_position[1]>1 or norm_position[2]<0 or norm_position[2]>1: #point out of box
				return False, -1
			else:
				if self._nParti==1 and self._nPartj==1 and self._nPartk==1 : return True, -1 	#point in box with a single partition
				else: 																			#point in a discretized box										
					i = int(norm_position[0]*self._nParti) 										#determines the ijk index position within the discrete box from the normalized coordinate
					j = int(norm_position[1]*self._nPartj)
					k = int(norm_position[2]*self._nPartk)

					if i==self._nParti: i = self._nParti-1
					if j==self._nPartj: j = self._nPartj-1
					if k==self._nPartk: k = self._nPartk-1

					idx = self._getIndex(i,j,k) 

					return True, idx

	#public method to evaluate whether a point is inside a box or not
	def isInside(self,point):

		if np.isnan(point[0]) or self._illDefined:
			return False
		else:
			norm_position = np.array([round((point[0]-self._pmin[0])/(self._pmax[0]-self._pmin[0]),8) \
						 ,round((point[1]-self._pmin[1])/(self._pmax[1]-self._pmin[1]),8) \
						 ,round((point[2]-self._pmin[2])/(self._pmax[2]-self._pmin[2]),8)]) 



			if norm_position[0]<0 or norm_position[0]>1 or norm_position[1]<0 or norm_position[1]>1 or norm_position[2]<0 or norm_position[2]>1: 
				return False
			else:
				if self._nParti==1 and self._nPartj==1 and self._nPartk==1: return True
				else:
					i = int(norm_position[0]*self._nParti) 
					j = int(norm_position[1]*self._nPartj)
					k = int(norm_position[2]*self._nPartk)

					if i==self._nParti: i = self._nParti-1
					if j==self._nPartj: j = self._nPartj-1
					if k==self._nPartk: k = self._nPartk-1

					idx = self._getIndex(i,j,k) 

					return self._isIn[idx]

	#discretizes the bounding box with a custom number of partitions for each direction
	def discretize_by_parts(self,points,nParti,nPartj,nPartk):
		
		if not self._illDefined:
			self._nParti = nParti
			self._nPartj = nPartj
			self._nPartk = nPartk
			self._isIn   = [False]*(self._nParti*self._nPartj*self._nPartk)

			for point in points:

				isInside ,idx = self._isInside(point)
				if isInside: self._isIn[idx] = True


	#discretizes the bounding box with subboxes of a custom minimum size. if only one size is given, it is applied for all directions.
	def discretize_by_minsize(self,points,minsizeX,minsizeY=-1,minsizeZ=-1):

		if not self._illDefined:
			Lx= self._pmax[0]-self._pmin[0]
			Ly= self._pmax[1]-self._pmin[1]
			Lz= self._pmax[2]-self._pmin[2]

			if minsizeX<=0  : minsizeX = Lx # To avoid rank 0 minsize = 0
			if minsizeY==-1 : minsizeY = minsizeX
			if minsizeY<=0  : minsizeY = Ly
			if minsizeZ==-1 : minsizeZ = minsizeX
			if minsizeZ<=0  : minsizeZ = Lz

			nParti = int(Lx/minsizeX)	
			nPartj = int(Ly/minsizeY)	
			nPartk = int(Lz/minsizeZ)

			if nParti < 1: nParti=1	
			if nPartj < 1: nPartj=1	
			if nPartk < 1: nPartk=1	
                                                           		
			self._nParti = nParti
			self._nPartj = nPartj
			self._nPartk = nPartk
			self._isIn   = [False]*(self._nParti*self._nPartj*self._nPartk)

			for point in points:

				isInside ,idx = self._isInside(point)
				if isInside: self._isIn[idx] = True

	#returns a list of the index of points that are inside the bounding box. The indexes are referred to the original input points list. 
	def areInside(self,points):

		idxPoints=[]

		if not self._illDefined:

			for idx,p in enumerate(points):

				isInside = self.isInside(p)

				if isInside:
					idxPoints.append(idx)

		return idxPoints


def Interpolate(smesh,tpoints,sfields):

	#initializing ball search parameters
	radius = np.cbrt(np.nanmin(smesh._vmass)) #initial search ball radius equal to partition's smallest element size
	r_incr = 1.0 #the ball radius doubles at each iteration
	ball_max_iter = 100 #this value assumes that the size ratio between the largest and the smallest element within a partition is less than 100. If this assumption is false, it may cause that some target nodes would not find their source countepart, making their inteprolated value to remain zero.
	
	#Computing the discretized bounding box of the source mesh
	pyAlya.cr_start("bound_box",0)
	maxSize_L = np.cbrt(np.nanmax(smesh._vmass))
	box = Bounding_Box(smesh.xyz)
	box.discretize_by_minsize(smesh.xyz,maxSize_L)

	#Computing the target mesh points contained in the source mesh's bounding box
	boundedIds = box.areInside(tpoints)
	pyAlya.cr_stop("bound_box",0)

	#Computing bounding box efficiency parameters
	bounded_points_L=len(boundedIds)
	bounded_points_MAX =mpi_comm.allreduce(bounded_points_L,op=MPI.MAX)
	if mpi_rank==0: bounded_points_L=1e14
	bounded_points_MIN =mpi_comm.allreduce(bounded_points_L,op=MPI.MIN)

	real_points_L=len(smesh.xyz)
	real_points_MAX = mpi_comm.allreduce(len(smesh.xyz),op=MPI.MAX)
	if mpi_rank==0: real_points_L =1e14 
	real_points_MIN = mpi_comm.allreduce(real_points_L,op=MPI.MIN) 
	if mpi_rank==0: print(f"pyAlya Interpolator: source points MAX= {real_points_MAX} | bounded points MAX={bounded_points_MAX} | diff={bounded_points_MAX-real_points_MAX} | source points MIN={real_points_MIN} | bounded points MIN={bounded_points_MIN}",flush=True)

	#Interpolation process
	tfield = []
	ownedIds =[]
	#loop over all the target nodes contained in the source bounding box
	for bnodeId in boundedIds:
		pyAlya.cr_start("iter_owned",0)

		tpoint = tpoints[bnodeId]

		mindist     = 1e10
		locGridSize = 1e10
		tnodeId     = -1
		for ii in range(ball_max_iter): #loop over a growing shpere to determine the target point's closest source nodes.

			pyAlya.cr_start("ball",0)
			ball = pyAlya.Geom.Ball(pyAlya.Geom.Point.from_array(tpoint),(1.+r_incr*ii)*radius)
			mask = ball.areinside(smesh.xyz)
			ssubset = np.where(mask==True)[0] #subset of source nodes within the sphere centered in the target point
			pyAlya.cr_stop("ball",0)
			
			
			if ssubset.shape[0] >= 2: #At least two close nodes in the subset are required to evaluate the source grid local size 												 
				pyAlya.cr_start("point_finder",0)
				
				for nodeId in ssubset: #loop to determine the source node with minimum distance to the target node. The search is performed only within the sphere subset 
					node = smesh.xyz[nodeId,:]
					dist= np.linalg.norm(tpoint-node)
					
					if dist < mindist: 
						mindist = dist
						tnodeId = nodeId
				pyAlya.cr_stop("point_finder",0)
					
				pyAlya.cr_start("char_length",0)
				nodeMin = smesh.xyz[tnodeId,:]
				for nnodeId in ssubset: # evaluation of the source grid local size: minimum distance between the final source node and its neighbours in the subset
					if tnodeId != nnodeId:
						node = smesh.xyz[nnodeId,:]
						dist= np.linalg.norm(nodeMin-node)
						if dist < locGridSize: 
							locGridSize = dist
				pyAlya.cr_stop("char_length",0)

				break
		
		if mindist/locGridSize < 0.99: #if the distance between the source and the target node is larger than the local grid size, the node is discarded since it means that is out of the source element and thus, out of the owned domain. Otherwise, it is considered an owned node. 
			tfield.append(sfields['VELOC'][tnodeId])
			ownedIds.append(bnodeId)

		pyAlya.cr_stop("iter_owned",0)
	
	print("pyAlya interpolator: rank=",mpi_rank,"num source points=",len(smesh.xyz)," num of target bounded points=",len(boundedIds)," num of target owned points=",len(ownedIds),flush=True)
	return ownedIds,np.array(tfield)



def rotate_vector(vector,gamma,beta,alpha,rotc_x=0.0,rotc_y=0.0,rotc_z=0.0):

	# https://en.wikipedia.org/wiki/Rotation_matrix
	R = np.ndarray(shape=(3,3))

	alpha = math.pi*alpha/180.0
	beta  = math.pi*beta/180.0
	gamma = math.pi*gamma/180.0

	R[0][0] = math.cos(alpha)*math.cos(beta)
	R[1][0] = math.cos(alpha)*math.sin(beta)*math.sin(gamma)-math.sin(alpha)*math.cos(gamma)
	R[2][0] = math.cos(alpha)*math.sin(beta)*math.cos(gamma)+math.sin(alpha)*math.sin(gamma)
	R[0][1] = math.sin(alpha)*math.cos(beta)
	R[1][1] = math.sin(alpha)*math.sin(beta)*math.sin(gamma)+math.cos(alpha)*math.cos(gamma)
	R[2][1] = math.sin(alpha)*math.sin(beta)*math.cos(gamma)-math.cos(alpha)*math.sin(gamma)
	R[0][2] = -math.sin(beta)
	R[1][2] = math.cos(beta)*math.sin(gamma)
	R[2][2] = math.cos(beta)*math.cos(gamma)

	rotc=np.array([rotc_x,rotc_y,rotc_z])
	vector = vector-rotc
	vector = np.dot(vector,R)
	vector = vector+rotc

	return vector

def rotate_field(field,gamma,beta,alpha,rotc_x=0.0,rotc_y=0.0,rotc_z=0.0):

	pyAlya.pprint(0,f"rotating field: angle={gamma},{beta},{alpha} | rot center={rotc_x},{rotc_y},{rotc_z}")
	output=[]
	R = np.ndarray(shape=(3,3))

	alpha = math.pi*alpha/180.0
	beta  = math.pi*beta/180.0	
	gamma = math.pi*gamma/180.0

	R[0][0] = math.cos(alpha)*math.cos(beta)
	R[1][0] = math.cos(alpha)*math.sin(beta)*math.sin(gamma)-math.sin(alpha)*math.cos(gamma)
	R[2][0] = math.cos(alpha)*math.sin(beta)*math.cos(gamma)+math.sin(alpha)*math.sin(gamma)
	R[0][1] = math.sin(alpha)*math.cos(beta)
	R[1][1] = math.sin(alpha)*math.sin(beta)*math.sin(gamma)+math.cos(alpha)*math.cos(gamma)
	R[2][1] = math.sin(alpha)*math.sin(beta)*math.cos(gamma)-math.cos(alpha)*math.sin(gamma)
	R[0][2] = -math.sin(beta)
	R[1][2] = math.cos(beta)*math.sin(gamma)
	R[2][2] = math.cos(beta)*math.cos(gamma)

	rotc=np.array([rotc_x,rotc_y,rotc_z])
	for v in field: 
		vector = v-rotc
		vector = np.dot(vector,R)
		vector = vector+rotc
		output.append(vector)

	return np.array(output)


