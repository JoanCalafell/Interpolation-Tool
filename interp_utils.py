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

def Init_file(fname,dime=1,dataset_len=0):

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
	data = array('q', [dime, dataset_len, 0, 1, 0, 1, -1])
	data.tofile(binFile)
	time = array('d', [0.0])
	time.tofile(binFile)
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

	binFile = open(fname, 'ab')
	data = array('d', [0.0]*dataset_len)
	data.tofile(binFile)
	binFile.close()

def Write_par2seq(fname,dataset_len,glob_ids,field):

	dime_L=0
	if mpi_rank != 0: 
		dime_L=field[0].shape[0]
	dime=mpi_comm.allreduce(dime_L,op=MPI.MAX)
	
	if mpi_rank==0:	Init_file(fname,dime=dime,dataset_len=dataset_len)
	mpi_comm.Barrier()
	
	header_offset=256
	data_size=8*dime
	f = open(fname, 'rb+')
	'''
	if mpi_rank==1:
		id=glob_ids[10]
		data=field[10]

		data=np.array([1.0,2.0,3.0])
		f.seek(header_offset+(id*data_size))
		values = array('d',data)
		values.tofile(f)

		print("id=",id," data=",data)

	'''
	for gi,line in zip(glob_ids,field):
		f.seek(header_offset+(gi*data_size))
		values = array('d',line)
		values.tofile(f)

	f.close
		
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


class Bounding_Box:

	def __init__(self,*args):

		if len(args)==1:
			self._pmin = np.array([1e100,1e100,1e100])
			self._pmax = np.array([-1e100,-1e100,-1e100])

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

		#print(f"rank={mpi_rank} is ill defined={self._illDefined}")

	def _getIndex(self,i,j,k):

		return i+self._nParti*j+self._nParti*self._nPartj*k

	def _isInside(self,point):

		if np.isnan(point[0]) or self._illDefined:
			return False, -1
		else:
			norm_position = np.array([(point[0]-self._pmin[0])/(self._pmax[0]-self._pmin[0]),(point[1]-self._pmin[1])/(self._pmax[1]-self._pmin[1]),(point[2]-self._pmin[2])/(self._pmax[2]-self._pmin[2])]) #normalizes position with box real dimensions

			if norm_position[0]<0 or norm_position[0]>1 or norm_position[1]<0 or norm_position[1]>1 or norm_position[2]<0 or norm_position[2]>1: 
				return False, -1
			else:
				if self._nParti==1 and self._nPartj==1 and self._nPartk==1 : return True, -1
				else:										
					i = int(norm_position[0]*self._nParti) #Determines the intdex position within the discrete box from the normalized coordinate
					j = int(norm_position[1]*self._nPartj)
					k = int(norm_position[2]*self._nPartk)

					if i==self._nParti: i = self._nParti-1
					if j==self._nPartj: j = self._nPartj-1
					if k==self._nPartk: k = self._nPartk-1

					idx = self._getIndex(i,j,k) 

					return True, idx

	def isInside(self,point):

		if np.isnan(point[0]) or self._illDefined:
			return False
		else:
			norm_position = np.array([(point[0]-self._pmin[0])/(self._pmax[0]-self._pmin[0]),(point[1]-self._pmin[1])/(self._pmax[1]-self._pmin[1]),(point[2]-self._pmin[2])/(self._pmax[2]-self._pmin[2])])

			if norm_position[0]<0 or norm_position[0]>1 or norm_position[1]<0 or norm_position[1]>1 or norm_position[2]<0 or norm_position[2]>1: 
				return False
			else:
				if self._nParti==1 and self._nPartj==1 and self._nPartk==1: return True
				else:
					i = int(norm_position[0]*self._nParti) #Determines the intdex position within the discrete box from the normalized coordinate
					j = int(norm_position[1]*self._nPartj)
					k = int(norm_position[2]*self._nPartk)

					if i==self._nParti: i = self._nParti-1
					if j==self._nPartj: j = self._nPartj-1
					if k==self._nPartk: k = self._nPartk-1

					idx = self._getIndex(i,j,k) 

					return self._isIn[idx]

	def discretize_by_parts(self,nParti,nPartj,nPartk,points):
		
		if not self._illDefined:
			self._nParti = nParti
			self._nPartj = nPartj
			self._nPartk = nPartk
			self._isIn   = [False]*(self._nParti*self._nPartj*self._nPartk)

			for point in points:

				isInside ,idx = self._isInside(point)
				if isInside: self._isIn[idx] = True


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

			'''
			if mpi_rank==3:
				for idx,i in enumerate(self._isIn):
					print(f"idx={idx} value={i}")
			'''

	def areInside(self,points):

		idxPoints=[]

		if not self._illDefined:

			for idx,p in enumerate(points):

				#print("idx=",idx," point=",p)

				isInside = self.isInside(p)

				if isInside:
					idxPoints.append(idx)

		return idxPoints














