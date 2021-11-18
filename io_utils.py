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
	asciiFile.write(('{:_<8}'.format('XFIEL00\0'))) # Object
	asciiFile.write(('{:_<8}'.format('VECTO00\0'))) # Dimension 
	asciiFile.write(('{:_<8}'.format('NPOIN00\0'))) # Results On
	asciiFile.write(('{:_<8}'.format('REAL000\0'))) # Type
	asciiFile.write(('{:_<8}'.format('8BYTE00\0'))) # Size
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






