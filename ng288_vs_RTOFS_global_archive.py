#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:46:02 2019

@author: aristizabal
"""

# files for global RTOFS output
Dir_rtofs= '/Volumes/aristizabal/ncep_model/rtofs.20181014/'

# RTOFS grid file name same as GOFS 3.1
url_GOFS31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# RTOFS a/b file name
prefix_ab = 'rtofs_glo.t00z.n-48.archv'

# Name of 3D variable
var_name = 'temp'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'


#%%
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 
from datetime import datetime
from matplotlib.dates import date2num, num2date
import matplotlib.dates as mdates

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')
from utils4HYCOM import readBinz, readgrids, readVar

import os
import os.path
import glob

#%% Reading RTOFS lat and lon (same as GOFS 3.1)

GOFS31 = xr.open_dataset(url_GOFS31,decode_times=False)

latRTOFS = GOFS31['lat'][:]
lonRTOFS = GOFS31['lon'][:]

#depth31 = GOFS31_ts['depth'][:]

#%% Reading RTOFS ab files

afiles = sorted(glob.glob(os.path.join(Dir_rtofs,prefix_ab+'*.a')))

# reading variable
temp_rtofs = readVar(afiles[0][:-2],'archive',var_name,[0,1])

#%%

# Reading depths
lines=[line.rstrip() for line in open(afiles[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]==var_name:
        #print(line.split()[1])
        z.append(float(line.split()[1]))
z_HMON_HYCOM = np.asarray(z) 

nz = len(z_HMON_HYCOM) 

target_temp_HMON_HYCOM = np.empty((len(afiles),nz,))
target_temp_HMON_HYCOM[:] = np.nan
time_HMON_HYCOM = []
for x, file in enumerate(afiles):
    print(x)
    lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    year = int(file.split('.')[1][0:4])
    month = int(file.split('.')[1][4:6])
    day = int(file.split('.')[1][6:8])
    hour = int(file.split('.')[1][8:10])
    dt = int(file.split('.')[3][1:])
    timestamp_HMON_HYCOM = date2num(datetime(year,month,day,hour)) + dt/24
    time_HMON_HYCOM.append(num2date(timestamp_HMON_HYCOM))
    
    # Interpolating latg and longlider into RTOFS grid
    #sublonHMON_HYCOM = np.interp(timestamp_HMON_HYCOM,timestampg,target_long)
    #sublatHMON_HYCOM = np.interp(timestamp_HMON_HYCOM,timestampg,target_latg)
    oklonHMON_HYCOM = np.int(np.round(np.interp(target_lonG,hlon[0,:],np.arange(len(hlon[0,:])))))
    oklatHMON_HYCOM = np.int(np.round(np.interp(target_latG,hlat[:,0],np.arange(len(hlat[:,0])))))
    
    # Reading 3D variable from binary file 
    temp_HMON_HYCOM = readBinz(file[:-2],'3z',var_name)
    #ts=readBin(afile,'archive','temp')
    target_temp_HMON_HYCOM[x,:] = temp_HMON_HYCOM[oklatHMON_HYCOM,oklonHMON_HYCOM,:]
    
time_HMON_HYCOM = np.asarray(time_HMON_HYCOM)
timestamp_HMON_HYCOM = date2num(time_HMON_HYCOM) 

#%% 
# read in "thknss" from archv*.[ab] and convert it to depth [m] in 3-D array

ztmp=readVar(fname,'archive','srfhgt',[0])*0.01 # converts [cm] to [m]
for lyr in tuple(layers):
    dp=readVar(fname,'archive','thknss',[lyr])/2/9806
    ztmp=ma.dstack((ztmp,dp))

z3d=ma.cumsum(ztmp,axis=2)              # [idm,jdm,kdm+1]
z3d=ma.squeeze(z3d[:,:,1:])             # [idm,jdm,kdm]
z3d=np.array(z3d)