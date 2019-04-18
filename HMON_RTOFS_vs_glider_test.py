#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:10:25 2019

@author: aristizabal
"""
import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')

from utils4HYCOM import readBinz, readgrids, readBin

import os
import os.path
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
import numpy as np

import netCDF4
import time
import matplotlib.dates as mdates

import xarray as xr

#%% User input

# Directories where RTOFS files reside 
Dir= '/Volumes/aristizabal/ncep_model/HMON-HYCOM_Michael/'
Dir_graph = '/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts/Figures/'

# RTOFS grid file name
gridfile = 'hwrf_rtofs_hat10.basin.regional.grid'

# RTOFS a/b file name
#prefix_ab = 'fourteen14l.2018100700.hmon_basin'
prefix_ab = 'michael14l.2018100718.hmon_rtofs_hat10_'
# Name of 3D variable
var_name = 'temp'

# Glider data 

# ng288
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

# date limits
date_ini = '2018-10-07T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)
tempglider = np.array(ncglider.temperature[0,:,:])
saltglider = np.array(ncglider.salinity[0,:,:])
depthglider = np.array(ncglider.depth[0,:,:])
#densglider = ncglider.density[0,:,:]

timestamp_glider = []
for t in time_glider[0,:]:
    timestamp_glider.append(time.mktime(t.timetuple()))
    
timestamp_glider = np.array(timestamp_glider)
   
# Conversion from glider longitude and latitude to RTOFS convention
target_lon = []
for lon in longlider[0,:]:
    if lon < 0: 
        target_lon.append(360 + lon)
    else:
        target_lon.append(lon)
target_lon = np.array(target_lon)
target_lat = np.array(latglider[0,:])

#%%

# Reading lat and lon
lines_grid=[line.rstrip() for line in open(Dir+gridfile+'.b')]
hlon = np.array(readgrids(Dir+gridfile,'plon:',[0]))
hlat = np.array(readgrids(Dir+gridfile,'plat:',[0]))

# Extracting the longitudinal and latitudinal size array
idm=int([line.split() for line in lines_grid if 'longitudinal' in line][0][0])
jdm=int([line.split() for line in lines_grid if 'latitudinal' in line][0][0])

afiles = sorted(glob.glob(os.path.join(Dir,prefix_ab+'*.a')))
# Note: the 3D output is 6 hourly. therefore half of the a files only contain 3D output
#afiles_6h = afiles[::2] 
nz = 41

target_temp_RTOFS = np.empty((len(afiles),nz,))
target_temp_RTOFS[:] = np.nan
target_thknss_RTOFS = np.empty((len(afiles),nz,))
target_thknss_RTOFS[:] = np.nan
time_RTOFS = []
dens = np.empty((len(afiles),nz,))
dens[:] = np.nan

#for x, file in enumerate(afiles_6h):
x = 0
file = afiles[x]
print(x)
lines = [line.rstrip() for line in open(file[:-2]+'.b')]

#Reading time stamp
'''
time_stamp = lines[-1].split()[2]
hycom_days = lines[-1].split()[3]
tzero = datetime(1901,1,1,0,0)
time_RT = tzero+timedelta(float(hycom_days))
time_RTOFS.append(time_RT)
timestamp_RTOFS = time.mktime(time_RT.timetuple())
'''
year = int(file.split('.')[1][0:4])
month = int(file.split('.')[1][4:6])
day = int(file.split('.')[1][6:8])
hour = int(file.split('.')[1][8:10])
dt = int(file.split('.')[3][1:])
timestamp_RTOFS= date2num(datetime(year,month,day,hour)) + date2num(datetime(1,1,1,dt))-1

# Reading depths
z = []
for line in lines[6:]:
    if line.split()[2]==var_name:
        #print(line.split()[1])
        z.append(float(line.split()[1]))
z_RTOFS = np.asarray(z)        
    
# Interpolating latglider and longlider into RTOFS grid
sublonRTOFS = np.interp(timestamp_RTOFS,timestamp_glider,target_lon)
sublatRTOFS = np.interp(timestamp_RTOFS,timestamp_glider,target_lat)
oklonRTOFS = np.int(np.round(np.interp(sublonRTOFS,hlon[0,:],np.arange(len(hlon[0,:])))))
oklatRTOFS = np.int(np.round(np.interp(sublatRTOFS,hlat[:,0],np.arange(len(hlat[:,0])))))
    
# Reading 3D variable from binary file 
temp_RTOFS = readBinz(file[:-2],'3z',var_name)
target_temp_RTOFS[x,:] = temp_RTOFS[oklatRTOFS,oklonRTOFS,:]
    
depths=[1,3,5,7.5,10,15,20,25,30,40,50,60,70,80,100,120,140,160,180,200,220,240,250,275,300,350]
