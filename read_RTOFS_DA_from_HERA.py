#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:05:25 2020

@author: aristizabal
"""
#%% User input
#cycle = '2020042500'
cycle = '2020042400'

# folder ab files HYCOM
folder_RTOFS_DA = '/scratch2/NOS/nosofs/Maria.Aristizabal/RTOFS-DA/data_' + cycle
prefix_RTOFS_DA = 'archv'

# RTOFS grid file name
folder_RTOFS_DA_grid_depth = '/scratch2/NOS/nosofs/Maria.Aristizabal/RTOFS-DA/GRID_DEPTH/'
RTOFS_DA_grid = folder_RTOFS_DA_grid_depth + 'regional.grid'
RTOFS_DA_depth = folder_RTOFS_DA_grid_depth + 'regional.depth'

#%% 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import os.path
import glob

import sys
sys.path.append('/home/Maria.Aristizabal/NCEP_scripts/')
#from utils4HYCOM_orig import readBinz 
from utils4HYCOM import readgrids,readdepth, readVar

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading RTOFS grid
print('Retrieving coordinates from RTOFS')
# Reading lat and lon
lines_grid = [line.rstrip() for line in open(RTOFS_DA_grid+'.b')]
lon_RTOFS_DA = np.array(readgrids(RTOFS_DA_grid,'plon:',[0]))
lat_RTOFS_DA = np.array(readgrids(RTOFS_DA_grid,'plat:',[0]))

depth_RTOFS_DA = np.asarray(readdepth(RTOFS_DA_depth,'depth'))

#%%
'''
N = 0
var_name = 'temp'

afiles = sorted(glob.glob(os.path.join(folder_RTOFS_DA,prefix_RTOFS_DA+'*.a')))
file = afiles[N]

# Reading 3D variable from binary file 
var_rtofs = readBinz(file[:-2],'3z',var_name)
temp_RTOFS_DA = var_rtofs[:,:,0]

#Reading time stamp
year = int(file.split('/')[-1].split('.')[1][0:4])
month = int(file.split('/')[-1].split('.')[1][4:6])
day = int(file.split('/')[-1].split('.')[1][6:8])
hour = int(file.split('/')[-1].split('.')[1][8:10])
dt = int(file.split('/')[-1].split('.')[3][1:])
timestamp_RTOFS_DA = mdates.date2num(datetime(year,month,day,hour)) + dt/24
time_RTOFS_DA = mdates.num2date(timestamp_RTOFS_DA)
'''
#%%
'''
#kw = dict(levels = np.linspace(28,30,41))
plt.figure()
plt.contourf(lon_RTOFS_DA[0,:],lat_RTOFS_DA[:,0],temp_RTOFS_DA,cmap='RdYlBu_r') #,**kw)
plt.colorbar()
plt.title('RTOFS-DA SST \n on '+str(time_RTOFS_DA)[0:13],fontsize=14)
plt.axis('scaled')
'''
#%% Reading RTOFS ab files
'''
var_name = 'temp'
afiles = sorted(glob.glob(os.path.join(folder_RTOFS_DA,prefix_RTOFS_DA+'*.a')))
nz = 41
layers = np.arange(0,nz)
target_temp_RTOFS = np.empty((len(afiles),nz))
target_temp_RTOFS[:] = np.nan
target_zRTOFS = np.empty((len(afiles),nz))
target_zRTOFS[:] = np.nan
time_RTOFS = []

timestampg = mdates.date2num([datetime(2020,4,25,2),datetime(2020,4,25,3)])
target_lon = [290,290.2]
target_lat = [40,40.1]

for tt,file in enumerate(afiles):
    print(file)
    file = afiles[0]
    lines = [line.rstrip() for line in open(file[:-2]+'.b')]
    time_stamp = lines[-1].split()[2]
    hycom_days = lines[-1].split()[3]
    tzero=datetime(1901,1,1,0,0)
    timeRT = tzero+timedelta(float(hycom_days)-1)
    time_RTOFS.append(timeRT)
    timestampRTOFS = mdates.date2num(timeRT)
    
    sublonRTOFS = np.interp(timestampRTOFS,timestampg,target_lon)
    sublatRTOFS = np.interp(timestampRTOFS,timestampg,target_lat)
    oklonRTOFS = np.int(np.round(np.interp(sublonRTOFS,lon_RTOFS_DA[0,:],np.arange(len(lon_RTOFS_DA[0,:])))))
    oklatRTOFS = np.int(np.round(np.interp(sublatRTOFS,lat_RTOFS_DA[:,0],np.arange(len(lat_RTOFS_DA[:,0])))))
    
    #temp_RTOFS = readBinz(file[:-2],'3z',var_name)
    #target_temp_RTOFS[tt,:] = temp_RTOFS[oklatRTOFS,oklonRTOFS,:]
    
    ztmp=readVar(file[:-2],'archive','srfhgt',[0])*0.01 # converts [cm] to [m]
    target_ztmp = ztmp[oklatRTOFS,oklonRTOFS]
    for lyr in tuple(layers):
        print(lyr)
        temp_RTOFS = readVar(file[:-2],'archive',var_name,[lyr+1])
        target_temp_RTOFS[tt,lyr] = temp_RTOFS[oklatRTOFS,oklonRTOFS]

        dp=readVar(file[:-2],'archive','thknss',[lyr+1])/2/9806
        target_ztmp = np.append(target_ztmp,dp[oklatRTOFS,oklonRTOFS])

    target_z3d = np.cumsum(target_ztmp)              # [idm,jdm,kdm+1]
    target_z3d = np.squeeze(target_z3d[1:])             # [idm,jdm,kdm]
    target_z3d = np.array(target_z3d)
    target_z3d[target_z3d > 10**8] = np.nan
    target_zRTOFS[tt,:] = target_z3d

time_RTOFS = np.asarray(time_RTOFS)
'''

#%% Reading RTOFS ab files

var_name = 'temp'
afiles = sorted(glob.glob(os.path.join(folder_RTOFS_DA,prefix_RTOFS_DA+'*.a')))
nz = 41
layers = np.arange(0,nz)
target_temp_RTOFS = np.empty((len(afiles),nz))
target_temp_RTOFS[:] = np.nan
target_zRTOFS = np.empty((len(afiles),nz))
target_zRTOFS[:] = np.nan
time_RTOFS = []

timestampg = mdates.date2num([datetime(2020,4,24,0),datetime(2020,4,24,3)])
tstamp_glider = timestampg

#target_lon = long + 360
#target_lat = latg

target_lon = np.tile(270,len(tstamp_glider))
target_lat = np.tile(25,len(tstamp_glider))

for tt,file in enumerate(afiles[0:3]):
    print(file)
    #file = afiles[0]
    lines = [line.rstrip() for line in open(file[:-2]+'.b')]
    time_stamp = lines[-1].split()[2]
    hycom_days = lines[-1].split()[3]
    tzero=datetime(1901,1,1,0,0)
    timeRT = tzero+timedelta(float(hycom_days)-1)
    time_RTOFS.append(timeRT)
    timestampRTOFS = mdates.date2num(timeRT)
    
    sublonRTOFS = np.interp(timestampRTOFS,tstamp_glider,target_lon)
    sublatRTOFS = np.interp(timestampRTOFS,tstamp_glider,target_lat)
    oklonRTOFS = np.int(np.round(np.interp(sublonRTOFS,lon_RTOFS_DA[0,:],np.arange(len(lon_RTOFS_DA[0,:])))))
    oklatRTOFS = np.int(np.round(np.interp(sublatRTOFS,lat_RTOFS_DA[:,0],np.arange(len(lat_RTOFS_DA[:,0])))))
    
    #temp_RTOFS = readBinz(file[:-2],'3z',var_name)
    #target_temp_RTOFS[tt,:] = temp_RTOFS[oklatRTOFS,oklonRTOFS,:]
    #ztmp=readVar(file[:-2],'archive','srfhgt',[0])*0.01 # converts [cm] to [m]
    #target_srfhgt = ztmp[oklatRTOFS,oklonRTOFS]  
    
    target_ztmp = [0]
    for lyr in tuple(layers):
        print(lyr)
        temp_RTOFS = readVar(file[:-2],'archive',var_name,[lyr+1])
        target_temp_RTOFS[tt,lyr] = temp_RTOFS[oklatRTOFS,oklonRTOFS]
        
        dp = readVar(file[:-2],'archive','thknss',[lyr+1])/9806
        target_ztmp = np.append(target_ztmp,dp[oklatRTOFS,oklonRTOFS])
        
    target_zRTOFS[tt,:] = np.cumsum(target_ztmp[0:-1]) + np.diff(np.cumsum(target_ztmp))/2    

time_RTOFS = np.asarray(time_RTOFS)
#%%

plt.figure()
plt.plot(target_temp_RTOFS[0,:],-target_zRTOFS[0,:],'.-')
