#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:46:02 2019

@author: aristizabal
"""

# files for global RTOFS output
Dir_rtofs= '/Volumes/aristizabal/ncep_model/RTOFS_global_Michael/rtofs.'

# RTOFS grid file name
gridfile = '/Volumes/aristizabal/ncep_model/RTOFS_global_Michael/rtofs_glo.navy_0.08.regional.depth'

# RTOFS a/b file name
prefix_ab = 'rtofs_glo.t00z.n-48.archv'

# Name of 3D variable
var_name = 'temp'

# ng288
gdata = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc'

# date limits
date_ini = '2018-10-07T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 
from datetime import datetime, timedelta
from matplotlib.dates import date2num, num2date
import matplotlib.dates as mdates

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')
from utils4HYCOM import readBinz, readgrids, readVar, parse_z, parse_b, readBin

import os
import os.path
import glob

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)
tempglider = np.array(ncglider.temperature[0,:,:])
saltglider = np.array(ncglider.salinity[0,:,:])
depthglider = np.array(ncglider.depth[0,:,:])

timestamp_glider = date2num(time_glider)[0]

tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

okg = np.where(np.logical_and(time_glider.T >= tmin, time_glider.T <= tmax))

timeg = time_glider[0,okg[0]]
timestampg = timestamp_glider[okg[0]]
latg = np.asarray(latglider[0,okg[0]])
long = np.asarray(longlider[0,okg[0]])
depthg = depthglider[okg[0],:]
tempg = tempglider[okg[0],:]
saltg = saltglider[okg[0],:]

#%% Grid glider variables according to depth

depthg_gridded = np.arange(0,np.nanmax(depthg),0.5)
tempg_gridded = np.empty((len(timeg),len(depthg_gridded)))
tempg_gridded[:] = np.nan

for t,tt in enumerate(timeg):
    depthu,oku = np.unique(depthg[t,:],return_index=True)
    tempu = tempg[t,oku]
    okdd = np.isfinite(depthu)
    depth_fin = depthu[okdd]
    temp_fin = tempu[okdd]
    ok = np.isfinite(temp_fin)
    
    if np.sum(ok) < 3:
        tempg_gridded[t,:] = np.nan
    else:
        okd = depthg_gridded < np.max(depth_fin[ok])
        tempg_gridded[t,okd] = np.interp(depthg_gridded[okd],depth_fin[ok],temp_fin[ok]) 
        
#%% Get rid off of profiles with no data below 100 m

tempg_full = []
timeg_full = []
for t,tt in enumerate(timeg):
    okt = np.isfinite(tempg_gridded[t,:])
    if sum(depthg_gridded[okt] > 100) > 10:
        if tempg_gridded[t,0] != tempg_gridded[t,20]:
            tempg_full.append(tempg_gridded[t,:]) 
            timeg_full.append(tt) 
       
tempg_full = np.asarray(tempg_full)
timeg_full = np.asarray(timeg_full)

#%%  Reading RTOFS grid

# Reading lat and lon
lines_grid=[line.rstrip() for line in open(gridfile+'.b')]
hlon = np.array(readgrids(gridfile,'relax'))

hlon = np.array(readgrids(gridfile,'plon:',[0]))
hlat = np.array(readgrids(gridfile,'plat:',[0]))

# Extracting the longitudinal and latitudinal size array
idm=int([line.split() for line in lines_grid if 'longitudinal' in line][0][0])
jdm=int([line.split() for line in lines_grid if 'latitudinal' in line][0][0])

#%% Reading RTOFS ab files
    
nz = 41
layers = np.arange(0,nz)
target_temp_RTOFS = np.empty(((tmax-tmin).days,nz,))
target_temp_RTOFS[:] = np.nan
target_zRTOFS = np.empty(((tmax-tmin).days,nz,))
target_zRTOFS[:] = np.nan
timeRTOFS = []
for tt in np.arange(0,(tmax-tmin).days):
    print(tt)
    t = tmin+timedelta(np.int(tt))
    if t.day < 9: 
        afile = sorted(glob.glob(os.path.join(Dir_rtofs+str(tmin.year)+str(tmin.month)+'0'+str(tmin.day+1+tt)+'/',prefix_ab+'*.a')))
    else:
        afile = sorted(glob.glob(os.path.join(Dir_rtofs+str(tmin.year)+str(tmin.month)+str(tmin.day+1+tt)+'/',prefix_ab+'*.a')))
      
    lines = [line.rstrip() for line in open(afile[0][:-2]+'.b')]    
    time_stamp = lines[-1].split()[2]
    hycom_days = lines[-1].split()[3]
    tzero=datetime(1901,1,1,0,0)
    timeRT = tzero+timedelta(float(hycom_days))
    timeRTOFS.append(timeRT)
    timestampRTOFS = date2num(timeRT) 

    oklonRTOFS = 2508
    oklatRTOFS = 1858
    
    # Interpolating latg and long into RTOFS grid
    #sublonRTOFS = np.interp(timestampRTOFS,timestamp_glider,target_lon)
    #sublatRTOFS = np.interp(timestampRTOFS,timestamp_glider,target_lat)
    #oklonRTOFS = np.int(np.round(np.interp(sublonRTOFS,hlon[0,:],np.arange(len(hlon[0,:])))))
    #oklatRTOFS = np.int(np.round(np.interp(sublatRTOFS,hlat[:,0],np.arange(len(hlat[:,0])))))
    
    ztmp=readVar(afile[0][:-2],'archive','srfhgt',[0])*0.01 # converts [cm] to [m]
    target_ztmp = ztmp[oklatRTOFS,oklonRTOFS]
    for lyr in tuple(layers):
        print(lyr)
        temp_RTOFS = readVar(afile[0][:-2],'archive',var_name,[lyr+1])
        target_temp_RTOFS[tt,lyr] = temp_RTOFS[oklatRTOFS,oklonRTOFS]
        
        dp=readVar(afile[0][:-2],'archive','thknss',[lyr+1])/2/9806
        
        target_ztmp = np.append(target_ztmp,dp[oklatRTOFS,oklonRTOFS])
        
    target_z3d=np.cumsum(target_ztmp)              # [idm,jdm,kdm+1]
    target_z3d=np.squeeze(target_z3d[1:])             # [idm,jdm,kdm]
    target_z3d=np.array(target_z3d)
    target_z3d[target_z3d > 10**8] = np.nan
    target_zRTOFS[tt,:] = target_z3d
    
timeRTOFS = np.asarray(timeRTOFS)
           
#%%
'''    
# read in "thknss" from archv*.[ab] and convert it to depth [m] in 3-D array

layers = np.arange(0,41)
ztmp=readVar(afiles[0][:-2],'archive','srfhgt',[0])*0.01 # converts [cm] to [m]
for lyr in tuple(layers):
    print(lyr)
    dp=readVar(afiles[0][:-2],'archive','thknss',[lyr+1])/2/9806
    ztmp=np.dstack((ztmp,dp))

z3d=np.cumsum(ztmp,axis=2)              # [idm,jdm,kdm+1]
z3d=np.squeeze(z3d[:,:,1:])             # [idm,jdm,kdm]
z3d=np.array(z3d)
z3d[z3d > 10**8] = np.nan
'''

#%% Figure

time_matrixRTOFS = np.transpose(np.tile(timeRTOFS,(nz,1)))

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(10, 3))
plt.contour(time_matrixRTOFS,-1*target_zRTOFS,target_temp_RTOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrixRTOFS,-1*target_zRTOFS,target_temp_RTOFS,[26],colors = 'k')
plt.contourf(time_matrixRTOFS,-1*target_zRTOFS,target_temp_RTOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),nz),-1*target_zRTOFS[0,:],'--k')
ax.set_ylim(-250,0)
ax.set_xlim(datetime(2018,10,7),datetime(2018,10,13))
yl = ax.set_ylabel('Depth (m)',fontsize=16,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('RTOFS Temperature',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/global_RTOFS_temp_Michael.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(10, 3))
plt.contour(timeg,-depthg_gridded,tempg_gridded.T,colors = 'lightgrey',**kw)
plt.contour(timeg,-depthg_gridded,tempg_gridded.T,[26],colors = 'k')
plt.contourf(timeg,-depthg_gridded,tempg_gridded.T,cmap='RdYlBu_r',**kw)
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('ng288',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng288_Michael_vs_depth.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

