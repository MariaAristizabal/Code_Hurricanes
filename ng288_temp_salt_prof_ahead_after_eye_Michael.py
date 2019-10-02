#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:39:26 2019

@author: aristizabal
"""
#%%
# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

# date limits
date_ini = '2018-10-09T00:00:00Z'
date_end = '2018-10-11T00:00:00Z'

# Glider data 

# ng288
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

#%%

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.dates import date2num
import xarray as xr
import netCDF4

# Increase fontsize of labels globally 
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('legend',fontsize=18)

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

#%%

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
saltg_gridded = np.empty((len(timeg),len(depthg_gridded)))
saltg_gridded[:] = np.nan

for t,tt in enumerate(timeg):
    depthu,oku = np.unique(depthg[t,:],return_index=True)
    tempu = tempg[t,oku]
    saltu = saltg[t,oku]
    okdd = np.isfinite(depthu)
    depth_fin = depthu[okdd]
    temp_fin = tempu[okdd]
    salt_fin = saltu[okdd]
    ok = np.isfinite(temp_fin)
    
    if np.sum(ok) < 3:
        tempg_gridded[t,:] = np.nan
        saltg_gridded[t,:] = np.nan
    else:
        okd = depthg_gridded < np.max(depth_fin[ok])
        tempg_gridded[t,okd] = np.interp(depthg_gridded[okd],depth_fin[ok],temp_fin[ok]) 
        saltg_gridded[t,okd] = np.interp(depthg_gridded[okd],depth_fin[ok],salt_fin[ok]) 

#%% Get rid off of profiles with no data below 100 m

tempg_full = []
saltg_full = []
timeg_full = []
for t,tt in enumerate(timeg):
    okt = np.isfinite(tempg_gridded[t,:])
    if sum(depthg_gridded[okt] > 100) > 10:
        if tempg_gridded[t,0] != tempg_gridded[t,20]:
            tempg_full.append(tempg_gridded[t,:]) 
            saltg_full.append(saltg_gridded[t,:]) 
            timeg_full.append(tt) 
                   
tempg_full = np.asarray(tempg_full)
saltg_full = np.asarray(saltg_full)
timeg_full = np.asarray(timeg_full)

#%%

twindow1 = np.logical_and(timeg_full >= tmin,timeg_full <= timeg[69])
twindow2 = np.logical_and(timeg_full > timeg[69],timeg_full <= timeg[-1])

#%% 

fig,ax = plt.subplots(figsize=(9,10))

ax1 = plt.subplot(2,1,1)
plt.plot(tempg_full[0,:].T,-depthg_gridded,'.-b',\
         markersize=10,label='Ahead of eye center '+\
         str(timeg[0])[5:16] + ' - ' +\
         str(timeg[69])[5:16])
plt.plot(tempg_full[-2,:].T,-depthg_gridded,'.-r',\
         markersize=10,label='After eye center '+\
         str(timeg[70])[5:16] + ' - ' +\
         str(timeg[-1])[5:16])
plt.plot(tempg_full[twindow1,:].T,-depthg_gridded,'.b',\
         markersize=1)
plt.plot(tempg_full[twindow2,:].T,-depthg_gridded,'.r',markersize=1)
plt.title('ng288',size = 24)
plt.xlabel('Temperature ($^oc$)',fontsize=18)
plt.ylabel('Depth (m)',fontsize=18)
plt.grid(True)
plt.legend()
plt.text(17,17,'(a)',fontsize=18)

plt.subplot(2,1,2)
plt.plot(saltg_full[twindow1,:].T,-depthg_gridded,'.b',markersize=1)
plt.plot(saltg_full[twindow2,:].T,-depthg_gridded,'.r',markersize=1)
plt.xlabel('Salinity',fontsize=18)
plt.ylabel('Depth (m)',fontsize=18)
plt.grid(True)
plt.text(36,17,'(b)',fontsize=18)

plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9,hspace=0.3)

plt.savefig(folder + 'ng288_prof_ahead_before_eye',\
            bbox_inches = 'tight',pad_inches = 0.1) 