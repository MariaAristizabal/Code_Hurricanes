#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:46:05 2019

@author: aristizabal
"""

#%% User input

mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU22_GOFS31.mat'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# date limits
date_ini = '2018-08-15T00:00:00Z'
date_end = '2018-08-25T00:00:00Z'

# Glider data 

# ng288
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc'

#%%

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.dates import date2num
import xarray as xr
import netCDF4

import scipy.io as sio
RU22 = sio.loadmat(mat_file)
RU22.keys()

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
        
#%% Get rid off of profiles with no data below 10 m

dc = 10
tempg_full = []
saltg_full = []
timeg_full = []
for t,tt in enumerate(timeg):
    okt = np.isfinite(tempg_gridded[t,:])
    if sum(depthg_gridded[okt] > dc) > 10:
        if tempg_gridded[t,0] != tempg_gridded[t,20]:
            tempg_full.append(tempg_gridded[t,:])
            saltg_full.append(saltg_gridded[t,:])
            timeg_full.append(tt)
             
tempg_full = np.asarray(tempg_full)
saltg_full = np.asarray(saltg_full)
timeg_full = np.asarray(timeg_full)

#%% load GOFS 3.1 

tstamp31 =  RU22['timem'][:,0]
depth31 = RU22['depth31'][:,0]
tem31 = RU22['tempm'][:]
sal31 = RU22['salm'][:]

#%% Changing timestamps to datenum

tim31 = []
for i in np.arange(len(tstamp31)):
    tim31.append(datetime.fromordinal(int(tstamp31[i])) + \
        timedelta(days=tstamp31[i]%1) - timedelta(days = 366))
tt31 = np.asarray(tim31)

tti = datetime.strptime(date_ini, '%Y-%m-%dT%H:%M:%SZ')
tte = datetime.strptime(date_end, '%Y-%m-%dT%H:%M:%SZ')

oktime31 = np.logical_and(tt31>=tti, tt31<=tte)

time31 = tt31[oktime31]
timestamp31 = date2num(time31)
temp31 = tem31[:,oktime31]
salt31 = sal31[:,oktime31]

#%% Figure GOFS 3.1 temperature

time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_temp_GOFS = temp31.T

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),21))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_SOULIK.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% Figure GOFS 3.1 temperature with incremental insertion window

time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_temp_GOFS = temp31.T

dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),21))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

patch = patches.Rectangle((date2num(datetime(2018,8,17,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_SOULIK2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% Figure GOFS 3.1 salinity with incremental insertion window

time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_salt_GOFS = salt31.T

dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

kw = dict(levels = np.linspace(31,34.6,19))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_salt_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_salt_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_salt_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Salinity',fontsize=16)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

patch = patches.Rectangle((date2num(datetime(2018,8,17,9)),-100),dt,100,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_salt_SOULIK2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% glider profile temperature with less gaps

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),21))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
#plt.contour(timeg_full,-depthg_gridded,tempg_full.T,colors = 'lightgrey',**kw)
plt.contour(timeg_full,-depthg_gridded,tempg_full.T,[26],colors = 'k')
plt.contourf(timeg_full,-depthg_gridded,tempg_full.T,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('RU22',fontsize=20)
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU22_SOULIK_vs_depth2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

plt.show()

#%% glider profile salinity with less gaps

kw = dict(levels = np.linspace(31,34.6,19))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(timeg_full,-depthg_gridded,saltg_full.T,colors = 'lightgrey',**kw)
plt.contour(timeg_full,-depthg_gridded,saltg_full.T,[26],colors = 'k')
plt.contourf(timeg_full,-depthg_gridded,saltg_full.T,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Salinity',fontsize=16)
ax.set_title('RU22',fontsize=20)
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU22_SOULIK_vs_depth_salt.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

plt.show()

#%% Time series at 10 m

d = 10
nz31 = np.where(depth31 >= d)[0][0]
nzg = np.where(depthg_gridded >= d)[0][0]
dt = date2num(datetime(2018,8,23,12)) - date2num(datetime(2018,8,23,9))

fig, ax = plt.subplots(figsize=(8.8, 1.7))
plt.plot(timeg_full,saltg_full[:,nzg],'o-',color='royalblue',label='ng288')
plt.plot(time31,target_salt_GOFS[:,nz31],'o-g',label='GOFS 3.1')
plt.legend(fontsize=14)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(np.arange(17,30,0.1))),\
         np.arange(17,30,0.1),'--k')
patch = patches.Rectangle((date2num(datetime(2018,8,17,9)),31),dt,4,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)

#ax.set_ylabel('($^\circ$C)',fontsize=16)
ax.set_title('Time Series Salinity at 10 m',fontsize=20)
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(31,34.6)

xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/salt_time_series_RU22_GOFS31_10m.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 





