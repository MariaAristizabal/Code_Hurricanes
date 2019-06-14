#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:44:36 2019

@author: aristizabal
"""

#%% User input

mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng288.mat'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

# date limits
date_ini = '2018-10-05T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'

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

import scipy.io as sio
ng288 = sio.loadmat(mat_file)

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

#%% load GOFS 3.1 

tstamp31 =  ng288['timeGOFS31'][:,0]
depth31 = ng288['depthGOFS31'][:,0]
tem31 = ng288['tempGOFS31'][:]

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

#%% Figure GOFS 3.1 

time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_temp_GOFS = temp31.T

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,10,5),datetime(2018,10,13))
ax.set_ylim(-260,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.tick_params(labelsize=14)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_Michael.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% Figure GOFS 3.1 with incremental insertion window

time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_temp_GOFS = temp31.T

dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,10,5),datetime(2018,10,13))
ax.set_ylim(-260,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.tick_params(labelsize=14)
plt.tick_params(labelsize=14)

patch = patches.Rectangle((date2num(datetime(2018,10,5,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,6,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,7,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,8,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,9,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,10,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,11,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,12,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_Michael2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% glider profile with less gaps

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(timeg_full,-depthg_gridded,tempg_full.T,colors = 'lightgrey',**kw)
plt.contour(timeg_full,-depthg_gridded,tempg_full.T,[26],colors = 'k')
plt.contourf(timeg_full,-depthg_gridded,tempg_full.T,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_title('ng288',fontsize=20)
ax.set_xlim(datetime(2018,10,5),datetime(2018,10,13))
ax.set_ylim(-260,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.tick_params(labelsize=14)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng288_Michael_vs_depth2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

plt.show()

#%% Time series at 10 m

d = 10
nz31 = np.where(depth31 >= d)[0][0]
nzg = np.where(depthg_gridded >= d)[0][0]
dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.plot(timeg_full,tempg_full[:,nzg],'o-',color='royalblue',label='ng288')
plt.plot(time31,target_temp_GOFS[:,nz31],'o-g',label='GOFS 3.1')
plt.legend(fontsize=14)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(np.arange(27.5,29.5,0.1))),\
         np.arange(27.5,29.5,0.1),'--k')
patch = patches.Rectangle((date2num(datetime(2018,10,5,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,6,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,7,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,8,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,9,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,10,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,11,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,12,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)

ax.set_ylabel('($^\circ$C)',fontsize=16)
ax.set_title('Time Series Temperature at 10 m',fontsize=20)
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
#cbar.ax.tick_params(labelsize=14)
plt.tick_params(labelsize=14)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_time_series_ng288_GOFS31_10m.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Time series at 100 m

d = 100
nz31 = np.where(depth31 >= d)[0][0]
nzg = np.where(depthg_gridded >= d)[0][0]
dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

fig, ax = plt.subplots(figsize=(8.8, 1.7))
plt.plot(timeg_full,tempg_full[:,nzg],'o-',color='royalblue',label='ng288')
plt.plot(time31,target_temp_GOFS[:,nz31],'o-g',label='GOFS 3.1')
plt.legend(fontsize=14,bbox_to_anchor = [0.45, 0.6])
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(np.arange(22,27.5,0.1))),\
         np.arange(22,27.5,0.1),'--k')

patch = patches.Rectangle((date2num(datetime(2018,10,5,9)),21.6),dt,270,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,6,9)),21.5),dt,270,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,7,9)),21.5),dt,270,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,8,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,9,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,10,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,11,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,12,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)

ax.set_ylabel('($^\circ$C)',fontsize=16)
ax.set_title('Time Series Temperature at 100 m',fontsize=20)
ax.set_xlim(datetime(2018,10,5),datetime(2018,10,13))
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.tick_params(labelsize=14)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_time_series_ng288_GOFS31_100m.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 




