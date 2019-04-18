#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:28:15 2019

@author: aristizabal
"""

#%% User input

mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng288.mat'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

# date limits
date_ini = '2018-10-08T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'

# Glider data 

# ng288
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

#%%

import matplotlib.pyplot as plt
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
'''
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
''' 

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