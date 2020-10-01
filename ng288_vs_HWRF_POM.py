#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:27:57 2019

@author: aristizabal
"""

#%% User input

#ncfolder = '/Volumes/aristizabal/ncep_model/HWRF-POM_Michael/'
ncfolder = '/home/aristizabal/ncep_model/HWRF-POM_Michael/'

Dir_graph = '/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts/Figures/'

# POM grid file name
grid_file = 'michael14l.2018100718.pom.grid.nc'

# POM file name
prefix = 'michael14l.2018100718.pom.'

# Name of 3D variable
var_name = 'temp'

# Glider data 

# ng288
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

# date limits
date_ini = '2018-10-08T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'

#%%

import matplotlib.pyplot as plt
import xarray as xr
import netCDF4
import numpy as np
from datetime import datetime 
import matplotlib.dates as mdates
from matplotlib.dates import date2num, num2date
import scipy.io as sio
import os
import os.path
import glob

#%% Reading glider data

ncglider = xr.open_dataset(gdata+'#fillmismatch') #,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = np.asarray(ncglider.time[0])
time_glider = np.asarray(mdates.num2date(mdates.date2num(time_glider)))
#time_glider = netCDF4.num2date(time_glider[:],time_glider.units)
tempglider = np.array(ncglider.temperature[0,:,:])
saltglider = np.array(ncglider.salinity[0,:,:])
depthglider = np.array(ncglider.depth[0,:,:])

timestamp_glider = date2num(time_glider)

#%%
tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

okg = np.where(np.logical_and(mdates.date2num(time_glider) >= mdates.date2num(tmin),\
                              mdates.date2num(time_glider) <= mdates.date2num(tmax)))

timeg = time_glider[okg[0]]
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

#%% Reading POM grid files

pom_grid = xr.open_dataset(ncfolder + grid_file)

lonc = np.asarray(pom_grid['east_e'][:])
latc = np.asarray( pom_grid['north_e'][:])
#lonu, latu = pom_grid['east_u'][:], pom_grid['north_u'][:]
#lonv, latv = pom_grid['east_v'][:], pom_grid['north_v'][:]
zlevc = np.asarray(pom_grid['zz'][:])
topoz = np.asarray(pom_grid['h'][:])

#%% Reading POM temperature

ncfiles = sorted(glob.glob(os.path.join(ncfolder,prefix+'*0*.nc')))

target_temp_pom = np.empty((len(ncfiles),len(zlevc),))
target_temp_pom[:] = np.nan
target_topoz_pom = np.empty((len(ncfiles),))
target_topoz_pom[:] = np.nan
time_pom = []
for x,file in enumerate(ncfiles):
    print(x)
    pom = xr.open_dataset(file)

    tpom = pom['time'][:]
    timestamp_pom = date2num(tpom)[0]
    time_pom.append(num2date(timestamp_pom))

    # Interpolating latg and longlider into RTOFS grid
    sublonpom = np.interp(timestamp_pom,timestampg,long)
    sublatpom = np.interp(timestamp_pom,timestampg,latg)
    oklonpom = np.int(np.round(np.interp(sublonpom,lonc[0,:],np.arange(len(lonc[0,:])))))
    oklatpom = np.int(np.round(np.interp(sublatpom,latc[:,0],np.arange(len(latc[:,0])))))

    target_temp_pom[x,:] = np.asarray(pom['t'][0,:,oklatpom,oklonpom])
    target_topoz_pom[x] = np.asarray(topoz[oklatpom,oklonpom])

timestamp_pom = date2num(time_pom)

#%% Figure

time_matrix_pom = np.tile(timestamp_pom,(zlevc.shape[0],1)).T
z_matrix_pom = np.dot(target_topoz_pom.reshape(-1,1),zlevc.reshape(1,-1))
 
time_pom_string = [datetime.strftime(tt,'%Y-%m-%d %H-%M') for tt in time_pom] 

mdic = {"time_pom_string": time_pom_string, "z_matrix_pom": z_matrix_pom,\
        "target_temp_pom":target_temp_pom}
sio.savemat("ng288_HWRF_POM_Michael_2018.mat",mdic)

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.contour(time_matrix_pom[:,0:-1],z_matrix_pom[:,0:-1],target_temp_pom[:,0:-1],colors = 'lightgrey',**kw)
plt.contour(time_matrix_pom[:,0:-1],z_matrix_pom[:,0:-1],target_temp_pom[:,0:-1],[26],colors = 'k')
plt.contourf(time_matrix_pom[:,0:-1],z_matrix_pom[:,0:-1],target_temp_pom[:,0:-1],cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(zlevc)),z_matrix_pom[0,:],'--k')
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
ax.set_ylim(-260,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('HWRF-POM',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/POM_temp_Michael.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%%
#target_temp_pom[0,0:-1]
#z_matrix_pom[0,26]
time_pom
#len(ncfiles)