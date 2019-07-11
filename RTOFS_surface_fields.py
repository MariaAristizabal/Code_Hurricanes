#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:41:45 2019

@author: aristizabal
"""

date = '20190608'
#date = '20190607'
folder_RTOFS = '/Volumes/coolgroup/RTOFS/forecasts/domains/hurricanes/'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

#%%

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean

import numpy as np
import xarray as xr
import netCDF4

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

#%% Read RTOFS output

print('Retrieving coordinates from RTOFS')
file_RTOFS_temp = folder_RTOFS + date + '/' + date + '/' + 'hurricanes_' + date + '_rtofs_glo_3dz_nowcast_daily_temp.nc4.nc' 
file_RTOFS_salt = folder_RTOFS + date + '/' + date + '/' + 'hurricanes_' + date + '_rtofs_glo_3dz_nowcast_daily_salt.nc4.nc' 
file_RTOFS_uvel = folder_RTOFS + date + '/' + date + '/' + 'hurricanes_' + date + '_rtofs_glo_3dz_nowcast_daily_uvel.nc4.nc' 
file_RTOFS_vvel = folder_RTOFS + date + '/' + date + '/' + 'hurricanes_' + date + '_rtofs_glo_3dz_nowcast_daily_vvel.nc4.nc' 

RTOFS_temp = xr.open_dataset(file_RTOFS_temp ,decode_times=False)
RTOFS_salt = xr.open_dataset(file_RTOFS_salt ,decode_times=False)
RTOFS_uvel = xr.open_dataset(file_RTOFS_uvel ,decode_times=False)
RTOFS_vvel = xr.open_dataset(file_RTOFS_vvel ,decode_times=False)
    
latRTOFS = RTOFS_temp.lat[:]
lonRTOFS = RTOFS_temp.lon[:]
depthRTOFS = RTOFS_temp.lev[:]
ttRTOFS = RTOFS_temp.time[:]
tRTOFS = netCDF4.num2date(ttRTOFS[:],ttRTOFS.units) 
timeRTOFS = mdates.num2date(mdates.date2num(tRTOFS))

RTOFStemp = RTOFS_temp.variables['temperature'][2,0,:,:]
RTOFSsalt = RTOFS_salt.variables['salinity'][2,0,:,:]
RTOFSuvel = RTOFS_uvel.variables['u'][2,0,:,:]
RTOFSvvel = RTOFS_vvel.variables['v'][2,0,:,:]

#%%

fig, ax = plt.subplots(figsize=(9, 3))
cs = plt.contourf(lonRTOFS-360,latRTOFS,RTOFStemp,cmap=cmocean.cm.thermal)
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
plt.title('Surface Temperature RTOFS ' + date)

#plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
ax.set_xlim(lon_lim[0],lon_lim[-1])
ax.set_ylim(lat_lim[0],lat_lim[-1])

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'surf_temp_' + date 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%

fig, ax = plt.subplots(figsize=(9, 3))
cs = plt.contourf(lonRTOFS,latRTOFS,RTOFStemp,cmap=cmocean.cm.thermal)
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
plt.title('Surface Temperature RTOFS ' + date)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'surf_temp_' + date 
#plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%

nlevels =11
kw = dict(levels = np.linspace(28,38,nlevels))

fig, ax = plt.subplots(figsize=(9, 3))
cs = plt.contourf(lonRTOFS-360,latRTOFS,RTOFSsalt,cmap=cmocean.cm.haline,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
#cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
plt.title('Surface Salinity RTOFS ' + date)

#plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
ax.set_xlim(lon_lim[0],lon_lim[-1])
ax.set_ylim(lat_lim[0],lat_lim[-1])

#plt.clim(28,35)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'surf_salt_' + date 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%

nlevels = 11
kw = dict(levels = np.linspace(-2.5,2.5,nlevels))

fig, ax = plt.subplots(figsize=(9, 3))
cs = plt.contourf(lonRTOFS-360,latRTOFS,RTOFSuvel,cmap=cmocean.cm.balance,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($m/s$)',fontsize=14,labelpad=15)
plt.title('Surface u Velocity RTOFS ' + date)

#plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
ax.set_xlim(lon_lim[0],lon_lim[-1])
ax.set_ylim(lat_lim[0],lat_lim[-1])

#plt.clim(28,35)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'surf_uvel_' + date 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
nlevels = 11
kw = dict(levels = np.linspace(-2.5,2.5,nlevels))

fig, ax = plt.subplots(figsize=(9, 3))
cs = plt.contourf(lonRTOFS-360,latRTOFS,RTOFSvvel,cmap=cmocean.cm.balance,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($m/s$)',fontsize=14,labelpad=15)
plt.title('Surface v Velocity RTOFS ' + date)

#plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
ax.set_xlim(lon_lim[0],lon_lim[-1])
ax.set_ylim(lat_lim[0],lat_lim[-1])

#plt.clim(28,35)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'surf_vvel_' + date 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
