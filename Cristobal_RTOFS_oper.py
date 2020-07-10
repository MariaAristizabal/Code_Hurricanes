#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:08:41 2020

@author: aristizabal
"""

cycle = '20200604'

# lat and lon bounds
lon_lim = [-98,-80.0]
lat_lim = [15.0,32.5]

# RTOFS files
folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

folder_figs = '/home/aristizabal/Figures/'

#%% 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import cmocean

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Time window

year = int(cycle[0:4])
month = int(cycle[4:6])
day = int(cycle[6:8])
#hour = int(cycle[9:])
tini = datetime(year, month, day)
tend = tini + timedelta(days=1)

#%% Reading bathymetry data
ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]  

#%% Read RTOFS grid and time
print('Retrieving coordinates from RTOFS')

if tini.month < 10:
    if tini.day < 10:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day)
else:
    if tini.day < 10:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + str(tini.day)

ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[0])
latRTOFS = np.asarray(ncRTOFS.Latitude[:])
lonRTOFS = np.asarray(ncRTOFS.Longitude[:])
depth_RTOFS = np.asarray(ncRTOFS.Depth[:])

tRTOFS = []
for t in np.arange(len(nc_files_RTOFS)):
    ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
          for t in np.arange(len(nc_files_RTOFS))])

#%%

oklonRTOFS = np.where(np.logical_and(lonRTOFS[0,:] >= lon_lim[0],lonRTOFS[0,:] <= lon_lim[1]))[0]
oklatRTOFS = np.where(np.logical_and(latRTOFS[:,0] >= lat_lim[0],latRTOFS[:,0] <= lat_lim[1]))[0]

i=0
nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[i]
ncRTOFS = xr.open_dataset(nc_file)
time_RTOFS = tRTOFS[0]
temp_RTOFS = np.asarray(ncRTOFS.variables['temperature'][0,0,oklatRTOFS,oklonRTOFS])
lon_RTOFS = lonRTOFS[0,oklonRTOFS]
lat_RTOFS = latRTOFS[oklatRTOFS,0]

kw = dict(levels = np.linspace(24,30,16))

fig, ax = plt.subplots()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
cs = plt.contourf(lon_RTOFS,lat_RTOFS,temp_RTOFS,cmap=cmocean.cm.thermal,**kw)
cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
cbar.ax.tick_params(labelsize=14)
plt.axis('scaled')
plt.title('RTOFS SST on ' + str(time_RTOFS)[0:13],fontsize=16)
ax.set_xlim(lon_lim[0],lon_lim[-1])
ax.set_ylim(lat_lim[0],lat_lim[-1])
plt.plot(np.tile(-90,len(lat_RTOFS)),lat_RTOFS,'-',color='k')

file = folder_figs + 'RTOFS_surf_temp_' + str(time_RTOFS)[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
oklon = np.where(lonRTOFS[0,:] >= -90)[0][0]
transect_temp_RTOFS = np.asarray(ncRTOFS.variables['temperature'][0,:,oklatRTOFS,oklon])

kw = dict(levels = np.linspace(12,32,21))
plt.figure()
plt.contourf(lat_RTOFS,-depth_RTOFS,transect_temp_RTOFS,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
plt.contour(lat_RTOFS,-depth_RTOFS,transect_temp_RTOFS,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.title('Temperature along Cristobal Path',fontsize=16)
plt.ylim([-200,0])
plt.xlim([20,30])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

file = folder_figs + 'RTOFS_temp_along_Cristobal_' + str(time_RTOFS)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%

oklonRTOFS = np.where(np.logical_and(lonRTOFS[0,:] >= lon_lim[0],lonRTOFS[0,:] <= lon_lim[1]))[0]
oklatRTOFS = np.where(np.logical_and(latRTOFS[:,0] >= lat_lim[0],latRTOFS[:,0] <= lat_lim[1]))[0]

i=0
nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[i]
ncRTOFS = xr.open_dataset(nc_file)
time_RTOFS = tRTOFS[0]
salt_RTOFS = np.asarray(ncRTOFS.variables['salinity'][0,0,oklatRTOFS,oklonRTOFS])
lon_RTOFS = lonRTOFS[0,oklonRTOFS]
lat_RTOFS = latRTOFS[oklatRTOFS,0]

kw = dict(levels = np.linspace(32,37,11))

fig, ax = plt.subplots()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
cs = plt.contourf(lon_RTOFS,lat_RTOFS,salt_RTOFS,cmap=cmocean.cm.haline,**kw)
cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.tick_params(labelsize=14)
plt.axis('scaled')
plt.title('RTOFS SSS on ' + str(time_RTOFS)[0:13],fontsize=16)
ax.set_xlim(lon_lim[0],lon_lim[-1])
ax.set_ylim(lat_lim[0],lat_lim[-1])
#plt.plot(np.tile(-90,len(lat_RTOFS)),lat_RTOFS,'-',color='k')

file = folder_figs + 'RTOFS_surf_salt_' + str(time_RTOFS)[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
