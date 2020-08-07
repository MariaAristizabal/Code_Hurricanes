#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 08:53:00 2020

@author: aristizabal
"""

#%% User input

# RU33 (MAB + SAB)
lon_lim = [-75,-70]
lat_lim = [36,42]

# Folder where to save figure
folder_fig = '/home/aristizabal/Figures/'

# Folder Fay Roms
folder_hwrf_pom = '/home/aristizabal/HWRF_POM_Fay/HWRF_POM_06l_2020070918/'

# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
#import netCDF4
#from datetime import datetime, timedelta
import cmocean
import os
import os.path
import glob
from matplotlib.dates import date2num, num2date

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
#oklatbath = oklatbath[:,np.newaxis]
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])
#oklonbath = oklonbath[:,np.newaxis]

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
#bath_elevsub = bath_elev[oklatbath,oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Defining cross transect

x1 = -74.1
y1 = 39.4
x2 = -73.0
y2 = 38.6
# Slope
m = (y1-y2)/(x1-x2)
# Intercept
b = y1 - m*x1

X = np.arange(x1,-72,0.05)
Y = b + m*X

dist = np.sqrt((X-x1)**2 + (Y-y1)**2)*111 # approx along transect distance in km

#%% Reading POM grid file
grid_file = sorted(glob.glob(os.path.join(folder_hwrf_pom,'*grid*.nc')))[0]
pom_grid = xr.open_dataset(grid_file)
lon_pom = np.asarray(pom_grid['east_e'][:])
lat_pom = np.asarray( pom_grid['north_e'][:])
zlevc = np.asarray(pom_grid['zz'][:])
topoz = np.asarray(pom_grid['h'][:])

#%% Getting list of POM files
ncfiles = sorted(glob.glob(os.path.join(folder_hwrf_pom,'*pom.0*.nc')))

# Reading POM time
time_pom = []
for i,file in enumerate(ncfiles):
    print(i)
    pom = xr.open_dataset(file)
    tpom = pom['time'][:]
    timestamp_pom = date2num(tpom)[0]
    time_pom.append(num2date(timestamp_pom))

time_POM = np.asarray(time_pom)

oklon = np.round(np.interp(X,lon_pom[0,:],np.arange(len(lon_pom[0,:])))).astype(int)
oklat = np.round(np.interp(Y,lat_pom[:,0],np.arange(len(lat_pom[:,0])))).astype(int)
topoz_pom = np.asarray(topoz[oklat,oklon])
zmatrix_POM = np.dot(topoz_pom.reshape(-1,1),zlevc.reshape(1,-1)).T
dist_matrix = np.tile(dist,(zmatrix_POM.shape[0],1))

trans_temp_POM = np.empty((zmatrix_POM.shape[0],zmatrix_POM.shape[1]))
trans_temp_POM[:] = np.nan

max_valt = 26
min_valt = 8  
nlevelst = max_valt - min_valt + 1

kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))

for i,file in enumerate(ncfiles[0:6]):
    print(i)
    pom = xr.open_dataset(file)
    tpom = pom['time'][:]
    timestamp_pom = date2num(tpom)[0]
    timePOM = num2date(timestamp_pom)
    print(timePOM)
    for x in np.arange(len(X)):
        trans_temp_POM[:,x] = np.asarray(pom['t'][0,:,oklat[x],oklon[x]])
    
    fig, ax = plt.subplots(figsize=(9, 3))
    plt.contourf(dist_matrix,zmatrix_POM,trans_temp_POM,cmap=cmocean.cm.thermal,**kw)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cs = plt.contour(dist_matrix,zmatrix_POM,trans_temp_POM,[24],color='k')
    fmt = '%i'
    plt.clabel(cs,fmt=fmt)
    plt.contour(dist_matrix,zmatrix_POM,trans_temp_POM,[12],color='k')
    cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    plt.title('HWRF-POM Transect on ' + str(timePOM)[0:13],fontsize=16)
    plt.ylim([-100,0])
    plt.xlim([0,200])
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Along Transect Distance (km)',fontsize=14)
    
    file = folder_fig + 'HWRF-POM_cross_shelf_transect_'+str(timePOM)[0:13]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)