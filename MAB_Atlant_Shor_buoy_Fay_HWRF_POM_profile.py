#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:07:35 2020

@author: aristizabal
"""

#%% User input

# RU33 (MAB + SAB)
lon_lim = [-75,-70]
lat_lim = [36,42]

lat_buoy = 39.2717
lon_buoy = -73.88919

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

oklon = np.round(np.interp(lon_buoy,lon_pom[0,:],np.arange(len(lon_pom[0,:])))).astype(int)
oklat = np.round(np.interp(lat_buoy,lat_pom[:,0],np.arange(len(lat_pom[:,0])))).astype(int)
topoz_pom = np.asarray(topoz[oklat,oklon])
zmatrix_POM = np.dot(topoz_pom.reshape(-1,1),zlevc.reshape(1,-1)).T

prof_temp_POM = np.empty((len(time_POM),zmatrix_POM.shape[0]))
prof_temp_POM[:] = np.nan

max_valt = 26
min_valt = 8  
nlevelst = max_valt - min_valt + 1

kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))

for i,file in enumerate(ncfiles):
    print(i)
    pom = xr.open_dataset(file)
    tpom = pom['time'][:]
    timestamp_pom = date2num(tpom)[0]
    timePOM = num2date(timestamp_pom)
    print(timePOM)
    prof_temp_POM[i,:] = np.asarray(pom['t'][0,:,oklat,oklon])
    
#%%    
    
fig, ax = plt.subplots(figsize=(4, 8))
plt.plot(prof_temp_POM.T,zmatrix_POM[:,0],'.-')


#file = folder_fig + 'HWRF-POM_temp_prof_Atlac_Shor_'+str(timePOM)[0:13]
#plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)