#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:35:31 2019

@author: aristizabal
"""

#%% User input

net_lwr_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/nlwrs.sfc.gauss.2018.nc' # long wave radiation
net_swr_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/nswrs.sfc.gauss.2018.nc' # short wave radiation
net_lht_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/lhtfl.sfc.gauss.2018.nc' # laten heat flux
net_sht_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/shtfl.sfc.gauss.2018.nc' # sesible heat flux

#uu = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L3U/WindSat/REMSS/v7.0.1a/2017/001/20170101000000-REMSS-L3U_GHRSST-SSTsubskin-WSAT-wsat_20170101v7.0.1-v02.0-fv01.0.nc';

date_ini = '2018-07-17T00:00:00Z'
date_end = '2018-09-18T00:00:00Z'

lon_lim = [-100.0,-60.0]
lat_lim = [5.0,45.0]

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'
#bath_file = '/Volumes/aristizabal/bathymetry_files/GEBCO_2019.nc'


#%%

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

#import urllib.request

import cmocean

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

# Getting subdomain for plotting glider track on bathymetry
oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[1])
        
bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Read net longwave radiation data

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

Netlwr = xr.open_dataset(net_lwr_nc)
Net_lwr_time = np.asarray(Netlwr.variables['time'][:])
Net_lwr_lat = np.asarray(Netlwr.variables['lat'][:])
Net_lwr_lonn = np.asarray(Netlwr.variables['lon'][:])
Net_lwr = np.asarray(Netlwr.variables['nlwrs'][:])

# Conversion from NCEP reanalaysis longitude convention to geographic convention
Net_lwr_lon = np.empty((len(Net_lwr_lonn),))
Net_lwr_lon[:] = np.nan
for i,ii in enumerate(Net_lwr_lonn):
    if ii > 180: 
        Net_lwr_lon[i] = ii - 360
    else:
        Net_lwr_lon[i] = ii
    
ok = np.argsort(Net_lwr_lon, axis=0, kind='quicksort', order=None)    
Net_lwr_lon =  Net_lwr_lon[ok] 
Net_lwr = Net_lwr[:,:,ok]  

ok_lat = np.where(np.logical_and(Net_lwr_lat >= lat_lim[0],Net_lwr_lat <= lat_lim[1]))
ok_lon = np.where(np.logical_and(Net_lwr_lon >= lon_lim[0],Net_lwr_lon <= lon_lim[1]))
ok_time = np.where(np.logical_and(mdates.date2num(Net_lwr_time) >= mdates.date2num(tini),\
                                  mdates.date2num(Net_lwr_time) <= mdates.date2num(tend)))

net_lwr_lon = Net_lwr_lon[ok_lon[0]]
net_lwr_lat = Net_lwr_lat[ok_lat[0]]
net_lwr_time= Net_lwr_time[ok_lat[0]]
net_lwr = np.asarray(Net_lwr[ok_time[0],:,:][:,ok_lat[0],:][:,:,ok_lon[0]]) 


#%% Read net shortwave radiation data

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

Netswr = xr.open_dataset(net_swr_nc)
Net_swr_time = np.asarray(Netswr.variables['time'][:])
Net_swr_lat = np.asarray(Netswr.variables['lat'][:])
Net_swr_lonn = np.asarray(Netswr.variables['lon'][:])
Net_swr = np.asarray(Netswr.variables['nswrs'][:])

# Conversion from NCEP reanalaysis longitude convention to geographic convention
Net_swr_lon = np.empty((len(Net_swr_lonn),))
Net_swr_lon[:] = np.nan
for i,ii in enumerate(Net_swr_lonn):
    if ii > 180: 
        Net_swr_lon[i] = ii - 360
    else:
        Net_swr_lon[i] = ii
    
ok = np.argsort(Net_swr_lon, axis=0, kind='quicksort', order=None)    
Net_swr_lon =  Net_swr_lon[ok] 
Net_swr = Net_swr[:,:,ok]  

ok_lat = np.where(np.logical_and(Net_swr_lat >= lat_lim[0],Net_swr_lat <= lat_lim[1]))
ok_lon = np.where(np.logical_and(Net_swr_lon >= lon_lim[0],Net_swr_lon <= lon_lim[1]))
ok_time = np.where(np.logical_and(mdates.date2num(Net_swr_time) >= mdates.date2num(tini),\
                                  mdates.date2num(Net_swr_time) <= mdates.date2num(tend)))

net_swr_lon = Net_swr_lon[ok_lon[0]]
net_swr_lat = Net_swr_lat[ok_lat[0]]
net_swr_time= Net_swr_time[ok_time[0]]
net_swr = np.asarray(Net_swr[ok_time[0],:,:][:,ok_lat[0],:][:,:,ok_lon[0]]) 

#%% Longwave radiation 

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(net_lwr_lon,net_lwr_lat,net_lwr[0,:,:],cmap=cmocean.cm.thermal)
plt.colorbar()
plt.title('Net Longwave Radiation, NCEP Reanalysis \n' + str(tini) )

#%% Shortwave radiation

t=3 
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(net_swr_lon,net_swr_lat,net_swr[t,:,:],cmap=cmocean.cm.thermal)
plt.colorbar()
plt.title('Net Shortwave Radiation, NCEP Reanalysis \n' + str(net_swr_time[t]) )
