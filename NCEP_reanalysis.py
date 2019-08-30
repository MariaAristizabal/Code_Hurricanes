#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:35:31 2019

@author: aristizabal
"""

#%% User input

# downloaded from 
# https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBSearch.pl?Dataset=NCEP+Reanalysis&Variable=Net+longwave+radiation+flux

net_lwr_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/nlwrs.sfc.gauss.2018.nc' # long wave radiation
net_swr_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/nswrs.sfc.gauss.2018.nc' # short wave radiation
net_lht_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/lhtfl.sfc.gauss.2018.nc' # laten heat flux
net_sht_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/shtfl.sfc.gauss.2018.nc' # sesible heat flux

date_ini = '2018-06-01T00:00:00Z'
date_end = '2018-11-30T00:00:00Z'

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
net_lwr_time= Net_lwr_time[ok_time[0]]
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

#%% Read net latent heat flux data

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

Netlht = xr.open_dataset(net_lht_nc)
Net_lht_time = np.asarray(Netlht.variables['time'][:])
Net_lht_lat = np.asarray(Netlht.variables['lat'][:])
Net_lht_lonn = np.asarray(Netlht.variables['lon'][:])
Net_lht = np.asarray(Netlht.variables['lhtfl'][:])

# Conversion from NCEP reanalaysis longitude convention to geographic convention
Net_lht_lon = np.empty((len(Net_lht_lonn),))
Net_lht_lon[:] = np.nan
for i,ii in enumerate(Net_lht_lonn):
    if ii > 180: 
        Net_lht_lon[i] = ii - 360
    else:
        Net_lht_lon[i] = ii
    
ok = np.argsort(Net_lht_lon, axis=0, kind='quicksort', order=None)    
Net_lht_lon =  Net_lht_lon[ok] 
Net_lht = Net_lht[:,:,ok]  

ok_lat = np.where(np.logical_and(Net_lht_lat >= lat_lim[0],Net_lht_lat <= lat_lim[1]))
ok_lon = np.where(np.logical_and(Net_lht_lon >= lon_lim[0],Net_lht_lon <= lon_lim[1]))
ok_time = np.where(np.logical_and(mdates.date2num(Net_lht_time) >= mdates.date2num(tini),\
                                  mdates.date2num(Net_lht_time) <= mdates.date2num(tend)))

net_lht_lon = Net_lht_lon[ok_lon[0]]
net_lht_lat = Net_lht_lat[ok_lat[0]]
net_lht_time= Net_lht_time[ok_time[0]]
net_lht = np.asarray(Net_lht[ok_time[0],:,:][:,ok_lat[0],:][:,:,ok_lon[0]]) 

#%% Read net sensible heat flux data

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

Netsht = xr.open_dataset(net_sht_nc)
Net_sht_time = np.asarray(Netsht.variables['time'][:])
Net_sht_lat = np.asarray(Netsht.variables['lat'][:])
Net_sht_lonn = np.asarray(Netsht.variables['lon'][:])
Net_sht = np.asarray(Netsht.variables['shtfl'][:])

# Conversion from NCEP reanalaysis longitude convention to geographic convention
Net_sht_lon = np.empty((len(Net_sht_lonn),))
Net_sht_lon[:] = np.nan
for i,ii in enumerate(Net_sht_lonn):
    if ii > 180: 
        Net_sht_lon[i] = ii - 360
    else:
        Net_sht_lon[i] = ii
    
ok = np.argsort(Net_sht_lon, axis=0, kind='quicksort', order=None)    
Net_sht_lon =  Net_sht_lon[ok] 
Net_sht = Net_sht[:,:,ok]  

ok_lat = np.where(np.logical_and(Net_sht_lat >= lat_lim[0],Net_sht_lat <= lat_lim[1]))
ok_lon = np.where(np.logical_and(Net_sht_lon >= lon_lim[0],Net_sht_lon <= lon_lim[1]))
ok_time = np.where(np.logical_and(mdates.date2num(Net_sht_time) >= mdates.date2num(tini),\
                                  mdates.date2num(Net_sht_time) <= mdates.date2num(tend)))

net_sht_lon = Net_sht_lon[ok_lon[0]]
net_sht_lat = Net_sht_lat[ok_lat[0]]
net_sht_time= Net_sht_time[ok_time[0]]
net_sht = np.asarray(Net_sht[ok_time[0],:,:][:,ok_lat[0],:][:,:,ok_lon[0]]) 

#%% Longwave radiation 

X,Y = np.meshgrid(net_lwr_lon,net_lwr_lat)

t=3
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(net_lwr_lon,net_lwr_lat,net_lwr[t,:,:],cmap=cmocean.cm.thermal)
plt.colorbar()
plt.plot(X,Y,'*k')
plt.title('Net Longwave Radiation, NCEP Reanalysis \n' + str(net_lwr_time[t]))

#%% Shortwave radiation

t=3 
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(net_swr_lon,net_swr_lat,net_swr[t,:,:],cmap=cmocean.cm.thermal)
plt.colorbar()
plt.title('Net Shortwave Radiation, NCEP Reanalysis \n' + str(net_swr_time[t]))

#%% Latent heat flux radiation

t=3 
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(net_lht_lon,net_lht_lat,net_lht[t,:,:],cmap=cmocean.cm.thermal)
plt.colorbar()
plt.title('Net Latent Heat Flux, NCEP Reanalysis \n' + str(net_lht_time[t]))

#%% Sensible heat flux radiation

t=3 
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(net_lht_lon,net_lht_lat,net_sht[t,:,:],cmap=cmocean.cm.thermal)
plt.colorbar()
plt.title('Net Sensible Heat Flux, NCEP Reanalysis \n' + str(net_sht_time[t]))
