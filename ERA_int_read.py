#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:21:46 2019

@author: aristizabal
"""

#%% User input

# downloaded from 
# https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBSearch.pl?Dataset=NCEP+Reanalysis&Variable=Net+longwave+radiation+flux

#ERA_int_nc = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/_grib2netcdf-atls19-95e2cf679cd58ee9b4db4dd119a05a8d-LvcaHi.nc'
ERA_int_nc = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/_grib2netcdf-atls17-a82bacafb5c306db76464bc7e824bb75-p0g7yv.nc'
#date_ini = '2018-06-01T00:00:00Z'
#date_end = '2018-11-30T00:00:00Z'

lon_lim = [-100.0,-60.0]
lat_lim = [5.0,45.0]

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'
#bath_file = '/Volumes/aristizabal/bathymetry_files/GEBCO_2019.nc'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

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

#tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
#tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

ERA_int = xr.open_dataset(ERA_int_nc)
ERA_int_time = np.asarray(ERA_int.variables['time'][:])
ERAint_lat = np.asarray(ERA_int.variables['latitude'][:])
ERAint_lonn = np.asarray(ERA_int.variables['longitude'][:])
#ERA_int = np.asarray(Netlwr.variables['nlwrs'][:])

#%%
# Conversion from NCEP reanalaysis longitude convention to geographic convention
ERAint_lon = np.empty((len(ERAint_lonn),))
ERAint_lon[:] = np.nan
for i,ii in enumerate(ERAint_lonn):
    if ii > 180: 
        ERAint_lon[i] = ii - 360
    else:
        ERAint_lon[i] = ii

#%%    
#ok = np.argsort(ERA_int_lon, axis=0, kind='quicksort', order=None)    
#Net_lwr_lon =  Net_lwr_lon[ok] 
#Net_lwr = Net_lwr[:,:,ok]  

ok_lat = np.where(np.logical_and(ERAint_lat >= lat_lim[0],ERAint_lat <= lat_lim[1]))
ok_lon = np.where(np.logical_and(ERAint_lon >= lon_lim[0],ERAint_lon <= lon_lim[1]))
#ok_time = np.where(np.logical_and(mdates.date2num(ERA_int_time) >= mdates.date2num(tini),\
#                                  mdates.date2num(Net_lwr_time) <= mdates.date2num(tend)))
#%%
ERA_int_lon = ERAint_lon[ok_lon[0]]
ERA_int_lat = ERAint_lat[ok_lat[0]]
#net_lwr_time= Net_lwr_time[ok_time[0]]
ERA_int_sshf = np.asarray(ERA_int.variables['sshf'][:,ok_lat[0],ok_lon[0]])
ERA_int_slhf = np.asarray(ERA_int.variables['slhf'][:,ok_lat[0],ok_lon[0]])
ERA_int_ssrd = np.asarray(ERA_int.variables['ssrd'][:,ok_lat[0],ok_lon[0]])
ERA_int_ssr = np.asarray(ERA_int.variables['ssr'][:,ok_lat[0],ok_lon[0]])
ERA_int_strd = np.asarray(ERA_int.variables['strd'][:,ok_lat[0],ok_lon[0]])
ERA_int_str = np.asarray(ERA_int.variables['str'][:,ok_lat[0],ok_lon[0]])


#%%
kw = dict(levels = np.linspace(-500,500,21))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(ERA_int_lon,ERA_int_lat,ERA_int_sshf[3,:,:]/(3*3600),cmap=cmocean.cm.balance,**kw)
plt.colorbar()
plt.title(ERA_int.variables['sshf'].attrs['long_name'])

#%%
kw = dict(levels = np.linspace(-1000,1000,21))
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(ERA_int_lon,ERA_int_lat,ERA_int_slhf[3,:,:]/(3*3600),cmap=cmocean.cm.balance,**kw)
plt.colorbar()
plt.title(ERA_int.variables['slhf'].attrs['long_name'])

#%%
'''
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(ERA_int_lon,ERA_int_lat,ERA_int_ssrd[3,:,:]/(3*3600),cmap=cmocean.cm.balance) #,**kw)
plt.colorbar()
plt.title(ERA_int.variables['ssrd'].attrs['long_name'])
'''
#%%
kw = dict(levels = np.linspace(-500,500,21))
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(ERA_int_lon,ERA_int_lat,ERA_int_ssr[3,:,:]/(3*3600),cmap=cmocean.cm.balance,**kw)
plt.colorbar()
plt.title(ERA_int.variables['ssr'].attrs['long_name'])

#%%
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(ERA_int_lon,ERA_int_lat,ERA_int_strd[3,:,:]/(3*3600),cmap=cmocean.cm.balance) #,**kw)
plt.colorbar()
plt.title(ERA_int.variables['strd'].attrs['long_name'])

#%%
kw = dict(levels = np.linspace(-500,500,21))
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(ERA_int_lon,ERA_int_lat,ERA_int_str[3,:,:]/(6*3600),cmap=cmocean.cm.balance,**kw)
plt.colorbar()
plt.title(ERA_int.variables['str'].attrs['long_name'])

#%%
'''
#%% Shortwave radiation

X,Y = np.meshgrid(net_swr_lon,net_swr_lat)
tind = np.where(mdates.date2num(net_swr_time) == mdates.date2num(datetime(2018,8,1,12)))[0][0]

kw = dict(levels = np.linspace(-1000,1000,21))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(net_swr_lon,net_swr_lat,-net_swr[tind,:,:],cmap=cmocean.cm.balance,**kw)
cbar = plt.colorbar()
plt.plot(X,Y,'*k')
plt.title('Net Shortwave Radiation, NCEP Reanalysis \n' + str(net_swr_time[tind]))
cbar.ax.set_ylabel(Netswr.variables['nswrs'].attrs['units'],fontsize=16)

file = folder + ' ' + 'net_swr' + str(net_swr_time[tind])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Latent heat flux radiation

X,Y = np.meshgrid(net_lht_lon,net_lht_lat)
tind = np.where(mdates.date2num(net_lht_time) == mdates.date2num(datetime(2018,8,1,12)))[0][0]

kw = dict(levels = np.linspace(-1000,1000,21))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(net_lht_lon,net_lht_lat,-net_lht[tind,:,:],cmap=cmocean.cm.balance,**kw)
cbar = plt.colorbar()
plt.title('Net Latent Heat Flux, NCEP Reanalysis \n' + str(net_lht_time[tind]))
cbar.ax.set_ylabel(Netlht.variables['lhtfl'].attrs['units'],fontsize=16)
plt.plot(X,Y,'*k')

file = folder + ' ' + 'net_lht' + str(net_lht_time[tind])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Sensible heat flux radiation

X,Y = np.meshgrid(net_lht_lon,net_lht_lat)
tind = np.where(mdates.date2num(net_lht_time) == mdates.date2num(datetime(2018,8,1,12)))[0][0]

kw = dict(levels = np.linspace(-1000,1000,21))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(net_lht_lon,net_lht_lat,-net_sht[tind,:,:],cmap=cmocean.cm.balance,**kw)
cbar = plt.colorbar()
plt.title('Net Sensible Heat Flux, NCEP Reanalysis \n' + str(net_sht_time[tind]))
plt.plot(X,Y,'*k')
cbar.ax.set_ylabel(Netsht.variables['shtfl'].attrs['units'],fontsize=16)

cbar.ax.set_ylabel(Netsht.variables['shtfl'].attrs['units'],fontsize=16)

file = folder + ' ' + 'net_sht' + str(net_sht_time[tind])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
'''