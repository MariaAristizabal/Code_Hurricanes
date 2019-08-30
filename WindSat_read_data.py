#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:02:15 2019

@author: aristizabal
"""

#%% User input

WindSat_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/windsat/L3/rss/v7/'

#'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/GDS2/L3U/WindSat/REMSS/v7.0.1a/'

date_ini = '2018-06-01T00:00:00Z'
date_end = '2018-11-30T00:00:00Z'

lon_lim = [-100.0,-60.0]
lat_lim = [5.0,45.0]

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# Folder to save WindSat files
folder_nc = '/Volumes/aristizabal/WindSat_data/' 

#%%

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from datetime import datetime, timedelta

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

#%% Download WindSat files
'''
tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

tvec = []
for t,tt in enumerate(np.arange((tend-tini).days+1)):
    tvec.append(tini + timedelta(t))

for t,tt in enumerate(np.arange((tend-tini).days+1)):
    tdate = tini + timedelta(t)
    file_name = tdate.strftime('%Y%m%d%H%M%S') + \
                  '-REMSS-L3U_GHRSST-SSTsubskin-WSAT-wsat_' + \
                  tdate.strftime('%Y%m%d') + 'v7.0.1-v02.0-fv01.0.nc'
    WindSat_url_nc = WindSat_url + str(tdate.year) + '/' +  tdate.strftime('%j') + '/' + \
                     file_name               
    urllib.request.urlretrieve(WindSat_url_nc, folder_nc+file_name)
'''

#%% Read WindSat data

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

kw = dict(levels = np.linspace(20,35,16))

#windsat_remss_ovw_l3_20190520_v7.0.1.nc.gz

#for t,tt in enumerate(np.arange((tend-tini).days+1)):
for t,tt in enumerate(np.arange(3)):
    print(t)
    tdate = tini + timedelta(t)
    file_name = 'windsat_remss_ovw_l3_' + tdate.strftime('%Y%m%d') + '_v7.0.1.nc.gz'
    
    #tdate.strftime('%Y%m%d%H%M%S') + \
    #              '-REMSS-L3U_GHRSST-SSTsubskin-WSAT-wsat_' + \
    #              tdate.strftime('%Y%m%d') + 'v7.0.1-v02.0-fv01.0.nc'
    WindSat_url_nc = WindSat_url + str(tdate.year) + '/' +  tdate.strftime('%j') + '/' + \
                     file_name               
    #urllib.request.urlretrieve(WindSat_url_nc, folder_nc+file_name)               
    WindSat = xr.open_dataset(WindSat_url_nc)
    
    WS_time = np.asarray(WindSat.variables['time'][:])
    WS_lat = np.asarray(WindSat.variables['lat'][:])
    WS_lon = np.asarray(WindSat.variables['lon'][:])

    ok_lon = np.where(np.logical_and(WS_lon >= lon_lim[0],WS_lon <= lon_lim[1]))
    ok_lat = np.where(np.logical_and(WS_lat >= lat_lim[0],WS_lat <= lat_lim[1]))

    WS_sst = np.asarray(WindSat.variables['sea_surface_temperature'][:]) #[:,ok_lat[0],ok_lon[0]])
    WS_wind_speed = np.asarray(WindSat.variables['wind_speed_aw'][:])
    WS_wind_dir = np.asarray(WindSat.variables['wind_direction'][:])
    
    ws_lon = WS_lon[ok_lon[0]]
    ws_lat = WS_lat[ok_lat[0]]
    ws_sst = WS_sst[:,ok_lat[0],:][:,:,ok_lon[0]]
    ws_wind_speed = WS_wind_speed[:,ok_lat[0],:][:,:,ok_lon[0]]
    ws_wind_dir = WS_wind_dir[:,ok_lat[0],:][:,:,ok_lon[0]]
    
    ws_u_wind = ws_wind_speed * np.cos(ws_wind_dir)
    ws_v_wind = ws_wind_speed * np.sin(ws_wind_dir)
    
    plt.figure()
    plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    #plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    #plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    plt.contourf(ws_lon,ws_lat,ws_sst[0,:,:],cmap=cmocean.cm.thermal,**kw)
    plt.quiver(ws_lon, ws_lat,ws_u_wind , ws_v_wind,units='xy' ,scale=1)
    cl = plt.colorbar()
    plt.ylim([lat_lim[0],lat_lim[1]])
    plt.xlim([lon_lim[0],lon_lim[1]])
    plt.title(WS_time[0])
    #plt.axis('scaled')

#%%
'''
WS_time = np.asarray(WindSat.variables['time'][:])
WS_lat = np.asarray(WindSat.variables['lat'][:])
WS_lon = np.asarray(WindSat.variables['lon'][:])

ok_lon = np.where(np.logical_and(WS_lon >= lon_lim[0],WS_lon <= lon_lim[1]))
ok_lat = np.where(np.logical_and(WS_lat >= lat_lim[0],WS_lat <= lat_lim[1]))

WS_sst = np.asarray(WindSat.variables['sea_surface_temperature'][:]) #[:,ok_lat[0],ok_lon[0]])
WS_wind_speed = np.asarray(WindSat.variables['wind_speed'][:])

ws_lon = WS_lon[ok_lon[0]]
ws_lat = WS_lat[ok_lat[0]]
ws_sst = WS_sst[:,ok_lat[0],:][:,:,ok_lon[0]]
ws_wind_speed = WS_wind_speed[:,ok_lat[0],:][:,:,ok_lon[0]]
'''

#%%

    plt.figure()
    plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    #plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    #plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    plt.contourf(ws_lon,ws_lat,ws_sst[0,:,:],cmap=cmocean.cm.thermal,**kw)
    cl = plt.colorbar()
    q = plt.quiver(ws_lon[::3], ws_lat[::3],ws_u_wind[0,::3,::3] , ws_v_wind[0,::3,::3],units='xy' ,scale=10)
    plt.quiverkey(q,-90,40,20,"20 m/s",coordinates='data',color='k')
    plt.ylim([lat_lim[0],lat_lim[1]])
    plt.xlim([lon_lim[0],lon_lim[1]])
    plt.title(WS_time[0])
    #plt.axis('scaled')

#%%

kw = dict(levels = np.linspace(20,35,16))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contourf(ws_lon,ws_lat,ws_sst[1,:,:]-273.15,cmap=cmocean.cm.thermal,**kw)
cl = plt.colorbar()
plt.ylim([lat_lim[0],lat_lim[1]])
plt.xlim([lon_lim[0],lon_lim[1]])
plt.axis('scaled')
plt.title(WS_time[0])

#%%

#kw = dict(levels = np.linspace(20,35,16))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contourf(ws_lon,ws_lat,ws_sst[1,:,:]-273.15,cmap=plt.get_cmap('Oranges')) #,**kw)
cl = plt.colorbar()
plt.ylim([lat_lim[0],lat_lim[1]])
plt.xlim([lon_lim[0],lon_lim[1]])
plt.axis('scaled')
