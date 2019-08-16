#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:14:17 2019

@author: aristizabal
"""

#%% User input

Aqua_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/modis/L3/aqua/11um/v2014.0/4km/daily/'
Terra_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/modis/L3/terra/11um/v2014.0/4km/daily/'

date_ini = '2018-07-17T00:00:00Z'
date_end = '2018-09-18T00:00:00Z'

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

#%% Read Aqua and Terra SST 

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

kw = dict(levels = np.linspace(20,35,7))

#for t,tt in enumerate(np.arange((tend-tini).days+1)):
for t,tt in enumerate(np.arange(2)):
    tdate = tini + timedelta(t)
    
    # Aqua
    file_name = 'A' + str(tdate.year) + tdate.strftime('%j') + \
                '.L3m_DAY_SST_sst_4km.nc'
    Aqua_url_nc = Aqua_url + str(tdate.year) + '/' +  tdate.strftime('%j') + '/' + \
                     file_name               
    #urllib.request.urlretrieve(WindSat_url_nc, folder_nc+file_name)               
    Aqua = xr.open_dataset(Aqua_url_nc)
    
    #Aqua_time = np.asarray(Aqua.variables['time'][:])
    Aqua_lat = np.asarray(Aqua.variables['lat'][:])
    Aqua_lon = np.asarray(Aqua.variables['lon'][:])

    ok_lon = np.where(np.logical_and(Aqua_lon >= lon_lim[0],Aqua_lon <= lon_lim[1]))
    ok_lat = np.where(np.logical_and(Aqua_lat >= lat_lim[0],Aqua_lat <= lat_lim[1]))

    Aqua_sst = np.asarray(Aqua.variables['sst'][:])
    
    aqua_lon = Aqua_lon[ok_lon[0]]
    aqua_lat = Aqua_lat[ok_lat[0]]
    aqua_sst = Aqua_sst[ok_lat[0],:][:,ok_lon[0]]
    
    plt.figure()
    plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    #plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    #plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    #plt.contour(aqua_lon,aqua_lat,aqua_sst[:,:],colors='k',**kw)
    plt.contourf(aqua_lon,aqua_lat,aqua_sst[:,:],cmap=cmocean.cm.thermal,**kw)
    cl = plt.colorbar()
    plt.ylim([lat_lim[0],lat_lim[1]])
    plt.xlim([lon_lim[0],lon_lim[1]])
    plt.title('Aqua L3 SST Thermal-IR Daily 4 km Daytime \n' + str(tdate))
    #plt.axis('scaled')
    
    # Terra
    file_name = 'T' + str(tdate.year) + tdate.strftime('%j') + \
                '.L3m_DAY_SST_sst_4km.nc'
    Terra_url_nc = Terra_url + str(tdate.year) + '/' +  tdate.strftime('%j') + '/' + \
                     file_name               
    #urllib.request.urlretrieve(WindSat_url_nc, folder_nc+file_name)               
    Terra = xr.open_dataset(Terra_url_nc)
    
    #Aqua_time = np.asarray(Aqua.variables['time'][:])
    Terra_lat = np.asarray(Terra.variables['lat'][:])
    Terra_lon = np.asarray(Terra.variables['lon'][:])

    ok_lon = np.where(np.logical_and(Terra_lon >= lon_lim[0],Terra_lon <= lon_lim[1]))
    ok_lat = np.where(np.logical_and(Terra_lat >= lat_lim[0],Terra_lat <= lat_lim[1]))

    Terra_sst = np.asarray(Terra.variables['sst'][:])
    
    terra_lon = Terra_lon[ok_lon[0]]
    terra_lat = Terra_lat[ok_lat[0]]
    terra_sst = Terra_sst[ok_lat[0],:][:,ok_lon[0]]
    
    plt.figure()
    plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    #plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    #plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    #plt.contour(aqua_lon,aqua_lat,aqua_sst[:,:],colors='k',**kw)
    plt.contourf(terra_lon,terra_lat,terra_sst[:,:],cmap=cmocean.cm.thermal,**kw)
    cl = plt.colorbar()
    plt.ylim([lat_lim[0],lat_lim[1]])
    plt.xlim([lon_lim[0],lon_lim[1]])
    plt.title('Terra L3 SST Thermal-IR Daily 4 km Daytime \n' + str(tdate))
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