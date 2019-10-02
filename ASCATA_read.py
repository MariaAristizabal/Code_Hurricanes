#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:39:14 2019

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-100.0,-40.0]
lat_lim = [0,50.0]

ASCATA_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ascat/preview/L2/metop_a/coastal_opt/2018/'

day_ini = '151' # June 1/2018
day_end = '332' # Nov 29/2018

#ASCATA_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ascat/preview/L2/metop_a/coastal_opt/2018/151/ascat_20180531_013600_metopa_60257_eps_o_coa_2401_ovw.l2.nc.gz'
#ASVATA_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ascat/preview/L2/metop_a/coastal_opt/2018/151/ascat_20180531_031800_metopa_60258_eps_o_coa_2401_ovw.l2.nc.gz'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

#%%

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
#from datetime import datetime, timedelta

from bs4 import BeautifulSoup
import requests

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

#%%
             
ASCATA = xr.open_dataset(ASCATA_url)
   
AS_time = np.asarray(ASCATA.variables['time'][:])
AS_lat = np.asarray(ASCATA.variables['lat'][:])
AS_lon = np.asarray(ASCATA.variables['lon'][:])
AS_wind_speed = np.asarray(ASCATA.variables['wind_speed'][:])

#%% Changing lon convention to glider convention

AS_long = np.empty((AS_lon.shape[0],AS_lon.shape[1]))
AS_long[:] = np.nan

for i,swath in enumerate(AS_lon.T):
    #print(i,swath.shape)
    for l,ln in enumerate(swath):
        if ln > 180:
            AS_long[l,i] = ln - 360
        else:
            AS_long[l,i] = ln
            
AS_latg = AS_lat

#%%
kw = dict(c=AS_wind_speed[0,:], marker='o', edgecolor='none')

fig, ax = plt.subplots(figsize=(10, 5))
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.axis([-100,-40,0,50])
plt.scatter(AS_long[0,:],AS_latg[0,:],**kw,cmap='RdYlBu_r')
plt.colorbar()

#%%
kw = dict(c=AS_wind_speed, marker='o', edgecolor='none')

fig, ax = plt.subplots(figsize=(10, 5))
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
#plt.axis([-100,-10,0,50])
plt.scatter(AS_long,AS_latg,**kw,cmap='RdYlBu_r')
plt.colorbar()

#%%

days = np.arange(int(day_ini),int(day_end)+1)

for day in days[0:5]:

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
    plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')

    r = requests.get(ASCATA_url + str(day))
    data = r.text
    soup = BeautifulSoup(data,"lxml")

    ascata_files = []
    for s in soup.find_all("a"):
        if s.get('href').split('_')[0] == 'ascat':
            if s.get('href').split('.')[-1] == 'info':
                ascata_files.append(s.get('href').split('.')[0]+'.l2.nc.gz')

    for file in ascata_files:
                
        ASCATA = xr.open_dataset(ASCATA_url + str(day) + '/' + file)
   
        AS_time = np.asarray(ASCATA.variables['time'][:])
        AS_lat = np.asarray(ASCATA.variables['lat'][:])
        AS_lon = np.asarray(ASCATA.variables['lon'][:])
        AS_wind_speed = np.asarray(ASCATA.variables['wind_speed'][:])

        AS_long = np.empty((AS_lon.shape[0],AS_lon.shape[1]))
        AS_long[:] = np.nan
        for i,swath in enumerate(AS_lon.T):
            for l,ln in enumerate(swath):
                if ln > 180:
                    AS_long[l,i] = ln - 360
                else:
                    AS_long[l,i] = ln
            
        AS_latg = AS_lat

        ok_lon = np.logical_and(AS_long >= lon_lim[0],AS_long <= lon_lim[-1])   
    
        As_long = AS_long[ok_lon]   
        As_latg = AS_latg[ok_lon]
        As_time = AS_time[ok_lon]
        As_wind_speed = AS_wind_speed[ok_lon]

        ok_lat = np.logical_and(As_latg >= lat_lim[0],As_latg <= lat_lim[-1]) 

        as_long = As_long[ok_lat]
        as_latg = As_latg[ok_lat]
        as_time = As_time[ok_lat]
        as_wind_speed = As_wind_speed[ok_lat]

        kw = dict(c=as_wind_speed, marker='o', edgecolor='none')
        plt.scatter(as_long,as_latg,**kw,cmap='RdYlBu_r')
    
    plt.colorbar()
