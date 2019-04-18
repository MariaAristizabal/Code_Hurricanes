#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:08:40 2019

@author: aristizabal
"""

#%% User input

# RU33 (MAB + SAB)
lon_lim = [-75,-70]
lat_lim = [36,42]
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

#Time window
date_ini = '2018-08-02T00:00:00Z'
date_end = '2018-08-03T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

# url for GOFS 3.1
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

#%%

from matplotlib import pyplot as plt
import numpy as np
#import xarray as xr
from netCDF4 import Dataset
import netCDF4

import datetime
#import matplotlib.dates as mdates

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
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

x1 = -74.32
y1 = 39.44
x2 = -73.52
y2 = 38.38
# Slope
m = (y1-y2)/(x1-x2)
# Intercept
b = y1 - m*x1

X = np.arange(x1,-72,0.05)
Y = b + m*X

#%%
# Conversion from glider longitude and latitude to GOFS convention
target_lon = np.empty((len(X),))
target_lon[:] = np.nan
for i,ii in enumerate(X):
    if ii < 0: 
        target_lon[i] = 360 + ii
    else:
        target_lon[i] = ii
target_lat = Y

dist = np.sqrt((X-x1)**2 + (Y-y1)**2)*111 # approx distance from shore in km

#%%

fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
#plot_date = mdates.num2date(time31[ind])
plt.title('Cross Shelf Transect')
    
ax.contour(bath_lonsub,bath_latsub,bath_elevsub,colors='k')
ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

ax.plot(X,Y,'.-k')

file = folder + 'MAB_cross_shelf_transect'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Read GOGF 3.1

#GOFS31 = xr.open_dataset(catalog31,decode_times=False)
GOFS31 = Dataset(catalog31,decode_times=False)

#%% Find cross shelf transect in GOFS 3.1
    
lat31 = GOFS31['lat'][:]
lon31 = GOFS31['lon'][:]
depth31 = GOFS31['depth'][:]
tt31= GOFS31['time']
t31 = netCDF4.num2date(tt31[:],tt31.units) 

tmin = datetime.datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

oktime31 = np.where(np.logical_and(t31 >= tmin, t31 <= tmax))
time31 = t31[oktime31]

# interpolating transect X and Y to lat and lon 
oklon31=np.round(np.interp(target_lon,lon31,np.arange(0,len(lon31)))).astype(int)
oklat31=np.round(np.interp(target_lat,lat31,np.arange(0,len(lat31)))).astype(int)

#%%
'''
# Get cross channel transect
target_temp31 = np.empty((len(depth31),len(target_lon)))
target_temp31[:] = np.nan
for t,tt in enumerate(oktime31[0]):
    target_temp31[:,i] = GOFS31.variables['water_temp'][oktime31[0][i],:,oklat31[i],oklon31[i]]

target_temp31[target_temp31 < -100] = np.nan

#%%
target_temp31 = np.empty((len(depth31),len(target_lon)))
target_temp31[:] = np.nan
tind = oktime31[0][0]
for pos in range(len(oklon31)):
    print(len(oklon31),pos)
    target_temp31[:,pos] = GOFS31.variables['water_temp'][tind,:,oklat31[pos],oklon31[pos]]

target_temp31[target_temp31 < -100] = np.nan
'''

#%%   
max_val = 27
min_val = 7   
nlevels = max_val - min_val + 1

target_temp31 = np.empty((len(depth31),len(target_lon)))
target_temp31[:] = np.nan
for t,tind in enumerate(oktime31[0]):
    print('tind = ',tind)
    for pos in range(len(oklon31)):
        print(len(oklon31),pos)
        target_temp31[:,pos] = GOFS31.variables['water_temp'][tind,:,oklat31[pos],oklon31[pos]]
    target_temp31[target_temp31 < -100] = np.nan

    fig, ax = plt.subplots(figsize=(6, 3))
    kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    cs = plt.contourf(dist,-depth31,target_temp31,cmap='RdYlBu_r',**kw)
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('Temperature',size=14)
        
    plt.contour(dist,-depth31,target_temp31,levels=10,colors='k')
    plt.title('Cross Shore Transect on ' + t31[tind].strftime("%Y-%m-%d %H"),size=16)
        
    ax.set_xlim(0,200)
    ax.set_ylim(-100,0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Distance from shore (km)',fontsize=14)

    file = folder + 'cross_shore_transect_'+ \
                        t31[tind].strftime("%Y-%m-%d-%H")
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 