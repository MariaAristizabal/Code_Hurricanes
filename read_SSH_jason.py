#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:45:54 2019

@author: aristizabal
"""

import matplotlib.pyplot as plt
import xarray as xr
import cmocean
from bs4 import BeautifulSoup
import requests
import netCDF4
import numpy as np

import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#%% User input

# Directories where increments files reside 
#jason_file = '/Users/aristizabal/Desktop/JA2_GPSOPR_2PdS605_079_20180908_223104_20180909_002857.nc.nc4'

# Ocean Surface Topography Mission/Jason-2 (OSTM)
# https://podaac.jpl.nasa.gov/dataset/OSTM_L2_OST_OGDR_GPS
'''
This dataset is similar to the OSTM/Jason-2 Operation Geophysical Data Record (OGDR)
 that is distributed at NOAA (ftp://data.nodc.noaa.gov/pub/data.nodc/jason2/ogdr/), 
 but also includes a GPS based orbit and Sea Surface Height Anomalies (SSHA) 
 calculated from that orbit, as supposed to the DORIS based orbit that is already 
 available in the OGDR. It has a 5 hour time lag due to the time needed to 
 recalculate the orbit and SSHA. The GPS orbits have been shown to be more accurate 
 than the DORIS orbits on a near real time scale and therefore produces a more accurate 
 SSHA.
 '''

# Florence
#jason_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ostm/preview/L2/GPS-OGDR/c605/' 
#date = ['20180911','20180910','20180909','20180908','20180907']

# Michael
#jason_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ostm/preview/L2/GPS-OGDR/c608/'
#date = ['20181010','20181009','20181008','20181007','20181006']

jason_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/jason3/preview/L2/GPS-OGDR/'
cc = 'c161/'
date = ['20200701']

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

lon_lim = [-100,-10]
lat_lim = [0,50]  

#%% Find url list

r = requests.get(jason_url+cc)
data = r.text
soup = BeautifulSoup(data,"lxml")

fold = []
for s in soup.find_all("a"):
    fold.append(s.get("href").split('/')[0])
 
nc_list = []
for f in fold:
    elem = f.split('_')
    for l in elem:    
        for m in date:
            if l == m:
                nc_list.append(f.split('.')[0]+'.nc')
nc_list = list(set(nc_list))

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.lat
bath_lon = ncbath.lon
bath_elev = ncbath.elevation

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevsub = bath_elev[oklatbath,oklonbath]

#%%
'''
fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot()
ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='grey')
ax.axis('equal')
'''    
#%% global map

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())

for l in nc_list:
    ncjason = xr.open_dataset(jason_url + cc + l, decode_times=False) 
    #mssh = ncjason.mean_sea_surface
    ssha = np.asarray(ncjason.ssha)
    lat_jason = np.asarray(ncjason.lat)
    lon_jason = np.asarray(ncjason.lon)
    time_jason = ncjason.time
    time_jason = np.transpose(netCDF4.num2date(time_jason[:],time_jason.units))
    
    kw = dict(s=30, c=ssha, marker='*', edgecolor='none',vmin=-0.5,vmax=0.5)
    cs = ax.scatter(lon_jason, lat_jason, **kw, cmap=cmocean.cm.balance)

cbar = fig.colorbar(cs, orientation='vertical')
cbar.set_label('(m)',rotation=270,size = 18,labelpad = 20)
plt.title('{0} {1}-{2}-{3}'.format('Sea Surface Height Anomaly Jason3',\
          time_jason[0].year,time_jason[0].month,time_jason[0].day))
   
# Draw coastlines
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='lightblue', alpha=0.4)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}'.format('SSHA',\
          time_jason[0].year,time_jason[0].month,time_jason[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Atlantic map # 1
'''
fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())

for l in nc_list:
    print(l)
    ncjason = xr.open_dataset(jason_url + l, decode_times=False) 
    #mssh = ncjason.mean_sea_surface
    ssha = ncjason.ssha
    lat_jason = ncjason.lat
    lon_jason = ncjason.lon
    time_jason = ncjason.time
    time_jason = np.transpose(netCDF4.num2date(time_jason[:],time_jason.units))
    
    oklon = np.logical_and(lon_jason > lon_lim[0]+360,lon_jason < lon_lim[-1]+360)
    sshasub = ssha[oklon]
    lonsub = lon_jason[oklon]
    latsub = lat_jason[oklon]
    oklat = np.logical_and(latsub > lat_lim[0],latsub < lat_lim[-1])
    kw = dict(c=sshasub[oklat], marker='*', edgecolor='none',vmin=-0.5,vmax=0.5)
    cs = ax.scatter(lonsub[oklat], latsub[oklat], **kw, cmap=cmocean.cm.balance)

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.axis('equal')

# Draw coastlines
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='lightblue', alpha=0.4)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}.nc'.format('SSHA_Atlantic',\
          time_jason[0].year,time_jason[0].month,time_jason[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
'''

#%% Atlantic map # 2

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot()

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)
for l in nc_list:
    print(l)
    ncjason = xr.open_dataset(jason_url + cc + l, decode_times=False) 
    #mssh = ncjason.mean_sea_surface
    ssha = ncjason.ssha
    lat_jason = ncjason.lat
    lon_jason = ncjason.lon
    time_jason = ncjason.time
    time_jason = np.transpose(netCDF4.num2date(time_jason[:],time_jason.units))
    
    oklon = np.logical_and(lon_jason > lon_lim[0]+360,lon_jason < lon_lim[-1]+360)
    sshasub = ssha[oklon]
    lonsub = lon_jason[oklon]
    latsub = lat_jason[oklon]
    oklat = np.logical_and(latsub > lat_lim[0],latsub < lat_lim[-1])
    
    kw = dict(c=sshasub[oklat], marker='*', edgecolor='none',vmin=-0.5,vmax=0.5)
    cs = ax.scatter(lonsub[oklat]-360, latsub[oklat], **kw, cmap=plt.get_cmap('seismic'))

plt.xlim(-100,-10)
plt.ylim(0,50)
plt.grid('on')

cbar = fig.colorbar(cs, orientation='vertical')
cbar.set_label('(m)',rotation=270,size = 18,labelpad = 20)
plt.title('{0} {1}-{2}-{3}'.format('Sea Surface Height Anomaly Jason3',\
          time_jason[-1].year,time_jason[-1].month,time_jason[-1].day))

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}.png'.format('SSHA_Atlantic',\
          time_jason[-1].year,time_jason[-1].month,time_jason[-1].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Atlantic map # 2
'''
ncjason = xr.open_dataset(jason_url + nc_list[0], decode_times=False) 
#mssh = ncjason.mean_sea_surface
ssha = ncjason.ssha
lat_jason = ncjason.lat
lon_jason = ncjason.lon
time_jason = ncjason.time
time_jason = np.transpose(netCDF4.num2date(time_jason[:],time_jason.units))

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot()
oklon = np.logical_and(lon_jason > lon_lim[0]+360,lon_jason < lon_lim[-1]+360)
sshasub = ssha[oklon]
lonsub = lon_jason[oklon]
latsub = lat_jason[oklon]
oklat = np.logical_and(latsub > lat_lim[0],latsub < lat_lim[-1])

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)
#ax.axis('equal')

kw = dict(c=sshasub[oklat], marker='*', edgecolor='none',vmin=-0.5,vmax=0.5)
cs = ax.scatter(lonsub[oklat]-360, latsub[oklat], **kw, cmap=cmocean.cm.balance)

plt.xlim(-100,-10)
plt.ylim(0,50)
plt.grid('on')

cbar = fig.colorbar(cs, orientation='vertical')
cbar.set_label('(m)',rotation=270,size = 18,labelpad = 20)
plt.title('{0} {1}-{2}-{3}'.format('Sea Surface Height Anomaly',\
          time_jason[0].year,time_jason[0].month,time_jason[0].day))

'''
#%% Atlantic map # 1
'''
fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())

ncjason = xr.open_dataset(jason_url + nc_list[5], decode_times=False) 
ssha = ncjason.ssha
lat_jason = ncjason.lat
lon_jason = ncjason.lon
time_jason = ncjason.time
time_jason = np.transpose(netCDF4.num2date(time_jason[:],time_jason.units))
    
kw = dict(s=30, c=ssha, marker='*', edgecolor='none',vmin=-0.5,vmax=0.5)
cs = ax.scatter(lon_jason, lat_jason, **kw, cmap=cmocean.cm.balance)

cbar = fig.colorbar(cs, orientation='vertical')
cbar.set_label('(m)',rotation=270,size = 18,labelpad = 20)
plt.title('{0} {1}-{2}-{3}'.format('Sea Surface Height Anomaly',\
          time_jason[0].year,time_jason[0].month,time_jason[0].day))
   

# Draw coastlines
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='lightblue', alpha=0.4)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}.nc'.format('SSHA',\
          time_jason[0].year,time_jason[0].month,time_jason[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
'''