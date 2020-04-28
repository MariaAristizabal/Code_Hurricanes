#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:19:46 2018

@author: aristizabal
"""

#GOFS3.1 outout model location
#url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
url_GOFS1 = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GOFS_2019082518_Caribbean_ts3z.nc'
url_GOFS2 = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GOFS_2019083018_Caribbean_ts3z.nc'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'
#bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

date_ini = '2019/08/25/18' # year/month/day/hour
date_end = '2019/08/30/18' # year/month/day/hour

#lat_lim = [-5,50]
#lon_lim = [260-360,360-360]

lat_lim = [10,30]
lon_lim = [-90,-60]

#catalog31 = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/hycom_glbv_930_2018010112_t000_ts3z.nc'

#%%

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import xarray as xr
import netCDF4
import cmocean                           
from datetime import datetime
                             
#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%%
GOFS1 = xr.open_dataset(url_GOFS1,decode_times=False)

latGOFS = np.asarray(GOFS1.lat[:])
lonGOFS = np.asarray(GOFS1.lon[:])
depth_GOFS = np.asarray(GOFS1.depth[:])

oklat_GOFS = np.where(np.logical_and(latGOFS > lat_lim[0], latGOFS < lat_lim[1]))[0]
oklon_GOFS = np.where(np.logical_and(lonGOFS > lon_lim[0]+360, lonGOFS < lon_lim[1]+360))[0]

#%% Figure temperature caribbean

tGOFS1 = GOFS1.time
time_GOFS1 = netCDF4.num2date(tGOFS1[:],tGOFS1.units) 

tini = datetime.strptime(date_ini, '%Y/%m/%d/%H') 
tend = datetime.strptime(date_end, '%Y/%m/%d/%H')

oktime_GOFS1 = np.where(np.logical_and(time_GOFS1 >= tini,time_GOFS1 <= tend))[0]

temp_GOFS = np.asarray(GOFS1.variables['water_temp'][oktime_GOFS1[0],0,oklat_GOFS,oklon_GOFS])
lat_GOFS = latGOFS[oklat_GOFS]
lon_GOFS = lonGOFS[oklon_GOFS]

kw = dict(levels = np.linspace(25,33,17))
fig, ax = plt.subplots(figsize=(7,3.5))
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
cs = plt.contourf(lon_GOFS-360,lat_GOFS,temp_GOFS,cmap=cmocean.cm.thermal,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
plt.title('Surface Temperature GOFS '+str(time_GOFS1[0])[0:13],fontsize=14)
plt.axis('scaled')

file = folder_fig + 'surf_temp_GOFS_Carib_' + str(time_GOFS1[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)


#%% Figure salinity caribbean

salt_GOFS = np.asarray(GOFS1.variables['salinity'][oktime_GOFS1[0],0,oklat_GOFS,oklon_GOFS])
lat_GOFS = latGOFS[oklat_GOFS]
lon_GOFS = lonGOFS[oklon_GOFS]

kw = dict(levels = np.linspace(33,37,17))
fig, ax = plt.subplots(figsize=(7,3.5))
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
cs = plt.contourf(lon_GOFS-360,lat_GOFS,salt_GOFS,cmap=cmocean.cm.haline,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
plt.axis('scaled')
#cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
plt.title('Surface Salinity GOFS '+str(time_GOFS1[0])[0:13][0:13])

file = folder_fig + 'surf_salt_GOFS_Carib_' + str(time_GOFS1[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure temperature caribbean

GOFS2 = xr.open_dataset(url_GOFS2,decode_times=False)

tGOFS2 = GOFS2.time
time_GOFS2 = netCDF4.num2date(tGOFS2[:],tGOFS2.units) 

tini = datetime.strptime(date_ini, '%Y/%m/%d/%H') 
tend = datetime.strptime(date_end, '%Y/%m/%d/%H')

oktime_GOFS2 = np.where(np.logical_and(time_GOFS2 >= tini,time_GOFS2 <= tend))[0]

temp_GOFS = np.asarray(GOFS2.variables['water_temp'][oktime_GOFS2[0],0,oklat_GOFS,oklon_GOFS])
lat_GOFS = latGOFS[oklat_GOFS]
lon_GOFS = lonGOFS[oklon_GOFS]

kw = dict(levels = np.linspace(25,33,17))
fig, ax = plt.subplots(figsize=(7,3.5))
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
cs = plt.contourf(lon_GOFS-360,lat_GOFS,temp_GOFS,cmap=cmocean.cm.thermal,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
plt.title('Surface Temperature GOFS '+str(time_GOFS2[0])[0:13],fontsize=14)
plt.axis('scaled')

file = folder_fig + 'surf_temp_GOFS_Carib_' + str(time_GOFS2[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)


#%% Figure salinity caribbean

salt_GOFS = np.asarray(GOFS2.variables['salinity'][oktime_GOFS2[0],0,oklat_GOFS,oklon_GOFS])
lat_GOFS = latGOFS[oklat_GOFS]
lon_GOFS = lonGOFS[oklon_GOFS]

kw = dict(levels = np.linspace(33,37,17))
fig, ax = plt.subplots(figsize=(7,3.5))
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
cs = plt.contourf(lon_GOFS-360,lat_GOFS,salt_GOFS,cmap=cmocean.cm.haline,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
plt.axis('scaled')
#cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
plt.title('Surface Salinity GOFS '+str(time_GOFS2[0])[0:13])

file = folder_fig + 'surf_salt_GOFS_Carib_' + str(time_GOFS2[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure salinity caribbean

lat_lim = [10,22.5]
lon_lim = [-75,-60]

oklat_GOFS = np.where(np.logical_and(latGOFS > lat_lim[0], latGOFS < lat_lim[1]))[0]
oklon_GOFS = np.where(np.logical_and(lonGOFS > lon_lim[0]+360, lonGOFS < lon_lim[1]+360))[0]

salt_GOFS = np.asarray(GOFS1.variables['salinity'][oktime_GOFS1[0],0,oklat_GOFS,oklon_GOFS])
lat_GOFS = latGOFS[oklat_GOFS]
lon_GOFS = lonGOFS[oklon_GOFS]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

kw = dict(levels = np.linspace(33,37,17))
fig, ax = plt.subplots(figsize=(6,3.5))
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
cs = plt.contourf(lon_GOFS-360,lat_GOFS,salt_GOFS,cmap=cmocean.cm.haline,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
#cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
plt.title('Surface Salinity GOFS '+str(time_GOFS1[0])[0:13][0:13],fontsize=14)
plt.axis('scaled')
plt.yticks([])
plt.xticks([])

file = folder_fig + 'surf_salt_GOFS_Carib1_' + str(time_GOFS1[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure salinity caribbean

lat_lim = [10,22.5]
lon_lim = [-75,-60]

oklat_GOFS = np.where(np.logical_and(latGOFS > lat_lim[0], latGOFS < lat_lim[1]))[0]
oklon_GOFS = np.where(np.logical_and(lonGOFS > lon_lim[0]+360, lonGOFS < lon_lim[1]+360))[0]

salt_GOFS = np.asarray(GOFS2.variables['salinity'][oktime_GOFS2[0],0,oklat_GOFS,oklon_GOFS])
lat_GOFS = latGOFS[oklat_GOFS]
lon_GOFS = lonGOFS[oklon_GOFS]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

kw = dict(levels = np.linspace(33,37,17))
fig, ax = plt.subplots(figsize=(6,3.5))
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
cs = plt.contourf(lon_GOFS-360,lat_GOFS,salt_GOFS,cmap=cmocean.cm.haline,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
plt.title('Surface Salinity GOFS '+str(time_GOFS2[0])[0:13][0:13],fontsize=14)
plt.axis('scaled')
plt.yticks([])
plt.xticks([])

file = folder_fig + 'surf_salt_GOFS_Carib2_' + str(time_GOFS2[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure Temperature
'''
fig = plt.figure(num=None, figsize=(12, 8) )
m = Basemap(projection='merc',llcrnrlat=-5,urcrnrlat=40,llcrnrlon=260,urcrnrlon=360,resolution='c')
m.drawcoastlines()
x, y = m(*np.meshgrid(lon31[oklon31],lat31[oklat31]))
m.pcolormesh(x,y,temp31,shading='flat',cmap=plt.cm.jet)
m.colorbar(location='right')
#m.fillcontinents(color='lightgrey',lake_color='lightblue')
'''