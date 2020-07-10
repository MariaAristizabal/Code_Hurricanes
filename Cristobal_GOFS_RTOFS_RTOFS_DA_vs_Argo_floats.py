#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:32:41 2020

@author: aristizabal
"""

#%% User input

# GoMex
lon_lim = [-98,80]
lat_lim = [15,32.5]

#GOFS3.1 output model location
url_GOFS_ts = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'

# RTOFS files
folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']
    
# RTOFS-DA files
folder_RTOFS_DA = '/home/aristizabal/RTOFS-DA/data_'
prefix_RTOFS_DA = 'archv' 

# RTOFS grid file name
folder_RTOFS_DA_grid_depth = '/home/aristizabal/RTOFS-DA/GRID_DEPTH/'
RTOFS_DA_grid = folder_RTOFS_DA_grid_depth + 'regional.grid'
RTOFS_DA_depth = folder_RTOFS_DA_grid_depth + 'regional.depth'   

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'
# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# Argo floats
#Argo_nc = '/Users/Aristizabal/Desktop/ArgoFloats_eb1e_1977_1b02.nc'
Argo_nc = '/home/aristizabal/ARGO_data/Hurric_season_2020/ArgoFloats_eb1e_1977_1b02.nc'

#folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
folder_fig = '/home/aristizabal/Figures/'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 
from datetime import datetime, timedelta
import cmocean
from netCDF4 import Dataset
import matplotlib.dates as mdates
import glob
import os

import sys
sys.path.append('/home/aristizabal/NCEP_scripts/')
from utils4HYCOM import readgrids,readdepth, readVar

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% GOGF 3.1

GOFS_ts = xr.open_dataset(url_GOFS_ts,decode_times=False)

latt31 = np.asarray(GOFS_ts['lat'][:])
lonn31 = np.asarray(GOFS_ts['lon'][:])
tt31 = GOFS_ts['time']
t_GOFS = netCDF4.num2date(tt31[:],tt31.units) 

depth_GOFS = np.asarray(GOFS_ts['depth'][:])

tt = GOFS_ts['time']
time_forec = netCDF4.num2date(tt[:],tt.units) 

# Conversion from glider longitude and latitude to GOFS convention
lon_limG = np.empty((len(lon_lim),))
lon_limG[:] = np.nan
for i in range(len(lon_lim)):
    if lon_lim[i] < 0: 
        lon_limG[i] = 360 + lon_lim[i]
    else:
        lon_limG[i] = lon_lim[i]
lat_limG = lat_lim

### Build the bbox for the xy data
botm  = int(np.where(latt31 > lat_limG[0])[0][0])
top   = int(np.where(latt31 > lat_limG[1])[0][0])
left  = np.where(lonn31 > lon_limG[0])[0][0]
right = np.where(lonn31 > lon_limG[1])[0][0]

lat31= latt31[botm:top]
lon31= lonn31[left:right]

# Conversion from GOFS convention to glider longitude and latitude
lon31g= np.empty((len(lon31),))
lon31g[:] = np.nan
for i in range(len(lon31)):
    if lon31[i] > 180: 
        lon31g[i] = lon31[i] - 360 
    else:
        lon31g[i] = lon31[i]
lat31g = lat31

#%% Reading RTOFS-DA grid
print('Retrieving coordinates from RTOFS_DA')
# Reading lat and lon
lines_grid = [line.rstrip() for line in open(RTOFS_DA_grid+'.b')]
lon_RTOFS_DA = np.array(readgrids(RTOFS_DA_grid,'plon:',[0]))
lat_RTOFS_DA = np.array(readgrids(RTOFS_DA_grid,'plat:',[0]))

depth_RTOFS_DA = np.asarray(readdepth(RTOFS_DA_depth,'depth'))

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

#%% Reading Argo data

ncargo = Dataset(Argo_nc)
argo_lat = np.asarray(ncargo.variables['latitude'][:])
argo_lon = np.asarray(ncargo.variables['longitude'][:])
argo_tim = ncargo.variables['time']#[:]
argo_time = netCDF4.num2date(argo_tim[:],argo_tim.units)  
argo_pres = np.asarray(ncargo.variables['pres'][:])
argo_temp = np.asarray(ncargo.variables['temp'][:])
argo_salt = np.asarray(ncargo.variables['psal'][:])
argo_id = ncargo.variables['platform_number'][:]

#%% Map Argo floats
 
lev = np.arange(-9000,9100,100)
plt.figure()
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo) 
#plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.plot(argo_lon,argo_lat,'s',color='g',markersize=3,markeredgecolor='k')
plt.title('Argo Floats ' + str(argo_time[0])[0:10]+'-'+str(argo_time[-1])[0:10],fontsize=14)
plt.axis('scaled')
plt.xlim(-98,-79.5)
plt.ylim(15,32.5)

file = folder_fig + 'ARGO_lat_lon_' + str(argo_time[0])[0:10]+'-'+str(argo_time[-1])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure argo float vs GOFS, RTOFS and RTOFS-DA

##### Argo
oklon = np.where(np.logical_and(argo_lon > -90.5,\
                argo_lon < -89.5))

argo_ids = argo_id[oklon]   
argo_lons = argo_lon[oklon]
argo_lats = argo_lat[oklon]
argo_tts = argo_time[oklon]
argo_depths = argo_pres[oklon]
argo_temps = argo_temp[oklon]
argo_salts = argo_salt[oklon]

oklat = np.where(np.logical_and(argo_lats > 26.5,\
                argo_lats < 27))

argo_idss = argo_ids[oklat] 
#id_uniq, ind = np.unique(argo_id[oklon],return_index=True)
argo_idss = argo_ids[oklat]   
argo_lonss = argo_lons[oklat]
argo_latss = argo_lats[oklat]
argo_ttss = argo_tts[oklat]
argo_depthss = argo_depths[oklat]
argo_tempss = argo_temps[oklat]
argo_saltss = argo_salts[oklat]

########### GOFS
oktt_GOFS = np.where(t_GOFS >= argo_ttss[0])[0][0]
oklat_GOFS = np.where(latt31 >= argo_latss[0])[0][0]
oklon_GOFS = np.where(lonn31 >= argo_lonss[0]+360)[0][0]
temp_GOFS = np.asarray(GOFS_ts['water_temp'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])
salt_GOFS = np.asarray(GOFS_ts['salinity'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])

############ RTOFS
#Time window
year = int(argo_ttss[0].year)
month = int(argo_ttss[0].month)
day = int(argo_ttss[0].day)
tini = datetime(year, month, day)
tend = tini + timedelta(days=1)

# Read RTOFS grid and time
print('Retrieving coordinates from RTOFS')

if tini.month < 10:
    if tini.day < 10:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day)
else:
    if tini.day < 10:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + str(tini.day)

ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[0])
latRTOFS = np.asarray(ncRTOFS.Latitude[:])
lonRTOFS = np.asarray(ncRTOFS.Longitude[:])
depth_RTOFS = np.asarray(ncRTOFS.Depth[:])

tRTOFS = []
for t in np.arange(len(nc_files_RTOFS)):
    ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
          for t in np.arange(len(nc_files_RTOFS))])

oktt_RTOFS = np.where(mdates.date2num(tRTOFS) >= mdates.date2num(argo_ttss[0]))[0][0]
oklat_RTOFS = np.where(latRTOFS[:,0] >= argo_latss[0])[0][0]
oklon_RTOFS = np.where(lonRTOFS[0,:]  >= argo_lonss[0])[0][0]
    
nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[oktt_RTOFS]
ncRTOFS = xr.open_dataset(nc_file)
time_RTOFS = tRTOFS[oktt_RTOFS]
temp_RTOFS = np.asarray(ncRTOFS.variables['temperature'][0,:,oklat_RTOFS,oklon_RTOFS])
salt_RTOFS = np.asarray(ncRTOFS.variables['salinity'][0,:,oklat_RTOFS,oklon_RTOFS])
lon_RTOFS = lonRTOFS[0,oklon_RTOFS]
lat_RTOFS = latRTOFS[oklat_RTOFS,0]

############# RTOFS-DA
# cycle
year = int(argo_ttss[0].year)
month = int(argo_ttss[0].month)
day = int(argo_ttss[0].day)
tini = datetime(year, month, day)
tend = tini + timedelta(days=1)

if tini.month < 10:
    if tini.day < 10:
        cycle = str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day) + '00'
    else:
        cycle = str(tini.year) + '0' + str(tini.month) + str(tini.day) + '00'
else:
    if cycle < 10:
        cycle = str(tini.year) + str(tini.month) + '0' + str(tini.day) + '00'
    else:
        cycle = str(tini.year) + str(tini.month) + str(tini.day) + '00'

afiles = sorted(glob.glob(os.path.join(folder_RTOFS_DA+cycle,prefix_RTOFS_DA+'*.a')))
       
tRTOFS_DA = []
for file in afiles:
    print(file)
    lines = [line.rstrip() for line in open(file[:-2]+'.b')]
    time_stamp = lines[-1].split()[2]
    hycom_days = lines[-1].split()[3]
    tzero=datetime(1901,1,1,0,0)
    tRTOFS_DA.append(tzero+timedelta(float(hycom_days)-1))

tRTOFS_DA = np.asarray(tRTOFS_DA)

oktt_RTOFS_DA = np.where(mdates.date2num(tRTOFS_DA) >= mdates.date2num(argo_ttss[0]-timedelta(hours=2)))[0][0]

file = afiles[oktt_RTOFS_DA]
time_RTOFS_DA = tRTOFS_DA[oktt_RTOFS_DA]

nz = 41
layers = np.arange(0,nz)
temp_RTOFS_DA = np.empty((nz))
temp_RTOFS_DA[:] = np.nan
salt_RTOFS_DA = np.empty((nz))
salt_RTOFS_DA[:] = np.nan
zRTOFS_DA = np.empty((nz))
zRTOFS_DA[:] = np.nan

oklat_RTOFS_da = np.where(lat_RTOFS_DA[:,0] >= argo_latss[0])[0][0]
oklon_RTOFS_da = np.where(lon_RTOFS_DA[0,:]  >= argo_lonss[0]+ 360)[0][0]

ztmp = [0]
for lyr in tuple(layers):
    print(lyr)
    temp_RTOFSDA = readVar(file[:-2],'archive','temp',[lyr+1])
    temp_RTOFS_DA[lyr] = temp_RTOFSDA[oklat_RTOFS_da,oklon_RTOFS_da]
    
    salt_RTOFSDA = readVar(file[:-2],'archive','salin',[lyr+1])
    salt_RTOFS_DA[lyr] = salt_RTOFSDA[oklat_RTOFS_da,oklon_RTOFS_da]
    
    dp = readVar(file[:-2],'archive','thknss',[lyr+1])/9806
    ztmp = np.append(ztmp,dp[oklat_RTOFS_da,oklon_RTOFS_da])
    
zRTOFS_DA = np.cumsum(ztmp[0:-1]) + np.diff(np.cumsum(ztmp))/2    

###############################

# Figure temp
plt.figure(figsize=(5,6))
plt.plot(argo_tempss,-argo_depthss,'.-',linewidth=2,label='ARGO Float id'+argo_idss[0])
plt.plot(temp_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1',color='red')
plt.plot(temp_RTOFS,-depth_RTOFS,'.-',linewidth=2,label='RTOFS',color='g')
plt.plot(temp_RTOFS_DA,-zRTOFS_DA,'.-',linewidth=2,label='RTOFS-DA',color='sandybrown')
#plt.ylim([-1000,0])
#plt.xlim([5,28])
plt.ylim([-200,0])
plt.xlim([15,28])
plt.title('Temperature Profile on '+ str(argo_ttss[0])[0:13] +
          '\n [lon,lat] = [' \
          + str(np.round(argo_lonss[0],3)) +',' +\
              str(np.round(argo_latss[0],3))+']',\
              fontsize=16)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('$^oC$',fontsize=14)
plt.legend(loc='lower right',fontsize=14)

file = folder_fig + 'temp_ARGO_vs_GOFS_RTOFS_RTOFS_DA_' + str(argo_ttss[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

# Figure salt
plt.figure(figsize=(5,6))
plt.plot(argo_saltss,-argo_depthss,'.-',linewidth=2,label='ARGO Float id'+argo_idss[0])
plt.plot(salt_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1',color='red')
plt.plot(salt_RTOFS,-depth_RTOFS,'.-',linewidth=2,label='RTOFS',color='g')
plt.plot(salt_RTOFS_DA,-zRTOFS_DA,'.-',linewidth=2,label='RTOFS-DA',color='sandybrown')
#plt.ylim([-1000,0])
plt.ylim([-200,0])
plt.xlim([36,37])
plt.title('Salinity Profile on '+ str(argo_ttss[0])[0:13] +
          '\n [lon,lat] = [' \
          + str(np.round(argo_lonss[0],3)) +',' +\
              str(np.round(argo_latss[0],3))+']',\
              fontsize=16)
plt.ylabel('Depth (m)',fontsize=14)
plt.legend(loc='lower right',fontsize=12)

file = folder_fig + 'salt_ARGO_vs_GOFS_RTOFS_RTOFS_DA_' + str(argo_ttss[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure argo float vs GOFS

######## Argo
oklon = np.where(np.logical_and(argo_lon > -90.5,\
                argo_lon < -89.5))

argo_ids = argo_id[oklon]   
argo_lons = argo_lon[oklon]
argo_lats = argo_lat[oklon]
argo_tts = argo_time[oklon]
argo_depths = argo_pres[oklon]
argo_temps = argo_temp[oklon]
argo_salts = argo_salt[oklon]

oklat = np.where(np.logical_and(argo_lats > 24.8,\
                argo_lats < 25))

argo_idss = argo_ids[oklat] 
#id_uniq, ind = np.unique(argo_id[oklon],return_index=True)
argo_idss = argo_ids[oklat]   
argo_lonss = argo_lons[oklat]
argo_latss = argo_lats[oklat]
argo_ttss = argo_tts[oklat]
argo_depthss = argo_depths[oklat]
argo_tempss = argo_temps[oklat]
argo_saltss = argo_salts[oklat]

########## GOFS
oktt_GOFS = np.where(t_GOFS >= argo_ttss[0])[0][0]
oklat_GOFS = np.where(latt31 >= argo_latss[0])[0][0]
oklon_GOFS = np.where(lonn31 >= argo_lonss[0]+360)[0][0]
temp_GOFS = np.asarray(GOFS_ts['water_temp'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])
salt_GOFS = np.asarray(GOFS_ts['salinity'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])

########### RTOFS
#Time window
year = int(argo_ttss[0].year)
month = int(argo_ttss[0].month)
day = int(argo_ttss[0].day)
tini = datetime(year, month, day)
tend = tini + timedelta(days=1)

# Read RTOFS grid and time
print('Retrieving coordinates from RTOFS')

if tini.month < 10:
    if tini.day < 10:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day)
else:
    if tini.day < 10:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + str(tini.day)

ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[0])
latRTOFS = np.asarray(ncRTOFS.Latitude[:])
lonRTOFS = np.asarray(ncRTOFS.Longitude[:])
depth_RTOFS = np.asarray(ncRTOFS.Depth[:])

tRTOFS = []
for t in np.arange(len(nc_files_RTOFS)):
    ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
          for t in np.arange(len(nc_files_RTOFS))])

oktt_RTOFS = np.where(mdates.date2num(tRTOFS) >= mdates.date2num(argo_ttss[0]))[0][0]
oklat_RTOFS = np.where(latRTOFS[:,0] >= argo_latss[0])[0][0]
oklon_RTOFS = np.where(lonRTOFS[0,:]  >= argo_lonss[0])[0][0]
    
nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[oktt_RTOFS]
ncRTOFS = xr.open_dataset(nc_file)
time_RTOFS = tRTOFS[oktt_RTOFS]
temp_RTOFS = np.asarray(ncRTOFS.variables['temperature'][0,:,oklat_RTOFS,oklon_RTOFS])
salt_RTOFS = np.asarray(ncRTOFS.variables['salinity'][0,:,oklat_RTOFS,oklon_RTOFS])
lon_RTOFS = lonRTOFS[0,oklon_RTOFS]
lat_RTOFS = latRTOFS[oklat_RTOFS,0]

############# RTOFS-DA
# cycle
year = int(argo_ttss[0].year)
month = int(argo_ttss[0].month)
day = int(argo_ttss[0].day)
tini = datetime(year, month, day)
tend = tini + timedelta(days=1)

if tini.month < 10:
    if tini.day < 10:
        cycle = str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day) + '00'
    else:
        cycle = str(tini.year) + '0' + str(tini.month) + str(tini.day) + '00'
else:
    if cycle < 10:
        cycle = str(tini.year) + str(tini.month) + '0' + str(tini.day) + '00'
    else:
        cycle = str(tini.year) + str(tini.month) + str(tini.day) + '00'

afiles = sorted(glob.glob(os.path.join(folder_RTOFS_DA+cycle,prefix_RTOFS_DA+'*.a')))
       
tRTOFS_DA = []
for file in afiles:
    print(file)
    lines = [line.rstrip() for line in open(file[:-2]+'.b')]
    time_stamp = lines[-1].split()[2]
    hycom_days = lines[-1].split()[3]
    tzero=datetime(1901,1,1,0,0)
    tRTOFS_DA.append(tzero+timedelta(float(hycom_days)-1))

tRTOFS_DA = np.asarray(tRTOFS_DA)

oktt_RTOFS_DA = np.where(mdates.date2num(tRTOFS_DA) >= mdates.date2num(argo_ttss[0]-timedelta(hours=2)))[0][0]

file = afiles[oktt_RTOFS_DA]
time_RTOFS_DA = tRTOFS_DA[oktt_RTOFS_DA]

nz = 41
layers = np.arange(0,nz)
temp_RTOFS_DA = np.empty((nz))
temp_RTOFS_DA[:] = np.nan
salt_RTOFS_DA = np.empty((nz))
salt_RTOFS_DA[:] = np.nan
zRTOFS_DA = np.empty((nz))
zRTOFS_DA[:] = np.nan

oklat_RTOFS_da = np.where(lat_RTOFS_DA[:,0] >= argo_latss[0])[0][0]
oklon_RTOFS_da = np.where(lon_RTOFS_DA[0,:]  >= argo_lonss[0]+ 360)[0][0]

ztmp = [0]
for lyr in tuple(layers):
    print(lyr)
    temp_RTOFSDA = readVar(file[:-2],'archive','temp',[lyr+1])
    temp_RTOFS_DA[lyr] = temp_RTOFSDA[oklat_RTOFS_da,oklon_RTOFS_da]
    
    salt_RTOFSDA = readVar(file[:-2],'archive','salin',[lyr+1])
    salt_RTOFS_DA[lyr] = salt_RTOFSDA[oklat_RTOFS_da,oklon_RTOFS_da]
    
    dp = readVar(file[:-2],'archive','thknss',[lyr+1])/9806
    ztmp = np.append(ztmp,dp[oklat_RTOFS_da,oklon_RTOFS_da])
    
zRTOFS_DA = np.cumsum(ztmp[0:-1]) + np.diff(np.cumsum(ztmp))/2    

###############################

# Figure temp
plt.figure(figsize=(5,6))
plt.plot(argo_tempss,-argo_depthss,'.-',linewidth=2,label='ARGO Float id'+argo_idss[0])
plt.plot(temp_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1',color='red')
plt.plot(temp_RTOFS,-depth_RTOFS,'.-',linewidth=2,label='RTOFS',color='g')
plt.plot(temp_RTOFS_DA,-zRTOFS_DA,'.-',linewidth=2,label='RTOFS-DA',color='sandybrown')
plt.ylim([-1000,0])
plt.xlim([5,28])
#plt.ylim([-200,0])
#plt.xlim([20,28])
plt.title('Temperature Profile on '+ str(argo_ttss[0])[0:13] +
          '\n [lon,lat] = [' \
          + str(np.round(argo_lonss[0],3)) +',' +\
              str(np.round(argo_latss[0],3))+']',\
              fontsize=16)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('$^oC$',fontsize=14)
plt.legend(loc='lower right',fontsize=14)

file = folder_fig + 'temp_ARGO_vs_GOFS_RTOFS_' + str(argo_ttss[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

# Figure salt
plt.figure(figsize=(5,6))
plt.plot(argo_saltss,-argo_depthss,'.-',linewidth=2,label='ARGO Float id'+argo_idss[0])
plt.plot(salt_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1',color='red')
plt.plot(salt_RTOFS,-depth_RTOFS,'.-',linewidth=2,label='RTOFS',color='g')
plt.plot(salt_RTOFS_DA,-zRTOFS_DA,'.-',linewidth=2,label='RTOFS-DA',color='sandybrown')
plt.ylim([-1000,0])
#plt.ylim([-200,0])
#plt.xlim([36,37])
plt.title('Salinity Profile on '+ str(argo_ttss[0])[0:13] +
          '\n [lon,lat] = [' \
          + str(np.round(argo_lonss[0],3)) +',' +\
              str(np.round(argo_latss[0],3))+']',\
              fontsize=16)
plt.ylabel('Depth (m)',fontsize=14)
plt.legend(loc='lower right',fontsize=12)

file = folder_fig + 'salt_ARGO_vs_GOFS_RTOFS' + str(argo_ttss[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 