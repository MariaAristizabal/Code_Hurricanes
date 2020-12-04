#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:49:06 2020

@author: aristizabal
"""

#%% User input

# GoMex
lon_lim = [-98,-80]
lat_lim = [15,32.5]

#cycle = '2020060400'
ti = '2020/06/04/06'

#GOFS3.1 output model location
url_GOFS_ts = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'
url_GOFS_uv = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'

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

#folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
#folder_fig = '/home/aristizabal/Figures/'
folder_fig = '/www/web/rucool/aristizabal/Figures/'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 
from datetime import datetime, timedelta
import cmocean
#from netCDF4 import Dataset
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

#%% GOFS 3.1
GOFS_ts = xr.open_dataset(url_GOFS_ts,decode_times=False)
GOFS_uv = xr.open_dataset(url_GOFS_uv,decode_times=False)

lt_GOFS = np.asarray(GOFS_ts['lat'][:])
ln_GOFS = np.asarray(GOFS_ts['lon'][:])
tt = GOFS_ts['time']
t_GOFS = netCDF4.num2date(tt[:],tt.units)

depth_GOFS = np.asarray(GOFS_ts['depth'][:])

# Conversion from glider longitude and latitude to GOFS convention
lon_limG = np.empty((len(lon_lim),))
lon_limG[:] = np.nan
for i in range(len(lon_lim)):
    if lon_lim[i] < 0: 
        lon_limG[i] = 360 + lon_lim[i]
    else:
        lon_limG[i] = lon_lim[i]
lat_limG = lat_lim

oklon_GOFS = np.where(np.logical_and(ln_GOFS >= lon_limG[0],ln_GOFS <= lon_limG[1]))[0]
oklat_GOFS = np.where(np.logical_and(lt_GOFS >= lat_limG[0],lt_GOFS <= lat_lim[1]))[0]

lon_GOFS = ln_GOFS[oklon_GOFS]
lat_GOFS = lt_GOFS[oklat_GOFS]

tini =  datetime.strptime(ti,'%Y/%m/%d/%H')   
ttGOFS = np.asarray([datetime(t_GOFS[i].year,t_GOFS[i].month,t_GOFS[i].day,t_GOFS[i].hour) for i in np.arange(len(t_GOFS))])
tstamp_GOFS = [mdates.date2num(ttGOFS[i]) for i in np.arange(len(ttGOFS))]
oktime_GOFS = np.unique(np.round(np.interp(mdates.date2num(tini),tstamp_GOFS,np.arange(len(tstamp_GOFS)))).astype(int))
time_GOFS = ttGOFS[oktime_GOFS][0]
#oktime_GOFS = np.where(t_GOFS == tini)[0][0]

# Conversion from GOFS convention to glider longitude and latitude
lon_GOFSg= np.empty((len(lon_GOFS),))
lon_GOFSg[:] = np.nan
for i in range(len(lon_GOFS)):
    if lon_GOFS[i] > 180:
        lon_GOFSg[i] = lon_GOFS[i] - 360
    else:
        lon_GOFSg[i] = lon_GOFS[i]
lat_GOFSg = lat_GOFS

temp_GOFS = np.asarray(GOFS_ts['water_temp'][oktime_GOFS,:,oklat_GOFS,oklon_GOFS])[0,:,:,:]
salt_GOFS = np.asarray(GOFS_ts['salinity'][oktime_GOFS,:,oklat_GOFS,oklon_GOFS])[0,:,:,:]
u_GOFS = np.asarray(GOFS_uv['water_u'][oktime_GOFS,:,oklat_GOFS,oklon_GOFS])[0,0,:,:]
v_GOFS = np.asarray(GOFS_uv['water_v'][oktime_GOFS,:,oklat_GOFS,oklon_GOFS])[0,0,:,:]

#%% Reading RTOFS-DA grid
print('Retrieving coordinates from RTOFS_DA')
# Reading lat and lon
lines_grid = [line.rstrip() for line in open(RTOFS_DA_grid+'.b')]
lon_RTOFS_da = np.array(readgrids(RTOFS_DA_grid,'plon:',[0]))
lat_RTOFS_da = np.array(readgrids(RTOFS_DA_grid,'plat:',[0]))

depth_RTOFS_da = np.asarray(readdepth(RTOFS_DA_depth,'depth'))

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

tini = datetime.strptime(ti,"%Y/%m/%d/%H")

#%% RTOFS
#Time window
year = int(tini.year)
month = int(tini.month)
day = int(tini.day)
tiniR = datetime(year, month, day)
tendR = tini + timedelta(days=1)

# Read RTOFS grid and time
print('Retrieving coordinates from RTOFS')

if tiniR.month < 10:
    if tiniR.day < 10:
        fol = 'rtofs.' + str(tiniR.year) + '0' + str(tiniR.month) + '0' + str(tiniR.day)
    else:
        fol = 'rtofs.' + str(tiniR.year) + '0' + str(tiniR.month) + str(tiniR.day)
else:
    if tiniR.day < 10:
        fol = 'rtofs.' + str(tiniR.year) + str(tiniR.month) + '0' + str(tiniR.day)
    else:
        fol = 'rtofs.' + str(tiniR.year) + str(tiniR.month) + str(tini.day)

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

oktt_RTOFS = np.where(mdates.date2num(tRTOFS) >= mdates.date2num(tini))[0][0]
oklat_RTOFS = np.where(np.logical_and(latRTOFS[:,0] >= lat_lim[0],latRTOFS[:,0] <= lat_lim[1]))[0]
oklon_RTOFS = np.where(np.logical_and(lonRTOFS[0,:] >= lon_lim[0],lonRTOFS[0,:] <= lon_lim[1]))[0]
    
nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[oktt_RTOFS]
ncRTOFS = xr.open_dataset(nc_file)
time_RTOFS = tRTOFS[oktt_RTOFS]
temp_RTOFS = np.asarray(ncRTOFS.variables['temperature'][0,:,oklat_RTOFS,oklon_RTOFS])
salt_RTOFS = np.asarray(ncRTOFS.variables['salinity'][0,:,oklat_RTOFS,oklon_RTOFS])
u_RTOFS = np.asarray(ncRTOFS.variables['u'][0,0,oklat_RTOFS,oklon_RTOFS])
v_RTOFS = np.asarray(ncRTOFS.variables['v'][0,0,oklat_RTOFS,oklon_RTOFS])
lon_RTOFS = lonRTOFS[oklat_RTOFS,:][:,oklon_RTOFS]
lat_RTOFS = latRTOFS[oklat_RTOFS,:][:,oklon_RTOFS]

#%% RTOFS-DA
# cycle
year = int(tini.year)
month = int(tini.month)
day = int(tini.day)
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

oktt_RTOFS_DA = np.where(mdates.date2num(tRTOFS_DA) >= mdates.date2num(tini-timedelta(hours=2)))[0][0]

file = afiles[oktt_RTOFS_DA]
time_RTOFS_DA = tRTOFS_DA[oktt_RTOFS_DA]

oklat_RTOFS_da = np.where(np.logical_and(lat_RTOFS_da[:,0] >= lat_lim[0],lat_RTOFS_da[:,0] <= lat_lim[1]))[0]
oklon_RTOFS_da = np.where(np.logical_and(lon_RTOFS_da[0,:] >= lon_lim[0]+360,lon_RTOFS_da[0,:] <= lon_lim[1]+360))[0]

lon_RTOFS_DA = lon_RTOFS_da[oklat_RTOFS_da,:][:,oklon_RTOFS_da]-360
lat_RTOFS_DA = lat_RTOFS_da[oklat_RTOFS_da,:][:,oklon_RTOFS_da]

nz = 41
layers = np.arange(0,nz)
temp_RTOFS_DA = np.empty((nz,len(oklat_RTOFS_da),len(oklon_RTOFS_da)))
temp_RTOFS_DA[:] = np.nan
salt_RTOFS_DA = np.empty((nz,len(oklat_RTOFS_da),len(oklon_RTOFS_da)))
salt_RTOFS_DA[:] = np.nan
zRTOFS_DA = np.empty((nz,len(oklat_RTOFS_da),len(oklon_RTOFS_da)))
zRTOFS_DA[:] = np.nan
ztmp = np.empty((nz,len(oklat_RTOFS_da),len(oklon_RTOFS_da)))
ztmp[:] = np.nan

for lyr in tuple(layers):
    print(lyr)
    temp_RTOFSDA = readVar(file[:-2],'archive','temp',[lyr+1])
    temp_RTOFS_DA[lyr,:,:] = temp_RTOFSDA[oklat_RTOFS_da,:][:,oklon_RTOFS_da]
    
    salt_RTOFSDA = readVar(file[:-2],'archive','salin',[lyr+1])
    salt_RTOFS_DA[lyr,:,:] = salt_RTOFSDA[oklat_RTOFS_da,:][:,oklon_RTOFS_da]
    
    dp = readVar(file[:-2],'archive','thknss',[lyr+1])/9806
    ztmp[lyr,:,:] = dp[oklat_RTOFS_da,:][:,oklon_RTOFS_da]
    
zRTOFS_DA = np.cumsum(ztmp[0:-1,:,:],axis=0) + np.diff(np.cumsum(ztmp,axis=0),axis=0)/2 

u_RTOFS_DA = readVar(file[:-2],'archive','u-vel.',[1])[oklat_RTOFS_da,:][:,oklon_RTOFS_da]
v_RTOFS_DA = readVar(file[:-2],'archive','v-vel.',[1])[oklat_RTOFS_da,:][:,oklon_RTOFS_da]

temp_RTOFS_DA[temp_RTOFS_DA > 100] = np.nan
salt_RTOFS_DA[salt_RTOFS_DA > 100] = np.nan
#zRTOFS_DA[zRTOFS_DA > 10^10] = 0

#%% Figure sst GOFS

tmin = np.int(np.floor(np.min([np.nanmin(temp_GOFS[0,:,:]),np.nanmin(temp_RTOFS[0,:,:]),np.nanmin(temp_RTOFS_DA[0,:,:])])))
tmax = np.int(np.ceil(np.max([np.nanmax(temp_GOFS[0,:,:]),np.nanmax(temp_RTOFS[0,:,:]),np.nanmax(temp_RTOFS_DA[0,:,:])])))
temp_lim = [tmin,tmax]

kw = dict(levels = np.arange(temp_lim[0],temp_lim[1],0.5))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contour(lon_GOFSg,lat_GOFSg,temp_GOFS[0,:,:],levels=[26],colors='white')
plt.contourf(lon_GOFSg,lat_GOFSg,temp_GOFS[0,:,:],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('GOFS SST \n on '+str(time_GOFS)[0:13],fontsize=16)

q=plt.quiver(lon_GOFSg[::6],lat_GOFSg[::6],u_GOFS[::6,::6],v_GOFS[::6,::6],\
             scale=3,scale_units='inches',alpha=0.7)
plt.quiverkey(q,np.max(lon_GOFSg)-0.2,np.max(lat_GOFSg)+0.5,1,"1 m/s",\
              coordinates='data',color='k',fontproperties={'size': 14})

file = folder_fig +'GOFS_SST_' + str(time_GOFS)[0:14]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure sst RTOFS

kw = dict(levels = np.arange(temp_lim[0],temp_lim[1],0.5))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contour(lon_RTOFS,lat_RTOFS,temp_RTOFS[0,:,:],levels=[26],colors='white')
plt.contourf(lon_RTOFS,lat_RTOFS,temp_RTOFS[0,:,:],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('RTOFS SST \n on '+str(time_GOFS)[0:13],fontsize=16)

q=plt.quiver(lon_RTOFS[::6,::6],lat_RTOFS[::6,::6],u_RTOFS[::6,::6],v_RTOFS[::6,::6],\
             scale=3,scale_units='inches',alpha=0.7)
plt.quiverkey(q,np.max(lon_RTOFS)-0.2,np.max(lat_RTOFS)+0.5,1,"1 m/s",\
              coordinates='data',color='k',fontproperties={'size': 14})

file = folder_fig +'RTOFS_SST_'+str(time_RTOFS)[0:14]  
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure sst RTOFS-DA

kw = dict(levels = np.arange(temp_lim[0],temp_lim[1],0.5))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contour(lon_RTOFS_DA,lat_RTOFS_DA,temp_RTOFS_DA[0,:,:],levels=[26],colors='white')
plt.contourf(lon_RTOFS_DA,lat_RTOFS_DA,temp_RTOFS_DA[0,:,:],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('RTOFS-DA SST \n on '+str(time_GOFS)[0:13],fontsize=16)

q=plt.quiver(lon_RTOFS_DA[::6,::6],lat_RTOFS_DA[::6,::6],u_RTOFS_DA[::6,::6],v_RTOFS_DA[::6,::6],\
             scale=3,scale_units='inches',alpha=0.7)
plt.quiverkey(q,np.max(lon_RTOFS_DA)-0.2,np.max(lat_RTOFS_DA)+0.5,1,"1 m/s",\
              coordinates='data',color='k',fontproperties={'size': 14})

file = folder_fig +'RTOFS_DA_SST_'+str(time_RTOFS_DA)[0:14]  
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -90 longitude GOFS

tempt_lim = [14,31]
kw = dict(levels = np.arange(tempt_lim[0],tempt_lim[1],1))

oklon90 = np.where(lon_GOFSg >= -90)[0][0]
okd200 = np.where(depth_GOFS <= 200)[0]

plt.figure()
plt.contourf(lat_GOFS,-depth_GOFS[okd200],temp_GOFS[okd200,:,oklon90],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lat_GOFS,-depth_GOFS[okd200],temp_GOFS[okd200,:,oklon90],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('GOFS Temperature Cross Section \n on '+str(time_GOFS)[0:13],fontsize=16)
plt.xlim(20,30)

file = folder_fig + 'GOFS_temp_cross_sect_' + str(time_GOFS)[0:14]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -90 longitude RTOFS

oklon90 = np.where(lon_RTOFS[0,:] >= -90)[0][0]
okd200 = np.where(depth_RTOFS <= 200)[0]

plt.figure()
plt.contourf(lat_RTOFS[:,oklon90],-depth_RTOFS[okd200],temp_RTOFS[okd200,:,oklon90],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lat_RTOFS[:,oklon90],-depth_RTOFS[okd200],temp_RTOFS[okd200,:,oklon90],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('RTOFS Temperature Cross Section \n on '+str(time_RTOFS)[0:13],fontsize=16)
plt.xlim(20,30)

file = folder_fig + 'RTOFS_temp_cross_sect_' + str(time_RTOFS)[0:14]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -90 longitude RTOFS-DA

oklon90 = np.where(lon_RTOFS_DA[0,:] >= -90)[0][0]
okd200 = np.where(zRTOFS_DA[:,:,oklon90] <= 200)[0]
matrix_lat = np.tile(lat_RTOFS_DA[:,oklon90],(zRTOFS_DA.shape[0],1))

plt.figure()
plt.contourf(matrix_lat,-zRTOFS_DA[:,:,oklon90],temp_RTOFS_DA[0:-1,:,oklon90],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(matrix_lat,-zRTOFS_DA[:,:,oklon90],temp_RTOFS_DA[0:-1,:,oklon90],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('RTOFS-DA Temperature Cross Section \n on '+str(time_RTOFS_DA)[0:13],fontsize=16)
plt.xlim(20,30)
plt.ylim(-200,0)

file = folder_fig + 'RTOFS_DA_temp_cross_sect_' + str(time_RTOFS_DA)[0:14]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)




