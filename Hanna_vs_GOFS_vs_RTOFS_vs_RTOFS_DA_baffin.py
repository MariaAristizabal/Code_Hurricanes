#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:02:53 2020

@author: aristizabal
"""

#%% User input

# GoMex
lon_lim = [-98,-80]
lat_lim = [15,32.5]

#cycle = '2020060400'
ti = '2020/07/22/06'

#GOFS3.1 output model location
url_GOFS_ts = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'
url_GOFS_uv = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'

# Argo floats
url_Argo = 'http://www.ifremer.fr/erddap'

# RTOFS files
folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

folder_RTOFS_DA = '/home/aristizabal/RTOFS-DA/rtofs.20200723/'
prefix_RTOFS_DA = 'rtofs_glo_3dz_n'

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
from erddapy import ERDDAP
import pandas as pd

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Look for Argo datasets 

#Time window
tini = datetime(2020,7,23)
tend = tini + timedelta(days=1)

e = ERDDAP(server = url_Argo)

# Grab every dataset available
#datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': str(tini),
    'max_time': str(tend),
}

search_url = e.get_search_url(response='csv', **kw)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
dataset = search['Dataset ID'].values

msg = 'Found {} Datasets:\n\n{}'.format
print(msg(len(dataset), '\n'.join(dataset)))

dataset_type = dataset[0]

constraints = {
    'time>=': str(tini),
    'time<=': str(tend),
    'latitude>=': lat_lim[0],
    'latitude<=': lat_lim[1],
    'longitude>=':lon_lim[0],
    'longitude<=': lon_lim[1],
}

variables = [
 'platform_number',
 'time',
 'pres',
 'longitude',
 'latitude',
 'temp',
 'psal',
]

e = ERDDAP(
    server = url_Argo,
    protocol = 'tabledap',
    response = 'nc'
)

e.dataset_id = dataset_type
e.constraints=constraints
e.variables=variables

print(e.get_download_url())

df = e.to_pandas(
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()

argo_ids = np.asarray(df['platform_number'])
argo_times = np.asarray(df['time (UTC)'])
argo_press = np.asarray(df['pres (decibar)'])
argo_lons = np.asarray(df['longitude (degrees_east)'])
argo_lats = np.asarray(df['latitude (degrees_north)'])
argo_temps = np.asarray(df['temp (degree_Celsius)'])
argo_salts = np.asarray(df['psal (PSU)'])

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

ncfiles_RTOFS_DA = sorted(glob.glob(os.path.join(folder_RTOFS_DA,prefix_RTOFS_DA+'*.nc')))

file = ncfiles_RTOFS_DA[0]
ncRTOFS_DA = xr.open_dataset(file)
lonRTOFS_DA = np.asarray(ncRTOFS_DA.Longitude[:])
latRTOFS_DA = np.asarray(ncRTOFS_DA.Latitude[:])

for file in ncfiles_RTOFS_DA:
    print(file)
    ncRTOFS_DA = xr.open_dataset(file)    
    timeRTOFS_DA = np.asarray(ncRTOFS_DA.MT[:])
    
tRTOFS_DA = np.asarray([mdates.num2date(mdates.date2num(timeRTOFS_DA[t])) \
          for t in np.arange(len(ncfiles_RTOFS_DA))])

oktt_RTOFS_DA = np.where(mdates.date2num(tRTOFS_DA) >= mdates.date2num(tini))[0][0]
oklat_RTOFS_DA = np.where(np.logical_and(latRTOFS_DA[:,0] >= lat_lim[0],latRTOFS_DA[:,0] <= lat_lim[1]))[0]
oklon_RTOFS_DA = np.where(np.logical_and(lonRTOFS_DA[0,:] >= lon_lim[0],lonRTOFS_DA[0,:] <= lon_lim[1]))[0]
    
nc_file = ncfiles_RTOFS_DA[oktt_RTOFS_DA]
ncRTOFS = xr.open_dataset(nc_file)
time_RTOFS_DA = tRTOFS_DA[oktt_RTOFS_DA]
depth_RTOFS_DA = np.asarray(ncRTOFS_DA.Depth[:])
temp_RTOFS_DA = np.asarray(ncRTOFS_DA.variables['temperature'][0,:,oklat_RTOFS_DA,oklon_RTOFS_DA])
salt_RTOFS_DA = np.asarray(ncRTOFS.variables['salinity'][0,:,oklat_RTOFS_DA,oklon_RTOFS_DA])
u_RTOFS_DA = np.asarray(ncRTOFS_DA.variables['u'][0,0,oklat_RTOFS_DA,oklon_RTOFS_DA])
v_RTOFS_DA = np.asarray(ncRTOFS_DA.variables['v'][0,0,oklat_RTOFS_DA,oklon_RTOFS_DA])
lon_RTOFS_DA = lonRTOFS_DA[oklat_RTOFS_DA,:][:,oklon_RTOFS_DA]
lat_RTOFS_DA = latRTOFS_DA[oklat_RTOFS_DA,:][:,oklon_RTOFS_DA]

#%% Figure sst GOFS

tmin = np.int(np.floor(np.min([np.nanmin(temp_GOFS[0,:,:]),np.nanmin(temp_RTOFS[0,:,:]),np.nanmin(temp_RTOFS_DA[0,:,:])])))
#tmax = np.int(np.ceil(np.max([np.nanmax(temp_GOFS[0,:,:]),np.nanmax(temp_RTOFS[0,:,:]),np.nanmax(temp_RTOFS_DA[0,:,:])])))
tmax = 32
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

file = folder_fig +'GOFS_SST_' + str(time_GOFS)[0:13]    
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

file = folder_fig +'RTOFS_SST_'+str(time_RTOFS)[0:13]  
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

file = folder_fig +'RTOFS_DA_SST_'+str(time_RTOFS_DA)[0:13]  
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -92 longitude GOFS

tempt_lim = [14,32]
kw = dict(levels = np.arange(tempt_lim[0],tempt_lim[1],1))

oklon92 = np.where(lon_GOFSg >= -92)[0][0]
okd200 = np.where(depth_GOFS <= 200)[0]

plt.figure()
plt.contourf(lat_GOFS,-depth_GOFS[okd200],temp_GOFS[okd200,:,oklon92],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lat_GOFS,-depth_GOFS[okd200],temp_GOFS[okd200,:,oklon92],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('GOFS Temperature Cross Section \n on '+str(time_GOFS)[0:13],fontsize=16)
plt.xlim(20,30)

file = folder_fig + 'GOFS_temp_cross_sect_' + str(time_GOFS)[0:13]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -92 longitude RTOFS

oklon92= np.where(lon_RTOFS[0,:] >= -92)[0][0]
okd200 = np.where(depth_RTOFS <= 200)[0]

plt.figure()
plt.contourf(lat_RTOFS[:,oklon92],-depth_RTOFS[okd200],temp_RTOFS[okd200,:,oklon92],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lat_RTOFS[:,oklon92],-depth_RTOFS[okd200],temp_RTOFS[okd200,:,oklon92],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('RTOFS Temperature Cross Section \n on '+str(time_RTOFS)[0:13],fontsize=16)
plt.xlim(20,30)

file = folder_fig + 'RTOFS_temp_cross_sect_' + str(time_RTOFS)[0:13]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -92 longitude RTOFS-DA

oklon92 = np.where(lon_RTOFS_DA[0,:] >= -92)[0][0]
okd200 = np.where(depth_RTOFS_DA <= 200)[0]

plt.figure()
plt.contourf(lat_RTOFS_DA[:,oklon92],-depth_RTOFS_DA[okd200],temp_RTOFS_DA[okd200,:,oklon92],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lat_RTOFS_DA[:,oklon92],-depth_RTOFS_DA[okd200],temp_RTOFS_DA[okd200,:,oklon92],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('RTOFS-DA Temperature Cross Section \n on '+str(time_RTOFS_DA)[0:13],fontsize=16)
plt.xlim(20,30)
plt.ylim(-200,0)

file = folder_fig + 'RTOFS_DA_temp_cross_sect_' + str(time_RTOFS_DA)[0:13]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -85 longitude GOFS

tempt_lim = [14,32]
kw = dict(levels = np.arange(tempt_lim[0],tempt_lim[1],1))

oklon85 = np.where(lon_GOFSg >= -85)[0][0]
okd200 = np.where(depth_GOFS <= 200)[0]

plt.figure()
plt.contourf(lat_GOFS,-depth_GOFS[okd200],temp_GOFS[okd200,:,oklon85],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lat_GOFS,-depth_GOFS[okd200],temp_GOFS[okd200,:,oklon85],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('GOFS Temperature Cross Section \n on '+str(time_GOFS)[0:13],fontsize=16)
plt.xlim(16,30)

file = folder_fig + 'GOFS_temp_cross_sect85_' + str(time_GOFS)[0:13]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -85 longitude RTOFS

oklon85= np.where(lon_RTOFS[0,:] >= -85)[0][0]
okd200 = np.where(depth_RTOFS <= 200)[0]

plt.figure()
plt.contourf(lat_RTOFS[:,oklon92],-depth_RTOFS[okd200],temp_RTOFS[okd200,:,oklon85],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lat_RTOFS[:,oklon92],-depth_RTOFS[okd200],temp_RTOFS[okd200,:,oklon85],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('RTOFS Temperature Cross Section \n on '+str(time_RTOFS)[0:13],fontsize=16)
plt.xlim(16,30)

file = folder_fig + 'RTOFS_temp_cross_sect85_' + str(time_RTOFS)[0:13]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -85 longitude RTOFS-DA

oklon85 = np.where(lon_RTOFS_DA[0,:] >= -85)[0][0]
okd200 = np.where(depth_RTOFS_DA <= 200)[0]

plt.figure()
plt.contourf(lat_RTOFS_DA[:,oklon85],-depth_RTOFS_DA[okd200],temp_RTOFS_DA[okd200,:,oklon85],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lat_RTOFS_DA[:,oklon85],-depth_RTOFS_DA[okd200],temp_RTOFS_DA[okd200,:,oklon85],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('RTOFS-DA Temperature Cross Section \n on '+str(time_RTOFS_DA)[0:13],fontsize=16)
plt.xlim(16,30)
#plt.ylim(-200,0)

file = folder_fig + 'RTOFS_DA_temp_cross_sect85_' + str(time_RTOFS_DA)[0:13]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Cross section along -85 longitude GOFS

tempt_lim = [14,32]
kw = dict(levels = np.arange(tempt_lim[0],tempt_lim[1],1))

oklon85 = np.where(lon_GOFSg >= -85)[0][0]
okd200 = np.where(depth_GOFS <= 200)[0]

plt.figure()
plt.contourf(lat_GOFS,-depth_GOFS[okd200],temp_GOFS[okd200,:,oklon85],cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lat_GOFS,-depth_GOFS[okd200],temp_GOFS[okd200,:,oklon85],[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('GOFS Temperature Cross Section \n on '+str(time_GOFS)[0:13],fontsize=16)
plt.xlim(16,30)

file = folder_fig + 'GOFS_temp_cross_sect85_' + str(time_GOFS)[0:13]    
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure argo float vs GOFS and vs RTOFS

folder_RTOFS_DA = '/home/aristizabal/RTOFS-DA/rtofs.20200724/'
prefix_RTOFS_DA = 'rtofs_glo_3dz_n'

argo_idd = np.unique(argo_ids)
argo_idd_eddy = [idd for idd in argo_idd if idd == 4901476]
 
for i,id in enumerate(argo_idd_eddy):
    print(id)
    okind = np.where(argo_ids == id)[0]
    argo_time = np.asarray([datetime.strptime(t,'%Y-%m-%dT%H:%M:%SZ') for t in argo_times[okind]])

    argo_lon = argo_lons[okind]
    argo_lat = argo_lats[okind]
    argo_pres = argo_press[okind]
    argo_temp = argo_temps[okind]
    argo_salt = argo_salts[okind]

    # GOFS
    print('Retrieving variables from GOFS')
    if isinstance(GOFS_ts,float):
        temp_GOFS = np.nan
        salt_GOFS = np.nan
    else:
        #oktt_GOFS = np.where(t_GOFS >= argo_time[0])[0][0]
        ttGOFS = np.asarray([datetime(t_GOFS[i].year,t_GOFS[i].month,t_GOFS[i].day,t_GOFS[i].hour) for i in np.arange(len(t_GOFS))])
        tstamp_GOFS = [mdates.date2num(ttGOFS[i]) for i in np.arange(len(ttGOFS))]
        oktt_GOFS = np.unique(np.round(np.interp(mdates.date2num(argo_time[0]),tstamp_GOFS,np.arange(len(tstamp_GOFS)))).astype(int))[0]
        oklat_GOFS = np.where(lt_GOFS >= argo_lat[0])[0][0]
        oklon_GOFS = np.where(ln_GOFS >= argo_lon[0]+360)[0][0]
        temp_GOFS = np.asarray(GOFS_ts['water_temp'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])
        salt_GOFS = np.asarray(GOFS_ts['salinity'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])
        
    # RTOFS 
    #Time window
    year = int(argo_time[0].year)
    month = int(argo_time[0].month)
    day = int(argo_time[0].day)
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

    oktt_RTOFS = np.where(mdates.date2num(tRTOFS) >= mdates.date2num(argo_time[0]))[0][0]
    oklat_RTOFS = np.where(latRTOFS[:,0] >= argo_lat[0])[0][0]
    oklon_RTOFS = np.where(lonRTOFS[0,:]  >= argo_lon[0])[0][0]

    nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[oktt_RTOFS]
    ncRTOFS = xr.open_dataset(nc_file)
    #time_RTOFS = tRTOFS[oktt_RTOFS]
    temp_RTOFS = np.asarray(ncRTOFS.variables['temperature'][0,:,oklat_RTOFS,oklon_RTOFS])
    salt_RTOFS = np.asarray(ncRTOFS.variables['salinity'][0,:,oklat_RTOFS,oklon_RTOFS])
    
    #% RTOFS-DA  
    print('Retrieving coordinates from RTOFS-DA')
    ncfiles_RTOFS_DA = sorted(glob.glob(os.path.join(folder_RTOFS_DA,prefix_RTOFS_DA+'*.nc')))
    
    file = ncfiles_RTOFS_DA[0]
    ncRTOFS_DA = xr.open_dataset(file)
    lonRTOFS_DA = np.asarray(ncRTOFS_DA.Longitude[:])
    latRTOFS_DA = np.asarray(ncRTOFS_DA.Latitude[:])
    
    timeRTOFS_DA = []
    for file in ncfiles_RTOFS_DA:
        print(file)
        ncRTOFS_DA = xr.open_dataset(file)    
        timeRTOFS_DA.append(ncRTOFS_DA.MT[:].values)
    timeRTOFS_DA = np.asarray(timeRTOFS_DA)
        
    tRTOFS_DA = np.asarray([mdates.num2date(mdates.date2num(timeRTOFS_DA[t])) \
              for t in np.arange(len(ncfiles_RTOFS_DA))])
        
    oktt_RTOFS_DA = np.where(mdates.date2num(tRTOFS) >= mdates.date2num(argo_time[0]))[0][0]
    oklat_RTOFS_DA = np.where(latRTOFS[:,0] >= argo_lat[0])[0][0]
    oklon_RTOFS_DA = np.where(lonRTOFS[0,:]  >= argo_lon[0])[0][0]
        
    nc_file = ncfiles_RTOFS_DA[oktt_RTOFS_DA]
    ncRTOFS = xr.open_dataset(nc_file)
    time_RTOFS_DA = tRTOFS_DA[oktt_RTOFS_DA]
    depth_RTOFS_DA = np.asarray(ncRTOFS_DA.Depth[:])
    temp_RTOFS_DA = np.asarray(ncRTOFS_DA.variables['temperature'][0,:,oklat_RTOFS_DA,oklon_RTOFS_DA])
    salt_RTOFS_DA = np.asarray(ncRTOFS.variables['salinity'][0,:,oklat_RTOFS_DA,oklon_RTOFS_DA])
    lon_RTOFS_DA = lonRTOFS_DA[oklat_RTOFS_DA,oklon_RTOFS_DA]
    lat_RTOFS_DA = latRTOFS_DA[oklat_RTOFS_DA,oklon_RTOFS_DA]
    
    # Figure temp
    plt.figure(figsize=(5,6))
    plt.plot(argo_temp,-argo_pres,'.-',linewidth=2,label='ARGO Float id '+str(id))
    plt.plot(temp_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1',color='red')
    plt.plot(temp_RTOFS,-depth_RTOFS,'.-',linewidth=2,label='RTOFS',color='g')
    plt.plot(temp_RTOFS_DA,-depth_RTOFS_DA,'.-',linewidth=2,label='RTOFS-DA',color='orange')
    plt.plot(np.tile(26,len(np.arange(100))),-np.arange(100),'--k')
    plt.ylim([-1000,0])
    plt.title('Temperature Profile on '+ str(argo_time[0])[0:13] +
              '\n [lon,lat] = [' \
              + str(np.round(argo_lon[0],3)) +',' +\
                  str(np.round(argo_lat[0],3))+']',\
                  fontsize=16)
    plt.ylabel('Depth (m)',fontsize=14)
    plt.xlabel('$^oC$',fontsize=14)
    plt.legend(loc='lower right',fontsize=14)
    plt.ylim([-100,0])
    plt.xlim([24,31])

    file = folder_fig + 'ARGO_vs_GOFS_RTOFS_RTOFS_DA_temp_' + str(id) + '_' + str(argo_time[0])[0:13]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)