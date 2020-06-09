#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:29:14 2020

@author: aristizabal
"""

#%% User input
# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

# urls
url_glider = 'https://data.ioos.us/gliders/erddap'
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
#url_RTOFS = 'https://nomads.ncep.noaa.gov:9090/dods/rtofs/rtofs_global'

# FTP server RTOFS
#ftp_RTOFS = 'ftp.ncep.noaa.gov'

#folder_RTOFS = '/Volumes/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'
folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'
#folder_RTOFS = '/Users/aristizabal/Desktop/rtofs/'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

# COPERNICUS MARINE ENVIRONMENT MONITORING SERVICE (CMEMS)
url_cmems = 'http://nrt.cmems-du.eu/motu-web/Motu'
service_id = 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS'
product_id = 'global-analysis-forecast-phy-001-024'
depth_min = '0.493'
#out_dir = '/Users/aristizabal/Desktop'
out_dir = '/home/aristizabal/Figures/'
ncCOP_global = '/home/aristizabal/global-analysis-forecast-phy-001-024_1565877333169.nc'

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

folder_fig = '/home/aristizabal/Figures/'

#%%
from erddapy import ERDDAP
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import matplotlib.patches as patches
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
#from ftplib import FTP
import netCDF4
import os
import os.path

# Do not produce figures on screen
#plt.switch_backend('agg')

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%%

#tend = datetime(2019, 7, 27, 0, 0)
#tini = datetime(2019, 7, 28, 0, 0)

tini = datetime(2019, 7, 27, 0, 0)
tend = datetime(2019, 7, 28, 0, 0)

#%% Look for datasets in IOOS glider dac
print('Looking for glider data sets')
e = ERDDAP(server = url_glider)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': tini.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'max_time': tend.strftime('%Y-%m-%dT%H:%M:%SZ'),
}

search_url = e.get_search_url(response='csv', **kw)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

# Setting constraints
constraints = {
        'time>=': tini,
        'time<=': tend,
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }

variables = [
        'depth',
        'latitude',
        'longitude',
        'time',
        'temperature',
        'salinity'
        ]

e = ERDDAP(
        server=url_glider,
        protocol='tabledap',
        response='nc'
        )

#%% Read GOFS 3.1 output
print('Retrieving coordinates from GOFS 3.1')
GOFS31 = xr.open_dataset(url_GOFS,decode_times=False)

latGOFS = GOFS31.lat[:]
lonGOFS = GOFS31.lon[:]
depthGOFS = GOFS31.depth[:]
ttGOFS = GOFS31.time
tGOFS = netCDF4.num2date(ttGOFS[:],ttGOFS.units)

#%% Read RTOFS grid and time
print('Retrieving coordinates from RTOFS')

if tend.month < 10:
    if tend.day < 10:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day)
else:
    if tend.day < 10:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + str(tini.day)

ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[0])
latRTOFS = np.asarray(ncRTOFS.Latitude[:])
lonRTOFS = np.asarray(ncRTOFS.Longitude[:])
depthRTOFS = np.asarray(ncRTOFS.Depth[:])

tRTOFS = []
for t in np.arange(len(nc_files_RTOFS)):
    ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
          for t in np.arange(len(nc_files_RTOFS))])

#%% Downloading and reading Copernicus grid
COP_grid = xr.open_dataset(ncCOP_global)

latCOP_glob = COP_grid.latitude[:]
lonCOP_glob = COP_grid.longitude[:]

#%% Reading bathymetry data
ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

#%%    
#id=gliders[6]
id=gliders[7]
print('Reading ' + id )
e.dataset_id = id
e.constraints = constraints
e.variables = variables

# Converting glider data to data frame
df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
        ).dropna()

# Coverting glider vectors into arrays
timeg, ind = np.unique(df.index.values,return_index=True)
latg = df['latitude (degrees_north)'].values[ind]
long = df['longitude (degrees_east)'].values[ind]

dg = df['depth (m)'].values
#vg = df['temperature (degree_Celsius)'].values
tg = df[df.columns[3]].values
sg = df[df.columns[4]].values

delta_z = 0.3
zn = np.int(np.round(np.max(dg)/delta_z))

depthg = np.empty((zn,len(timeg)))
depthg[:] = np.nan
tempg = np.empty((zn,len(timeg)))
tempg[:] = np.nan
saltg = np.empty((zn,len(timeg)))
saltg[:] = np.nan

# Grid variables
depthg_gridded = np.arange(0,np.nanmax(dg),delta_z)
tempg_gridded = np.empty((len(depthg_gridded),len(timeg)))
tempg_gridded[:] = np.nan
saltg_gridded = np.empty((len(depthg_gridded),len(timeg)))
saltg_gridded[:] = np.nan

for i,ii in enumerate(ind):
    if i < len(timeg)-1:
        depthg[0:len(dg[ind[i]:ind[i+1]]),i] = dg[ind[i]:ind[i+1]]
        tempg[0:len(tg[ind[i]:ind[i+1]]),i] = tg[ind[i]:ind[i+1]]
        saltg[0:len(sg[ind[i]:ind[i+1]]),i] = sg[ind[i]:ind[i+1]]
    else:
        depthg[0:len(dg[ind[i]:len(dg)]),i] = dg[ind[i]:len(dg)]
        tempg[0:len(tg[ind[i]:len(tg)]),i] = tg[ind[i]:len(tg)]
        saltg[0:len(sg[ind[i]:len(sg)]),i] = sg[ind[i]:len(sg)]

for t,tt in enumerate(timeg):
    depthu,oku = np.unique(depthg[:,t],return_index=True)
    tempu = tempg[oku,t]
    saltu = saltg[oku,t]
    okdd = np.isfinite(depthu)
    depthf = depthu[okdd]
    tempf = tempu[okdd]
    saltf = saltu[okdd]

    okt = np.isfinite(tempf)
    if np.sum(okt) < 3:
        tempg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[okt]),\
                             depthg_gridded < np.max(depthf[okt]))
        tempg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[okt],tempf[okt])

    oks = np.isfinite(saltf)
    if np.sum(oks) < 3:
        saltg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[oks]),\
                             depthg_gridded < np.max(depthf[oks]))
        saltg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[oks],saltf[oks])

# Conversion from glider longitude and latitude to GOFS convention
target_lonGOFS = np.empty((len(long),))
target_lonGOFS[:] = np.nan
for i,ii in enumerate(long):
    if ii < 0:
        target_lonGOFS[i] = 360 + ii
    else:
        target_lonGOFS[i] = ii
target_latGOFS = latg

# Conversion from glider longitude and latitude to RTOFS convention
target_lonRTOFS = long
target_latRTOFS = latg

# Narrowing time window of GOFS 3.1 to coincide with glider time window
tmin = mdates.num2date(mdates.date2num(timeg[0]))
tmax = mdates.num2date(mdates.date2num(timeg[-1]))
oktimeGOFS = np.where(np.logical_and(mdates.date2num(tGOFS) >= mdates.date2num(tmin),\
                                     mdates.date2num(tGOFS) <= mdates.date2num(tmax)))
timeGOFS = tGOFS[oktimeGOFS]

# Narrowing time window of RTOFS to coincide with glider time window
#tmin = tini
#tmax = tend
tmin = mdates.num2date(mdates.date2num(timeg[0]))
tmax = mdates.num2date(mdates.date2num(timeg[-1]))
oktimeRTOFS = np.where(np.logical_and(tRTOFS >= tmin, tRTOFS <= tmax))
timeRTOFS = mdates.num2date(mdates.date2num(tRTOFS[oktimeRTOFS]))

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
tstamp_GOFS = [mdates.date2num(timeGOFS[i]) for i in np.arange(len(timeGOFS))]
tstamp_RTOFS = [mdates.date2num(timeRTOFS[i]) for i in np.arange(len(timeRTOFS))]

# interpolating glider lon and lat to lat and lon on GOFS 3.1 time
sublonGOFS=np.interp(tstamp_GOFS,tstamp_glider,target_lonGOFS)
sublatGOFS=np.interp(tstamp_GOFS,tstamp_glider,target_latGOFS)

# getting the model grid positions for sublonm and sublatm
oklonGOFS=np.round(np.interp(sublonGOFS,lonGOFS,np.arange(len(lonGOFS)))).astype(int)
oklatGOFS=np.round(np.interp(sublatGOFS,latGOFS,np.arange(len(latGOFS)))).astype(int)

# Getting glider transect from GOFS 3.1
print('Getting glider transect from GOFS 3.1. If it breaks is because GOFS 3.1 server is not responding')
target_tempGOFS = np.empty((len(depthGOFS),len(oktimeGOFS[0])))
target_tempGOFS[:] = np.nan
for i in range(len(oktimeGOFS[0])):
    print(len(oktimeGOFS[0]),' ',i)
    target_tempGOFS[:,i] = GOFS31.variables['water_temp'][oktimeGOFS[0][i],:,oklatGOFS[i],oklonGOFS[i]]
target_tempGOFS[target_tempGOFS < -100] = np.nan

target_saltGOFS = np.empty((len(depthGOFS),len(oktimeGOFS[0])))
target_saltGOFS[:] = np.nan
for i in range(len(oktimeGOFS[0])):
    print(len(oktimeGOFS[0]),' ',i)
    target_saltGOFS[:,i] = GOFS31.variables['salinity'][oktimeGOFS[0][i],:,oklatGOFS[i],oklonGOFS[i]]
target_saltGOFS[target_saltGOFS < -100] = np.nan

# interpolating glider lon and lat to lat and lon on RTOFS time
sublonRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,target_lonRTOFS)
sublatRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,target_latRTOFS)

# getting the model grid positions for sublonm and sublatm
oklonRTOFS = np.round(np.interp(sublonRTOFS,lonRTOFS[0,:],np.arange(len(lonRTOFS[0,:])))).astype(int)
oklatRTOFS = np.round(np.interp(sublatRTOFS,latRTOFS[:,0],np.arange(len(latRTOFS[:,0])))).astype(int)

# Getting glider transect from RTOFS
print('Getting glider transect from RTOFS')
target_tempRTOFS = np.empty((len(depthRTOFS),len(oktimeRTOFS[0])))
target_tempRTOFS[:] = np.nan
for i in range(len(oktimeRTOFS[0])):
    print(len(oktimeRTOFS[0]),' ',i)
    nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[i]
    ncRTOFS = xr.open_dataset(nc_file)
    target_tempRTOFS[:,i] = ncRTOFS.variables['temperature'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
target_tempRTOFS[target_tempRTOFS < -100] = np.nan

target_saltRTOFS = np.empty((len(depthRTOFS),len(oktimeRTOFS[0])))
target_saltRTOFS[:] = np.nan
for i in range(len(oktimeRTOFS[0])):
    print(len(oktimeRTOFS[0]),' ',i)
    nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[i]
    ncRTOFS = xr.open_dataset(nc_file)
    target_saltRTOFS[:,i] = ncRTOFS.variables['salinity'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
target_saltRTOFS[target_saltRTOFS < -100] = np.nan

#os.system('rm ' + out_dir + '/' + '*.nc')

# Downloading and reading Copernicus output
motuc = 'python -m motuclient --motu ' + url_cmems + \
    ' --service-id ' + service_id + \
    ' --product-id ' + product_id + \
    ' --longitude-min ' + str(np.min(long)-2/12) + \
    ' --longitude-max ' + str(np.max(long)+2/12) + \
    ' --latitude-min ' + str(np.min(latg)-2/12) + \
    ' --latitude-max ' + str(np.max(latg)+2/12) + \
    ' --date-min ' + str(tini-timedelta(0.5)) + \
    ' --date-max ' + str(tend+timedelta(0.5)) + \
    ' --depth-min ' + depth_min + \
    ' --depth-max ' + str(np.nanmax(depthg)) + \
    ' --variable ' + 'thetao' + ' ' + \
    ' --variable ' + 'so'  + ' ' + \
    ' --out-dir ' + out_dir + \
    ' --out-name ' + id + '.nc' + ' ' + \
    ' --user ' + 'maristizabalvar' + ' ' + \
    ' --pwd ' +  'MariaCMEMS2018'

os.system(motuc)

COP_file = out_dir + '/' + id + '.nc'
COP = xr.open_dataset(COP_file)

latCOP = COP.latitude[:]
lonCOP = COP.longitude[:]
depthCOP = COP.depth[:]
tCOP = np.asarray(mdates.num2date(mdates.date2num(COP.time[:])))

tmin = tini
tmax = tend

oktimeCOP = np.where(np.logical_and(mdates.date2num(tCOP) >= mdates.date2num(tmin),\
                                    mdates.date2num(tCOP) <= mdates.date2num(tmax)))
timeCOP = tCOP[oktimeCOP]

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
tstamp_COP = [mdates.date2num(timeCOP[i]) for i in np.arange(len(timeCOP))]

# interpolating glider lon and lat to lat and lon on Copernicus time
sublonCOP = np.interp(tstamp_COP,tstamp_glider,long)
sublatCOP = np.interp(tstamp_COP,tstamp_glider,latg)

# getting the model grid positions for sublonm and sublatm
oklonCOP = np.round(np.interp(sublonCOP,lonCOP,np.arange(len(lonCOP)))).astype(int)
oklatCOP = np.round(np.interp(sublatCOP,latCOP,np.arange(len(latCOP)))).astype(int)

# getting the model global grid positions for sublonm and sublatm
oklonCOP_glob = np.round(np.interp(sublonCOP,lonCOP_glob,np.arange(len(lonCOP_glob)))).astype(int)
oklatCOP_glob = np.round(np.interp(sublatCOP,latCOP_glob,np.arange(len(latCOP_glob)))).astype(int)

# Getting glider transect from Copernicus model
print('Getting glider transect from Copernicus model')
target_tempCOP = np.empty((len(depthCOP),len(oktimeCOP[0])))
target_tempCOP[:] = np.nan
for i in range(len(oktimeCOP[0])):
    print(len(oktimeCOP[0]),' ',i)
    target_tempCOP[:,i] = COP.variables['thetao'][oktimeCOP[0][i],:,oklatCOP[i],oklonCOP[i]]
target_tempCOP[target_tempCOP < -100] = np.nan

target_saltCOP = np.empty((len(depthCOP),len(oktimeCOP[0])))
target_saltCOP[:] = np.nan
for i in range(len(oktimeCOP[0])):
    print(len(oktimeCOP[0]),' ',i)
    target_saltCOP[:,i] = COP.variables['so'][oktimeCOP[0][i],:,oklatCOP[i],oklonCOP[i]]
target_saltCOP[target_saltCOP < -100] = np.nan

os.system('rm ' + out_dir + '/' + id + '.nc')

# Convertion from GOFS to glider convention
lonGOFSg = np.empty((len(lonGOFS),))
lonGOFSg[:] = np.nan
for i,ii in enumerate(lonGOFS):
    if ii >= 180.0:
        lonGOFSg[i] = ii - 360
    else:
        lonGOFSg[i] = ii
latGOFSg = latGOFS

# Convertion from RTOFS to glider convention
lonRTOFSg = lonRTOFS[0,:]
latRTOFSg = latRTOFS[:,0]

meshlonGOFSg = np.meshgrid(lonGOFSg,latGOFSg)
meshlatGOFSg = np.meshgrid(latGOFSg,lonGOFSg)

meshlonRTOFSg = np.meshgrid(lonRTOFSg,latRTOFSg)
meshlatRTOFSg = np.meshgrid(latRTOFSg,lonRTOFSg)

meshlonCOP = np.meshgrid(lonCOP,latCOP)
meshlatCOP = np.meshgrid(latCOP,lonCOP)

#%% Salinity profile
fig, ax = plt.subplots(figsize=(5.5, 10))

plt.plot(saltg,-depthg,'.',color='cyan')
plt.plot(np.nanmean(saltg_gridded,axis=1),-depthg_gridded,'.-b',\
         label=id[:-14]+' '+str(timeg[0])[0:4]+' '+'['+str(timeg[0])[5:19]+','+str(timeg[-1])[5:19]+']')
plt.plot(target_saltGOFS,-depthGOFS,'.-',color='lightcoral',label='_nolegend_')
plt.plot(np.nanmean(target_saltGOFS,axis=1),-depthGOFS,'.-r',markersize=12,linewidth=2,\
         label='GOFS 3.1'+' '+str(timeGOFS[0].year)+' '+'['+str(timeGOFS[0])[5:13]+','+str(timeGOFS[-1])[5:13]+']')
plt.plot(target_saltRTOFS,-depthRTOFS,'.-',color='mediumseagreen',label='_nolegend_')
plt.plot(np.nanmean(target_saltRTOFS,axis=1),-depthRTOFS,'.-g',markersize=12,linewidth=2,\
         label='RTOFS'+' '+str(timeRTOFS[0].year)+' '+'['+str(timeRTOFS[0])[5:13]+','+str(timeRTOFS[-1])[5:13]+']')
plt.plot(target_saltCOP,-depthCOP,'.-',color='plum',label='_nolegend_')
plt.plot(np.nanmean(target_saltCOP,axis=1),-depthCOP,'.-',color='darkorchid',markersize=12,linewidth=2,\
         label='Copernicus'+' '+str(timeCOP[0].year)+' '+'['+str(timeCOP[0])[5:13]+','+str(timeCOP[-1])[5:13]+']')
plt.ylabel('Depth (m)',fontsize=20)
plt.xlabel('Salinity',fontsize=20)
plt.title('Salinity Profile ' + id,fontsize=20)
#plt.ylim([-np.nanmax(depthg)-100,0.1])
plt.ylim([-75,0.1])
#plt.ylim([-50,0.1])
plt.xlim([35.9,37.0])
plt.legend(loc='lower left',bbox_to_anchor=(-0.1,0.0),fontsize=14)
plt.grid('on')

file = folder_fig + 'salt_profile_' + id + '_' + str(tini).split()[0] + '_' + str(tend).split()[0]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)