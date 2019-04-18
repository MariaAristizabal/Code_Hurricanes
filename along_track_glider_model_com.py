#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 10:21:19 2018

@author: aristizabal
"""

#%% User input

# lat and lon limits


# RU22
lon_min = 125.0
lon_max = 127.0
lat_min = 32.0
lat_max = 34.0

# date limits
date_ini = '2018-08-01T00:00:00Z'
date_end = '2018-08-26T00:00:00Z'


'''
# ng288
lon_min = -90.0
lon_max = -80.0
lat_min = 26.0
lat_max = 28.0

date_ini = '2018-10-07T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'
'''

# Glider dac location
server = 'https://data.ioos.us/gliders/erddap'

#GOFS3.1 outout model location
#catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
catalog31 = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/ts3z.nc'

# Bathymetry data
#bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

#%%

from erddapy import ERDDAP
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#from mpl_toolkits.basemap import Basemap

import netCDF4
from netCDF4 import Dataset
import xarray as xr

import numpy as np

import datetime
import time

#%% Look for datasets 

e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw2018 = {
    'min_lon': lon_min,
    'max_lon': lon_max,
    'min_lat': lat_min,
    'max_lat': lat_max,
    'min_time': date_ini,
    'max_time': date_end,
}

search_url = e.get_search_url(response='csv', **kw2018)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

#%%

server = 'https://data.ioos.us/gliders/erddap'

dataset_id = gliders[0]

constraints = {
    'time>=': date_ini,
    'time<=': date_end,
    'latitude>=': lat_min,
    'latitude<=': lat_max,
    'longitude>=': lon_min,
    'longitude<=': lon_max,
}

variables = [
 'depth',
 'latitude',
 'longitude',
 'salinity',
 'temperature',
 'time',
]

#%%

e = ERDDAP(
    server=server,
    protocol='tabledap',
    response='nc'
)

e.dataset_id=gliders[0]
e.constraints=constraints
e.variables=variables

print(e.get_download_url())

#%% Using xarray
'''
ds = e.to_xarray(decode_times=False)
ds.time
'''

#%%

df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()

df.head()

dff = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
)

#%% Coverting glider vectors into arrays

timeg, ind = np.unique(df.index.values,return_index=True)
latg = np.unique(df.latitude.values)
long = np.unique(df.longitude.values)

dg = df.depth.values
tg = df.temperature.values
sg = df.salinity.values

zn = np.int(np.max(dg)/0.3)

depthg = np.empty((zn,len(timeg)))
depthg[:] = np.nan
tempg = np.empty((zn,len(timeg)))
tempg[:] = np.nan
saltg = np.empty((zn,len(timeg)))
saltg[:] = np.nan
for i,ii in enumerate(ind):
    print(i)
    if i < len(timeg)-1:
        depthg[0:len(dg[ind[i]:ind[i+1]]),i] = dg[ind[i]:ind[i+1]] 
        tempg[0:len(tg[ind[i]:ind[i+1]]),i] = tg[ind[i]:ind[i+1]]
        saltg[0:len(sg[ind[i]:ind[i+1]]),i] = sg[ind[i]:ind[i+1]]
    else:
        depthg[0:len(dg[ind[i]:len(dg)]),i] = dg[ind[i]:len(dg)] 
        tempg[0:len(tg[ind[i]:len(tg)]),i] = tg[ind[i]:len(tg)]
        saltg[0:len(sg[ind[i]:len(sg)]),i] = sg[ind[i]:len(sg)]
        
#%% Grid variables

depthg_gridded = np.arange(0,np.nanmax(depthg),0.5)
tempg_gridded = np.empty((len(depthg_gridded),len(timeg)))
tempg_gridded[:] = np.nan

for t,tt in enumerate(timeg):
    depthu,oku = np.unique(depthg[:,t],return_index=True)
    tempu = tempg[oku,t]
    okdd = np.isfinite(depthu)
    depthf = depthu[okdd]
    tempf = tempu[okdd]
    ok = np.isfinite(tempf)
    if np.sum(ok) < 3:
        tempg_gridded[:,t] = np.nan
    else:
        okd = depthg_gridded < np.max(depthf[ok])
        tempg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[ok],tempf[ok])
    
#%% Read Bathymetry data
'''
Bathy = Dataset(bath_data)

latbath = Bathy.variables['lat'][:]
lonbath = Bathy.variables['lon'][:]
elevbath = Bathy.variables['elevation'][:]
'''

#%% Read GOFS 3.1 output

GOFS31 = xr.open_dataset(catalog31,decode_times=False)

lat31 = GOFS31.lat
lon31 = GOFS31.variables['lon'][:]
depth31 = GOFS31.variables['depth'][:]
tt31 = GOFS31.variables['time']
#t31 = netCDF4.num2date(tt31[:],tt31.units) 
t31 = netCDF4.num2date(tt31[:],'hours since 2000-01-01 00:00:00') 

tmin = datetime.datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

oktime31 = np.where(np.logical_and(t31 >= tmin, t31 <= tmax))

# Conversion from glider longitude and latitude to GOFS convention
target_lon = np.empty((len(df['longitude']),))
target_lon[:] = np.nan
for i in range(len(df['longitude'])):
    if df['longitude'][i] < 0: 
        target_lon[i] = 360 + df['longitude'][i]
    else:
        target_lon[i] = df['longitude'][i]
target_lat = df['latitude'][:]

################
'''
ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.lat
bath_lon = ncbath.lon
bath_elev = ncbath.elevation

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevsub = bath_elev[oklatbath,oklonbath]
'''

#%% Changing times to timestamp

timeg = [time.mktime(df.index[i].timetuple()) for i in np.arange(len(df))]
time31 = [time.mktime(t31[i].timetuple()) for i in np.arange(len(t31))]

# interpolating glider lon and lat to lat and lon on model time
sublon31=np.interp(time31,timeg,target_lon)
sublat31=np.interp(time31,timeg,target_lat)

# getting the model grid positions for sublon31 and sublat31
oklon31=np.round(np.interp(sublon31,lon31,np.arange(len(lon31)))).astype(int)
oklat31=np.round(np.interp(sublat31,lat31,np.arange(len(lat31)))).astype(int)

#mat_lat_lon = np.array([oklon31,oklat31])
#n_grid_points1 = len(np.unique(mat_lat_lon.T,axis=0))

#%%

target_temp31 = np.empty((len(depth31),len(oktime31[0])))
target_temp31[:] = np.nan
for i in range(len(oktime31[0])):
    target_temp31[:,i] = GOFS31.variables['water_temp'][oktime31[0][i],:,oklat31[i],oklon31[i]]

target_temp31[target_temp31 < -100] = np.nan

#%%

fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

kw = dict(s=30, c=df['temperature'], marker='*', edgecolor='none')
cs = ax.scatter(df.index, -df['depth'], **kw, cmap='RdYlBu_r')

ax.set_xlim(df.index[0], df.index[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('Temperature ($^\circ$C)')
ax.set_ylabel('Depth (m)');

plt.show()

#%%

fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

kw = dict(levels = np.linspace(np.nanmin(tempg_gridded),np.nanmax(tempg_gridded),10))
ax.contour(timeg,-depthg_gridded,tempg_gridded,colors = 'lightgrey',**kw)
ax.contourf(timeg,-depthg_gridded,tempg_gridded,cmap='RdYlBu_r',**kw)


ax.set_xlim(df.index[0], df.index[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

#cb = plt.colorbar()
#cb.set_label('Salinity',rotation=270, labelpad=25, fontsize=12)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('Temperature ($^\circ$C)')
ax.set_ylabel('Depth (m)');

plt.show()

#%%
fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

ax.contourf(t31[oktime31],depth31,target_temp31,cmap='RdYlBu_r')
plt.ylim(0,100)

ax.invert_yaxis()
#ax2.set_xlim(df.index[0], df.index[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('Temperature ($^\circ$C)')
ax.set_ylabel('Depth (m)');
