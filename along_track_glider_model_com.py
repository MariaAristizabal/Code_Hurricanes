#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 10:21:19 2018

@author: aristizabal
"""

#%% User input

# lat and lon limits
lon_min = -90.0
lon_max = -80.0
lat_min = 20.0
lat_max = 30.0

# date limits
date_ini = '2018-10-01T00:00:00Z'
date_end = '2018-10-08T00:00:00Z'


# Glider dac location
server = 'https://data.ioos.us/gliders/erddap'

#GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'



# Bathymetry data
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

#%%

from erddapy import ERDDAP
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns # package for nice plotting defaults
sns.set()

from mpl_toolkits.basemap import Basemap

import netCDF4
from netCDF4 import Dataset

import numpy as np

import datetime

#%% Look for datasets 

e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw2017 = {
    'min_lon': lon_min,
    'max_lon': lon_max,
    'min_lat': lat_min,
    'max_lat': lat_max,
    'min_time': date_ini,
    'max_time': date_end,
}

search_url = e.get_search_url(response='csv', **kw2017)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

#%%

server = 'https://data.ioos.us/gliders/erddap'

dataset_id = gliders[4]

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

e.dataset_id=gliders[4]
e.constraints=constraints
e.variables=variables

print(e.get_download_url())


#%%

df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()

df.head()

#%% Read Bathymetry data

Bathy = Dataset(bath_data)

latbath = Bathy.variables['lat'][:]
lonbath = Bathy.variables['lon'][:]
elevbath = Bathy.variables['elevation'][:]

#%% Read GOFS 3.1 output

GOFS31 = Dataset(catalog31)

lat31 = GOFS31.variables['lat'][:]
lon31 = GOFS31.variables['lon'][:]
depth = GOFS31.variables['depth'][:]
time31 = GOFS31.variables['time']
time31 = netCDF4.num2date(time31[:],time31.units) 

tmin = datetime.datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

oktime31 = np.where(np.logical_and(time31 >= tmin, time31 <= tmax))

# Conversion from glider longitude and latitude to GOFS convention

target_lon = np.empty((len(df['longitude']),))
target_lon[:] = np.nan
for i in range(len(df['longitude'])):
    if df['longitude'][i] < 0: 
        target_lon[i] = 360 + df['longitude'][i]
    else:
        target_lon[i] = df['longitude'][i]
target_lat = df['latitude'][:]

#%%

sublon31=np.interp(df.index,target_lon,time31[oktime31])

time = np.array(df.index.to_pydatetime(), dtype=numpy.datetime64)

sublon31=np.interp(time31[oktime31],time,target_lon)

sublat31=interp1(time,target_lat,time31(oktime31));

oklon31 = np.round(np.interp(target_lon,lon31,np.arange(len(lon31))))
oklat31 = np.round(np.interp(target_lat,lat31,np.arange(len(lat31))))

target_temp31_41025 = GOFS31.variables['water_temp'][oktime31[0],1,oklat31,oklon31]


#%%
sublon31=interp1(time,target_lon,time31(oktime31));
sublat31=interp1(time,target_lat,time31(oktime31));

oklon31=np.round(interp1(lon31,1:length(lon31),sublon31));
oklat31=np.round(interp1(lat31,1:length(lat31),sublat31));

target_temp31(length(depth31),length(oktime31))=nan;
for i=1:length(oklon31)
    target_temp31(:,i) = squeeze(double(ncread(catalog31,'water_temp',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
end



#%%

fig, ax=plt.subplots(figsize=(6, 10), dpi=150, facecolor='w', edgecolor='w')

ax1 = plt.subplot(511)
ax1.contour(lonbath,latbath,elevbath,colors='k')
plt.axis('equal')
plt.axis([lon_min,lon_max,lat_min,lat_max])
ax1.plot(df['longitude'],df['latitude'],'*')

ax2 = plt.subplot(512)
kw = dict(s=15, c=df['temperature'], marker='*', edgecolor='none')
cs = ax2.scatter(df.index, df['depth'], **kw, cmap='RdYlBu_r')

ax2.invert_yaxis()
ax2.set_xlim(df.index[0], df.index[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax2.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical', extend='both')
cbar.ax.set_ylabel('Temperature ($^\circ$C)')
ax2.set_ylabel('Depth (m)');


