#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:42:39 2019

@author: aristizabal
"""

#%% User input

# Caribbean
lon_lim = [-75,-72];
lat_lim = [38,40];

# Time bounds
min_time = '2018-08-01T00:00:00Z'
max_time = '2018-08-09T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'

#%%

from erddapy import ERDDAP
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pytz
from netCDF4 import Dataset

#%% Look for datasets 

e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw2018 = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': min_time,
    'max_time': max_time,
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

constraints = {
    'time>=': min_time,
    'time<=': max_time,
    'latitude>=': lat_lim[0],
    'latitude<=': lat_lim[1],
    'longitude>=': lon_lim[0],
    'longitude<=': lon_lim[1],
}

variables = ['latitude','longitude','time']

#%%

e = ERDDAP(
    server=server,
    protocol='tabledap',
    response='nc'
)

for id in gliders:
    e.dataset_id=id
    e.constraints=constraints
    e.variables=variables
    
    df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
                    ).dropna()
    
#%% Reading bathymetry data

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]        

#%% Reading glider data and plotting the tracks on the map
    
siz=12
i = 1

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lon,bath_lat,-bath_elev,cmap=plt.get_cmap('BrBG'))
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title(gliders[1].split('-')[0]+'-'+min_time[0:10]+'-'+max_time[0:10],size = 18)

i = 1
e.dataset_id = gliders[i]
e.constraints = constraints
e.variables = variables
    
df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
    ).dropna()
plt.plot(df['longitude'],df['latitude'],'.-',markersize = 2,\
         color='black',label='Track')
ax.plot(df['longitude'].mean(),df['latitude'].mean(),'o',markersize = 10,\
        markeredgecolor='black', markeredgewidth=2,label=gliders[i].split('-')[0])
ax.legend(loc='upper right',fontsize = siz)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ru33_track.png"\
             ,bbox_inches = 'tight',pad_inches = 0)  