#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:09:51 2020

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-100.0,-10.0]
lat_lim = [0.0,60.0]

# Time bounds
min_time = '2018-06-01T00:00:00Z'
max_time = '2018-11-30T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'

#%%

from erddapy import ERDDAP
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import cmocean

#%% Look for datasets 

e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

kw2018 = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[-1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[-1],
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
    'latitude<=': lat_lim[-1],
    'longitude>=': lon_lim[0],
    'longitude<=': lon_lim[-1],
}

variables = [
 'time','latitude','longitude'
]

#%%

e = ERDDAP(
    server=server,
    protocol='tabledap',
    response='nc'
)
'''
for id in gliders:
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables
    
    df = e.to_pandas(
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()
    
    print(id,df.index[-1])
'''

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
    
#%% Map all gliders during hurricane season 2018

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 10))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)    
plt.yticks([])
plt.xticks([])
plt.title('Glider Tracks Hurricane Season 2018',fontsize=30)

for id in gliders:
    if id[0:2] == 'ng':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='darkorange',\
                markersize=1)
        if np.mean(df['latitude (degrees_north)']) > 23:
            ax.plot(np.mean(df['longitude (degrees_east)']),\
                np.mean(df['latitude (degrees_north)']),'*k',\
                label=id.split('-')[0]) 
        if np.mean(df['latitude (degrees_north)']) < 23:
            ax.plot(np.mean(df['longitude (degrees_east)']),\
                np.mean(df['latitude (degrees_north)']),'^',color='firebrick',\
                label=id.split('-')[0]) 
                
        #ax.text(np.mean(df['longitude (degrees_east)']),np.mean(df['latitude (degrees_north)']),\
        #        id.split('-')[0],weight='bold',
        #        bbox=dict(facecolor='white',alpha=0.4,edgecolor='none'))
        '''    
        if id[0:5] == 'ng289' or id[0:5] == 'ng619' or id[0:5] == 'ng231':
            print(id)
            ax.text(np.mean(df['longitude (degrees_east)']),np.mean(df['latitude (degrees_north)']),\
                id.split('-')[0],weight='bold',
                bbox=dict(facecolor='white',alpha=0.4,edgecolor='none'))
        if id[0:5] == 'ng618' or id[0:5] == 'ng278':
            print(id)
            ax.text(np.mean(df['longitude (degrees_east)']),np.mean(df['latitude (degrees_north)']-2),\
                id.split('-')[0],weight='bold',
                bbox=dict(facecolor='white',alpha=0.4,edgecolor='none'))
        if id[0:5] == 'ng282':
            print(id)
            ax.text(np.mean(df['longitude (degrees_east)']),np.mean(df['latitude (degrees_north)']-4),\
                id.split('-')[0],weight='bold',
                bbox=dict(facecolor='white',alpha=0.4,edgecolor='none'))
        '''
plt.axis('scaled') 
plt.axis([-100,-60,10,45])
plt.legend(loc='upper left')
#plt.axis([-100,-10,0,60])
#plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2019.png",\
#            bbox_inches = 'tight',pad_inches = 0.1)
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/map_navy_gliders_hurric_season_2018.png",\
            bbox_inches = 'tight',pad_inches = 0.1)    