#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:14:09 2020

@author: aristizabal
"""
#%% User input

# lat and lon bounds
#lon_lim = [-100.0,-10.0]
#lat_lim = [0.0,60.0]

lat_lim = [10,25]
lon_lim = [-85,-60]

# Time bounds
min_time = '2019-06-01T00:00:00Z'
max_time = '2019-11-30T00:00:00Z'

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

for id in gliders:
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables
    
    df = e.to_pandas(
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()
    
    print(id,df.index[-1])


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

#%% Map

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(7, 5))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo) 
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell') 
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')   
#plt.yticks([])
#plt.xticks([])
plt.title('Glider Tracks Hurricane Season 2019',fontsize=16)
   
for id in gliders:
    if id[0:4] != 'glos':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='darkorange',markersize=1)   
  
        #ax.text(np.mean(df['longitude (degrees_east)']),np.mean(df['latitude (degrees_north)']),id.split('-')[0])
        
#plt.legend(loc='upper left',bbox_to_anchor=[-0.22,1.0])     
plt.axis('scaled') 
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.savefig("/Users/aristizabal/Desktop/map_gliders_hurric_track_season_2019.png",\
            bbox_inches = 'tight',pad_inches = 0.1)
    
#%% Map

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(7, 5))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo) 
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell') 
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')   
#plt.yticks([])
#plt.xticks([])
plt.title('Glider Tracks Hurricane Season 2019',fontsize=16)
   
for id in gliders:
    if id[0:4] != 'glos':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='darkorange',markersize=1)   
  
        ax.text(np.mean(df['longitude (degrees_east)']),np.mean(df['latitude (degrees_north)']),id.split('-')[0])
        
#plt.legend(loc='upper left',bbox_to_anchor=[-0.22,1.0])     
plt.axis('scaled') 
plt.axis([-70,-62,14,22])
plt.savefig("/Users/aristizabal/Desktop/map_gliders_hurric_track_season_2019.png",\
            bbox_inches = 'tight',pad_inches = 0.1)
    
#%% Look for datasets 2018
    
# Time bounds
min_time = '2018-06-01T00:00:00Z'
max_time = '2018-11-30T00:00:00Z'    

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

for id in gliders:
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables
    
    df = e.to_pandas(
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()
    
    print(id,df.index[-1])


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

#%% Map 2018

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(7, 5))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo) 
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell') 
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')   
#plt.yticks([])
#plt.xticks([])
plt.title('Glider Tracks Hurricane Season 2018',fontsize=16)
   
for id in gliders:
    if id[0:5] != 'silbo':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='darkorange',markersize=1)   
  
        #ax.text(np.mean(df['longitude (degrees_east)']),np.mean(df['latitude (degrees_north)']),id.split('-')[0])
        
#plt.legend(loc='upper left',bbox_to_anchor=[-0.22,1.0])     
plt.axis('scaled') 
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.savefig("/Users/aristizabal/Desktop/map_gliders_hurric_track_season_2018.png",\
            bbox_inches = 'tight',pad_inches = 0.1)
    
#%% Map 2018

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(7, 5))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo) 
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell') 
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')   
#plt.yticks([])
#plt.xticks([])
plt.title('Glider Tracks Hurricane Season 2018',fontsize=16)
   
for id in gliders:
    if id[0:5] != 'silbo':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='darkorange',markersize=1)   
  
        ax.text(np.mean(df['longitude (degrees_east)']),np.mean(df['latitude (degrees_north)']),id.split('-')[0])
        
#plt.legend(loc='upper left',bbox_to_anchor=[-0.22,1.0])     
plt.axis('scaled') 
plt.axis([-70,-62,14,22])
plt.savefig("/Users/aristizabal/Desktop/map_gliders_hurric_track_season_2018.png",\
            bbox_inches = 'tight',pad_inches = 0.1)    
    