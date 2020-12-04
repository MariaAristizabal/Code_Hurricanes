#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:56:51 2020

@author: aristizabal
"""

#%% User input

# lat and lon bounds
#lon_lim = [-100.0,-10.0]
#lat_lim = [0.0,60.0]

# lat and lon bounds
lon_lim = [-100.0,-10.0]
lat_lim = [0.0,50.0]

# Time bounds
min_time = '2020-06-01T00:00:00Z'
max_time = '2020-11-30T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/gebco_2020_n50.0_s0.0_w-100.0_e0.0.nc'
#bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'

# storm track files
track_folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/'
basin = 'al'
year = '2020'
fname = '_best_track.kmz' 

# Argo floats
url_Argo = 'http://www.ifremer.fr/erddap'

#%%

from erddapy import ERDDAP
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import cmocean
import glob
import os
#from netCDF4 import Dataset
#import netCDF4 
from bs4 import BeautifulSoup
from zipfile import ZipFile
import cartopy
import cartopy.feature as cfeature
from datetime import datetime

#%% Look for datasets 

e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[-1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[-1],
    'min_time': min_time,
    'max_time': max_time,
}

search_url = e.get_search_url(response='csv', **kw)
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
    parse_dates=True)
    
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

#%% Look for Argo datasets 

e = ERDDAP(server = url_Argo)

# Grab every dataset available
#datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': min_time,
    'max_time': max_time,
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
    'time>=': min_time,
    'time<=': max_time,
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
#argo_press = np.asarray(df['pres (decibar)'])
argo_lons = np.asarray(df['longitude (degrees_east)'])
argo_lats = np.asarray(df['latitude (degrees_north)'])
#argo_temps = np.asarray(df['temp (degree_Celsius)'])
#argo_salts = np.asarray(df['psal (PSU)']) 

Number_argo_profiles = np.max([np.unique(argo_lons).shape,\
                               np.unique(argo_lats).shape,\
                               np.unique(argo_times).shape]) 

#%%
lev = np.arange(-9000,9100,100)
#fig, ax = plt.subplots(figsize=(10, 5))
fig, ax = plt.subplots(figsize=(10, 5),subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo) 
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')   
plt.yticks([])
plt.xticks([]) 

coast = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
ax.add_feature(coast, edgecolor='black', facecolor='none')
ax.add_feature(cfeature.BORDERS)  # adds country borders  

argo_idd = np.unique(argo_ids)
  

for i,id in enumerate(argo_idd):
    print(id)
    okind = np.where(argo_ids == id)[0]
    argo_time = np.asarray([datetime.strptime(t,'%Y-%m-%dT%H:%M:%SZ') for t in argo_times[okind]])
    argo_lon = argo_lons[okind]
    argo_lat = argo_lats[okind]
        
    ax.plot(argo_lon,argo_lat,'ok',markersize = 4,markeredgecolor='g') 

for id in gliders:        
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables

    df = e.to_pandas(
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
        ).dropna()
    if ~np.logical_and(np.mean(df['longitude (degrees_east)']) < -80,\
                      np.mean(df['latitude (degrees_north)']) > 40):
        print(id)
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='r',markersize=1)   

plt.axis('scaled') 
plt.axis([-100,-10,0,50])
#plt.axis([-100,-10,0,60])
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2020_Argos.png",\
            bbox_inches = 'tight',pad_inches = 0.1)