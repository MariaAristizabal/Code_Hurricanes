#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:37:32 2020

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

#%%

from erddapy import ERDDAP
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import cmocean
import glob
import os
from netCDF4 import Dataset
import netCDF4 
from bs4 import BeautifulSoup
from zipfile import ZipFile
import cartopy
import cartopy.feature as cfeature
#from datetime import datetime

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

#%% Reading and plotting best track data

track_files = sorted(glob.glob(os.path.join(track_folder+year+'/','*'+basin+'*kmz')))

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 10))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo) 
#plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell') 
#plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')   
plt.yticks([])
plt.xticks([])
plt.title('Glider Tracks and Storm Tracks \n Hurricane Season 2020',fontsize=30)

for n,f in enumerate(track_files):
    print(f)
    os.system('cp ' + f + ' ' + f[:-3] + 'zip')
    os.system('unzip -o ' + f + ' -d ' + track_files[0][:-4])
    zip_handle = ZipFile(f[:-3]+'zip', 'r')
    kml_file = f.split('/')[-1].split('_')[0] + '.kml'
    kml_best_track = zip_handle.open(kml_file, 'r').read()
    
    # best track coordinates
    soup = BeautifulSoup(kml_best_track,'html.parser')
    
    lon_best_track = np.empty(len(soup.find_all("point")))
    lon_best_track[:] = np.nan
    lat_best_track = np.empty(len(soup.find_all("point")))
    lat_best_track[:] = np.nan
    for i,s in enumerate(soup.find_all("point")):
        #print(s.get_text("coordinates"))
        if len(s.get_text("coordinates").split('coordinates')) != 1:
            lon_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
            lat_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])
        
        stormname = soup.find_all("stormname")[-1].get_text('stormname')
    '''  
    cat = []
    for i,s in enumerate(soup.find_all("styleurl")):
        cat.append(s.get_text('#').split('#')[-1]) 
    cat = np.asarray(cat)      
    '''

    plt.text(lon_best_track[0],lat_best_track[0],str(n+1),color='k',\
             bbox=dict(facecolor='white', edgecolor='none',alpha=0.4))
    plt.plot(lon_best_track,lat_best_track,'.-k',label=str(n+1) + ' ' + stormname)

#%%   
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
  
plt.legend(loc='upper left',bbox_to_anchor=[-0.32,1.0])     
plt.axis('scaled') 
#plt.axis([-100,-50,10,50])
plt.axis([-100,-57,10,50])
#plt.axis([-100,-10,10,60])
#plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_track_season_2019.png",\
#            bbox_inches = 'tight',pad_inches = 0.1)
plt.savefig("/Users/aristizabal/Desktop/map_gliders_hurric_track_season_2020.png",\
            bbox_inches = 'tight',pad_inches = 0.1)
    
#%% Map all gliders during hurricane season 2020

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 10))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)    
plt.yticks([])
plt.xticks([])
plt.title('Glider Tracks Hurricane Season 2019',fontsize=30)

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
        #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])

plt.axis('scaled') 
#plt.axis([-100,-50,10,50])
plt.axis([-100,-10,0,60])
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2019.png",\
            bbox_inches = 'tight',pad_inches = 0.1)