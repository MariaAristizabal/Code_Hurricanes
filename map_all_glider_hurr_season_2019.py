#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:16:57 2020

@author: root
"""

#%% User input

# lat and lon bounds
lon_lim = [-100.0,-10.0]
lat_lim = [0.0,50.0]

# Time bounds
min_time = '2019-06-01T00:00:00Z'
max_time = '2019-11-30T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'

# Jun1 -Jun30
Dir_Argo1 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2019/DataSelection_20200131_211606_9547943'

# Jul1 - Jul31
Dir_Argo2 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2019/DataSelection_20200131_211715_9547947'

# Aug1 - Aug30
Dir_Argo3 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2019/DataSelection_20200131_211807_9547965'

# Sep1 - Sep31
Dir_Argo4 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2019/DataSelection_20200131_211910_9547986'

# Oct1 - Oct31
Dir_Argo5 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2019/DataSelection_20200131_212020_9548037'

# Nov1 - Nov30
Dir_Argo6 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2019/DataSelection_20200131_212231_9548139'

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

#%% Reading Argo data

argo_files1 = sorted(glob.glob(os.path.join(Dir_Argo1,'*profiles*.nc')))
argo_files2 = sorted(glob.glob(os.path.join(Dir_Argo2,'*profiles*.nc')))
argo_files3 = sorted(glob.glob(os.path.join(Dir_Argo3,'*profiles*.nc')))
argo_files4 = sorted(glob.glob(os.path.join(Dir_Argo4,'*profiles*.nc')))
argo_files5 = sorted(glob.glob(os.path.join(Dir_Argo5,'*profiles*.nc')))
argo_files6 = sorted(glob.glob(os.path.join(Dir_Argo6,'*profiles*.nc')))

ncargo = Dataset(argo_files1[-1])
argo_id = ncargo.variables['PLATFORM_NUMBER'][:]
argo_lat = ncargo.variables['LATITUDE'][:]
argo_lon = ncargo.variables['LONGITUDE'][:]

argo_tim = ncargo.variables['JULD']#[:]
argo_time = netCDF4.num2date(argo_tim[:],argo_tim.units) 

#%% Plotting Argo time

plt.figure()
for l in argo_files1:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    argo_tim = ncargo.variables['JULD']
    argo_time = np.asarray(netCDF4.num2date(argo_tim[:],argo_tim.units))
    plt.plot(argo_time,'*')
    
#%% Finding number of Argo profiles

argo_files = argo_files1 + argo_files2 + argo_files3 + \
             argo_files4 + argo_files5 + argo_files6

Argo_time = np.empty((0))
#Argo_lon = np.empty((0))
#Argo_lat = np.empty((0))
#Argo_temp = np.empty((0))

for i,l in enumerate(argo_files):
    print(len(argo_files),' ', i)
    ncargo = Dataset(l)
    #argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    #argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    #argo_temp = np.asarray(ncargo.variables['TEMP'][:,0])
    argo_tim = ncargo.variables['JULD']    
    argo_time = np.asarray(netCDF4.num2date(argo_tim[:],argo_tim.units))
    Argo_time = np.concatenate((Argo_time,argo_time))
    #Argo_lon = np.concatenate((Argo_lon,argo_lon))
    #Argo_lat = np.concatenate((Argo_lat,argo_lat))
    #Argo_temp = np.concatenate((Argo_temp,argo_temp))

Number_argo_profiles = Argo_time.shape
    

#%% Map all gliders durimg hurricane season 2019

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
plt.axis([-100,-50,10,50])
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2019.png",\
            bbox_inches = 'tight',pad_inches = 0.1)

#%% Map with ARGO + glider during hurricane season 2019

'''    
lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 5))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)    
plt.yticks([])
plt.xticks([])
plt.axis([-100,-10,0,50])    
'''
    
fig, ax = plt.subplots(figsize=(10, 5))
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,cmap='Blues_r')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.yticks([])
plt.xticks([])
plt.axis([-100,-10,0,50])

for l in argo_files1:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')
    
for l in argo_files2:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')    
    
for l in argo_files3:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')
    
for l in argo_files4:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')

for l in argo_files5:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:]
    argo_lon = ncargo.variables['LONGITUDE'][:]
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')

for l in argo_files6:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')  

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
                df['latitude (degrees_north)'],'.',color='r',markersize=1)   
        #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_ARGOS_gliders_hurric_season_2019.png",\
            bbox_inches = 'tight',pad_inches = 0.1)
    