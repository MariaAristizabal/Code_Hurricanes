#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:44:20 2018

@author: aristizabal
"""

#%% User input

# Caribbean
lon_lim = [-69,-63];
lat_lim = [16,22];

# Time bounds
min_time = '2018-06-01T00:00:00Z'
max_time = '2018-11-30T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

# lat lon GOFS 3.1 file
#lat31_lon31_matfile = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU22_GOFS31.mat'
lat31_lon31_matfile = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_dimensions.mat'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'
#server = 'https://glider.ioos.us/gliders/erddap'

#%%

from erddapy import ERDDAP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import scipy.io as sio 

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
    index_col='time (UTC)',
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
         
#%% Reading lat and lon from GOFS 3.1

latlon_array = sio.loadmat(lat31_lon31_matfile) 
lat31 = latlon_array['lat31']
lon31 = latlon_array['lon31']

# Converting lat31 and lon31 to glider convention
lon31_g = np.empty((len(lon31)))
lon31_g[:] = np.nan
for x,lon in enumerate(lon31):
    if lon[0] > 180:
        lon31_g[x] = lon[0] - 360
    else:
        lon31_g[x] = lon[0]

lat31_g = lat31    

meshlon31 = np.meshgrid(lon31_g,lat31_g)
meshlat31 = np.meshgrid(lat31_g,lon31_g)

#%% Reading glider data and plotting lat and lon in the map
    
fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
#ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lon,bath_lat,-bath_elev)#,cmap=plt.get_cmap('BrBG'))
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Gliders in the Caribbean during hurricane season 2018 ',size = 20)
plt.yticks([])
plt.xticks([])

for i, id in enumerate(gliders):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        if np.logical_and(id.split('-')[0][0:2] == 'ng',id != 'ng616-20180701T0000'):
            ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),'o',markersize = 10,\
                markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
        else:
            if id == 'ng616-20180701T0000':
                ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),'o',markersize = 15,\
                        markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
            else:
                ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),'^',markersize = 10,\
                        markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
        
        ax.legend(loc='upper right',fontsize = 12)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Caribbean_map_hurr_2018.png"\
             ,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Reading glider data and plotting glider track in the map
    
fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
#ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lon,bath_lat,-bath_elev)#,cmap=plt.get_cmap('BrBG'))
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Gliders in the Caribbean during hurricane season 2018 ',size = 20)
plt.yticks([])
plt.xticks([])


for i, id in enumerate(gliders):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],df['latitude (degrees_north)'],'.',markersize=2, color='k')

for i, id in enumerate(gliders):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        if np.logical_and(id.split('-')[0][0:2] == 'ng',id != 'ng616-20180701T0000'):
            ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),'o',markersize = 10,\
                markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
        else:
            if id == 'ng616-20180701T0000':
                ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),'o',markersize = 15,\
                        markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
            else:
                ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),'^',markersize = 10,\
                        markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
        
        #ax.legend(loc='upper right',fontsize = siz)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Caribbean_map2_hurr_2018.png"\
             ,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Map of channel between Costa Rica and US Virgin Island

siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lon,bath_lat,-bath_elev,cmap=plt.get_cmap('BrBG'))
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Gliders in the Virgin Island Channel \n during hurricane season 2018 ',size = 20)
plt.xlim([-66,-64])
plt.ylim([17,19])
plt.yticks([])
plt.xticks([])

for i, id in enumerate(gliders[[4,3,0]]):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        if id == 'ng616-20180701T0000':
            ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),'o',markersize = 15,\
                markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
        else:
            ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),'o',markersize = 10,\
                markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
        ax.legend(loc='upper right',fontsize = 22)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Caribbean_map_channel_hurr_2018.png"\
             ,bbox_inches = 'tight',pad_inches = 0) 

#%% Map of channel between Costa Rica and US Virgin Island

siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lon,bath_lat,-bath_elev,cmap=plt.get_cmap('BrBG'))
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Gliders in the Virgin Island Channel \n during hurricane season 2018 ',size = 20)
plt.xlim([-66,-64])
plt.ylim([17,19])
plt.yticks([])
plt.xticks([])

for i, id in enumerate(gliders[[4,3,0]]):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],df['latitude (degrees_north)'],'o',color='k',markersize = 5,\
                markeredgecolor='black', markeredgewidth=2,label = ' ') 
        ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),'o',markersize = 15,\
                markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
        #ax.legend(loc='upper right',fontsize = 18)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Caribbean_map_channel2_hurr_2018.png"\
             ,bbox_inches = 'tight',pad_inches = 0) 
   
#%% Map of channel between Costa Rica and US Virgin Island 
#with grid points

# Find grid points between ng291 and ng487
lon_lim2 = (-65.0 , -64.5)
lat_lim2 = (17.75 , 18.25)
lon_pos = np.where(np.logical_and(lon31_g > lon_lim2[0], lon31_g < lon_lim2[1]))
lat_pos = np.where(np.logical_and(lat31_g[:,0] > lat_lim2[0], lat31_g[:,0] < lat_lim2[1]))

siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lon,bath_lat,-bath_elev,cmap=plt.get_cmap('BrBG'))
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Gliders in the Virgin Island Channel \n during hurricane season 2018 ',size = 20)
plt.xlim([-66,-64])
plt.ylim([17,19])

plt.plot(meshlon31[0],meshlat31[0].T,'.k')
plt.plot(lon31_g[lon_pos[0][1]],lat31_g[lat_pos[0][3]],'*r')
#plt.plot(lon31_g[lon_pos[0][2]],lat31_g[lat_pos[0][3]],'*r')
plt.plot(lon31_g[lon_pos[0][1]],lat31_g[lat_pos[0][4]],'*r')
#plt.plot(lon31_g[lon_pos[0][2]],lat31_g[lat_pos[0][4]],'*r')
plt.plot(lon31_g[lon_pos[0][1]],lat31_g[lat_pos[0][2]],'*r')
#plt.plot(lon31_g[lon_pos[0][2]],lat31_g[lat_pos[0][2]],'*r')

for i, id in enumerate(gliders[[4,3,0]]):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        if id == 'ng616-20180701T0000':
            ax.plot(df['longitude'].mean(),df['latitude'].mean(),'o',markersize = 15,\
                markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
        else:
            ax.plot(df['longitude'].mean(),df['latitude'].mean(),'o',markersize = 10,\
                markeredgecolor='black', markeredgewidth=2,label = id.split('-')[0]) 
        ax.legend(loc='upper right',fontsize = 18)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Caribbean_map_channel_hurr_2018_grid.png"\
             ,bbox_inches = 'tight',pad_inches = 0) 