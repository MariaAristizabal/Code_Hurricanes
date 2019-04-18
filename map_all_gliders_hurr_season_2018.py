#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:51:16 2018

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-110.0,-20.0]
lat_lim = [15.0,45.0]

# Time bounds
min_time = '2018-06-01T00:00:00Z'
max_time = '2018-11-30T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'


#%%

from erddapy import ERDDAP
import pandas as pd

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
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()
    
    print(id,df.index[-1])


#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
#plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
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
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*',markersize = 10 )   
        #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018.png")
plt.show() 

#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*', markersize = 10)   
        #ax.legend(loc='upper left',fontsize = 16)
        ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
        
plt.axis('equal')
plt.axis([-94,-82,25,28.5])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_Gulf.png")
plt.show() 
 
#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*', markersize = 10)   
        #ax.legend(loc='upper left',fontsize = 16)
        ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
        
plt.axis('equal')
plt.axis([-69,-62,16,22])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_Caribbean.png")
plt.show() 

#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*', markersize = 10)   
        #ax.legend(loc='upper left',fontsize = 16)
        ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
        
plt.axis('equal')
plt.axis([-80,-75,28,36])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_SAB.png")
plt.show() 

#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*', markersize = 10)   
        #ax.legend(loc='upper left',fontsize = 16)
        ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
        
plt.axis('equal')
plt.axis([-80,-75,28,35.5])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_SAB.png")
plt.show() 

#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*', markersize = 10)   
        #ax.legend(loc='upper left',fontsize = 16)
        ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
        
plt.axis('equal')
plt.axis([-75,-68,34.5,42])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_MAB.png")
plt.show() 