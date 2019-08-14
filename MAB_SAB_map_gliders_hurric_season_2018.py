#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:31:15 2019

@author: aristizabal
"""

#%% User input

# Caribbean
lon_lim = [-83,-60];
lat_lim = [25,42];

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

gliders_no_silbo = np.concatenate((gliders[0:13],gliders[14:]),axis=0)
for id in gliders_no_silbo:
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
    
#%% Tentative Florence path

lonFl = np.array([-61,-63,-66,-69,-72,-77,-78,-79])
latFl = np.array([25,26,27,29,31,34,36,37])
tFl = ['2018/09/10/17/00','2018/09/11/10/00','2018/09/11/14/00','2018/09/12/02/00',
       '2018/09/12/14/00','2018/09/13/14/00','2018/09/14/14/00','2018/09/15/14/00']

# Convert time to UTC
pst = pytz.timezone('America/New_York') # time zone
utc = pytz.UTC 

timeFl = [None]*len(tFl) 
for x in range(len(tFl)):
    d = datetime.datetime.strptime(tFl[x], '%Y/%m/%d/%H/%M') # time in time zone
    d = pst.localize(d) # add time zone to date
    d = d.astimezone(utc)
    timeFl[x] = d.astimezone(utc)
    print(timeFl[x].strftime('%d, %H %M'))
          

#%% Reading glider data and plotting lat and lon in the map

col = ['red','darkcyan','gold','m','darkorange','crimson','lime',\
       'darkorchid','brown','sienna','yellow','orchid','gray',\
       'gold','sienna','crimson','lime','m','darkcyan','darkorchid',\
       'yellow','orchid','gray']
mark = ['o','P','p','^','D','X','o','P','p','^','D','X',\
        'o','P','p','^','D','X','o','P','p','^','D','X',\
        'o','P','p','^','D','X','o','P','p','^','D','X']

fig, ax = plt.subplots(figsize=(8, 6))
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,cmap='Blues_r')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Gliders Deplyomenst in the MAB and SAB \n during hurricane season 2018 ',size = 18)

gliders_no_silbo = np.concatenate((gliders[0:13],gliders[14:]),axis=0)
for i,id in enumerate(gliders_no_silbo):
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        #plt.plot(df['longitude (degrees_east)'],df['latitude (degrees_north)'],'.',markersize = 2,\
        #       color='black')
        ax.plot(df['longitude (degrees_east)'].mean(),df['latitude (degrees_north)'].mean(),\
                        marker=mark[i],color=col[i],markersize = 10,\
                markeredgecolor='black', markeredgewidth=2,label = id[0:13]) 
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

'''        
ax.plot(lonFl,latFl,'o-',markersize = 5,color = 'dimgray') #label = 'Florence Track',
for x in range(2,len(tFl)-2):
    ax.text(lonFl[x],latFl[x],timeFl[x].strftime('%d, %H:%M'),size = 12)        
'''
   
folder = '/Users/aristizabal/Desktop/my_papers/IEEE_Oceans_2019/Code_and_figures/'
file = 'MAB_SAB_map_gliders_hurric_season_2018'
plt.savefig(folder+file,bbox_inches = 'tight',pad_inches = 0) 