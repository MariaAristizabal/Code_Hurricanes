#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:41:17 2018

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-100,-80]
lat_lim = [18,30]

# Time bounds
min_time = '2018-10-10T00:00:00Z'
max_time = '2018-10-11T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'

#%% User input
'''
# Gulf of Mexico
lon_lim = [-98,-78];
lat_lim = [18,32];

#Initial and final date

dateini = '2018/10/10/00/00'
dateend = '2018/10/17/00/00'

url = 'https://data.ioos.us/thredds/dodsC/deployments/'

#id_list = ['rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc']


id_list = ['rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc',\
           'rutgers/ng261-20180801T0000/ng261-20180801T0000.nc3.nc',\
           'rutgers/ng257-20180801T0000/ng257-20180801T0000.nc3.nc',\
           'rutgers/ng290-20180701T0000/ng290-20180701T0000.nc3.nc',\
           'rutgers/ng230-20180801T0000/ng230-20180801T0000.nc3.nc',\
           'rutgers/ng279-20180801T0000/ng279-20180801T0000.nc3.nc',\
           'rutgers/ng429-20180701T0000/ng429-20180701T0000.nc3.nc']

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

'''

#%%

import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
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
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()
    
    print(id,df.index[-1])

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

#%% Best Track Michael

lonMc = np.array([-86.9,-86.7,-86.0,\
                  -85.3,-85.4,-85.07,-85.05,\
                  -85.2,-85.7,-86.1,-86.3,\
                  -86.5,-86.6,-86.3,-85.4,\
                  -84.5,-83.2,-81.7,-80.0])

latMc = np.array([18.4,18.7,19.0,\
                  19.7,20.2,20.9,21.7,\
                  22.7,23.6,24.6,25.6,\
                  26.6,27.8,29.0,30.2,\
                  31.5,32.8,34.1,35.6])

tMc = [                   '2018/10/07/06/00','2018/10/07/12/00','2018/10/07/18/00',\
       '2018/10/08/00/00','2018/10/08/06/00','2018/10/08/12/00','2018/10/08/18/00',\
       '2018/10/09/00/00','2018/10/09/06/00','2018/10/09/12/00','2018/10/09/18/00',\
       '2018/10/10/00/00','2018/10/10/06/00','2018/10/10/12/00','2018/10/10/18/00',
       '2018/10/11/00/00','2018/10/11/06/00','2018/10/11/12/00','2018/10/11/18/00']

timeMc = [None]*len(tMc) 
for x in range(len(tMc)):
    timeMc[x] = datetime.strptime(tMc[x], '%Y/%m/%d/%H/%M') # time in time zone  

#%% gliders during hurricane Michael
    
fig, ax = plt.subplots(figsize=(7, 5))
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,)
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.yticks([])
plt.xticks([])
plt.axis([-98,-78,18,32])
       
# Michael track and intensity
plt.plot(lonMc,latMc,'.-',markersize = 10,color = 'k',linewidth=2)
plt.plot(lonMc[0],latMc[0],'o',markersize = 10,\
         color = 'white',markeredgecolor='green',
         markeredgewidth=3,label='Tropical Storm')
plt.plot(lonMc[5],latMc[5],'o',markersize = 10,\
         color = 'yellow',markeredgecolor='yellow',label='Cat 1')
plt.plot(lonMc[9],latMc[9],'o',markersize = 10,\
         color = 'orange',markeredgecolor='orange',label='Cat 2')
plt.plot(lonMc[10],latMc[10],'o',markersize = 10,\
         color = 'red',markeredgecolor='red',label='Cat 3')
plt.plot(lonMc[12],latMc[12],'o',markersize = 10,\
         color = 'purple',markeredgecolor='purple',label='Cat 4')
plt.legend(loc='lower left',fontsize=14)
plt.plot(lonMc[0:5],latMc[0:5],'o',markersize = 10,\
         color = 'white',markeredgecolor='green',markeredgewidth=3)
plt.plot(lonMc[5:9],latMc[5:9],'o',markersize = 10,\
         color = 'yellow',markeredgecolor='yellow')
plt.plot(lonMc[9],latMc[9],'o',markersize = 10,\
         color = 'orange',markeredgecolor='orange')
plt.plot(lonMc[10:12],latMc[10:12],'o',markersize = 10,\
         color = 'red',markeredgecolor='red')
plt.plot(lonMc[12:15],latMc[12:15],'o',markersize = 10,\
         color = 'purple',markeredgecolor='purple')
plt.plot(lonMc[15],latMc[15],'o',markersize = 10,\
         color = 'yellow',markeredgecolor='yellow')
plt.plot(lonMc[16:],latMc[16:],'o',markersize = 10,\
         color = 'white',markeredgecolor='green')

props = dict(boxstyle='square', facecolor='white', alpha=0.5)
for x in range(0, len(tMc)-1, 3):
    plt.text(lonMc[x]+0.6,latMc[x],timeMc[x].strftime('%d, %H:%M'),\
             size = 12,color='k',weight='bold',bbox=props)

for id in gliders[1:-1]:
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(df['longitude (degrees_east)']),\
                np.mean(df['latitude (degrees_north)']),'^',\
                markersize=10,markeredgecolor='k',label=id.split('-')[0])
#plt.legend(loc='upper left',fontsize=14)
    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_GoM_hurric_Michael.png",\
            bbox_inches = 'tight',pad_inches = 0.1)



#%% Reading glider data and plotting lat and lon in the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lon,bath_lat,-bath_elev,cmap=plt.get_cmap('BrBG'))
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
ax.plot(lonMc,latMc,'o-',markersize = 5,label = 'Michael Track',color = 'dimgray')

for x in range(0, len(tMc), 2):
    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = siz)

for l in id_list:
    ncng = Dataset(url + l)
    lat_ng = ncng.variables['latitude'][:]
    lon_ng = ncng.variables['longitude'][:]
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018, 10, 10,0,0,0),time_ng <= datetime.datetime(2018, 10, 11,0,0,0))
    ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10,\
        markeredgecolor='black',markeredgewidth=2,label = ncng.id.split('-')[0]) 
    legend = ax.legend(loc='upper left',fontsize = siz)
    legend.get_frame().set_facecolor('white') 

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Micheal_track.png"\
             ,bbox_inches = 'tight',pad_inches = 0)
plt.show()    
