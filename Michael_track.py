#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:41:17 2018

@author: aristizabal
"""

#%% User input

# Gulf of Mexico
lon_lim = [-98,-78];
lat_lim = [18,32];

#Initial and final date

dateini = '2018/10/10/00/00'
dateend = '2018/10/17/00/00'

url = 'https://data.ioos.us/thredds/dodsC/deployments/'

id_list = ['rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc',\
           'rutgers/ng261-20180801T0000/ng261-20180801T0000.nc3.nc',\
           'rutgers/ng257-20180801T0000/ng257-20180801T0000.nc3.nc',\
           'rutgers/ng290-20180701T0000/ng290-20180701T0000.nc3.nc',\
           'rutgers/ng230-20180801T0000/ng230-20180801T0000.nc3.nc',\
           'rutgers/ng279-20180801T0000/ng279-20180801T0000.nc3.nc',\
           'rutgers/ng429-20180701T0000/ng429-20180701T0000.nc3.nc']

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'


#%%

import netCDF4
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import datetime
import pytz
import numpy as np
#import xarray as xr 

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

#%% Tentative Michael path

lonMc = np.array([-84.9,-85.2,-85.3,-85.9,-86.2,-86.4,-86.5,-86.5,-86.3,-86.2,\
                  -86.0,-85.8,-85.5,-85.2,-84.9,-84.5,-84.1])
latMc = np.array([21.2,22.2,23.2,24.1,25.0,26.0,27.1,28.3,28.8,29.1,29.4,29.6,\
                  30.0,30.6,31.1,31.5,31.9])
tMc = ['2018/10/08/15/00','2018/10/08/21/00','2018/10/09/03/00','2018/10/09/09/00',\
       '2018/10/09/15/00','2018/10/09/21/00','2018/10/10/03/00','2018/10/10/09/00',\
       '2018/10/10/11/00','2018/10/10/13/00','2018/10/10/15/00','2018/10/10/16/00',\
       '2018/10/10/18/00','2018/10/10/20/00','2018/10/10/22/00','2018/10/11/00/00',\
       '2018/10/11/02/00']

# Convert time to UTC
#pst = pytz.timezone('America/New_York') # time zone
#utc = pytz.UTC 

timeMc = [None]*len(tMc) 
for x in range(len(tMc)):
    timeMc[x] = datetime.datetime.strptime(tMc[x], '%Y/%m/%d/%H/%M') # time in time zone
    #d = pst.localize(d) # add time zone to date
    #d = d.astimezone(utc)
    #timeMc[x] = d.astimezone(utc)
    #print(timeMc[x].strftime('%d, %H %M'))

#%% Reading glider data and plotting lat and lon in the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
ax.plot(lonMc,latMc,'o-',markersize = 10,label = 'Michael Track',color = 'indianred')

for x in range(0, len(tMc), 2):
    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = siz)

for l in id_list:
    ncng = Dataset(url + l)
    lat_ng = ncng.variables['latitude'][:]
    lon_ng = ncng.variables['longitude'][:]
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018, 10, 10,0,0,0),time_ng >= datetime.datetime(2018, 10, 11,0,0,0))
    ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'*',markersize = 10, label = ncng.id.split('-')[0]) 
    ax.legend(loc='upper left',fontsize = siz)
    
#ax.text(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),ncng.id.split('-')[0],size = 12)   
    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Micheal_track.png")
plt.show()    
