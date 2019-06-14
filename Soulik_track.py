#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:35:31 2019

@author: aristizabal
"""

#%% User input

# RU22
lon_lim = [120,134]
lat_lim = [30,40]
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc'

# date limits
date_ini = '2018-08-17T00:00:00Z'
date_end = '2018-08-18T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_90.0_0.0_180.0_45.0.nc'


#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import netCDF4
from matplotlib.dates import date2num
from datetime import datetime

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)

timestamp_glider = date2num(time_glider)[0]

tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

okg = np.where(np.logical_and(time_glider.T >= tmin, time_glider.T <= tmax))

timeg = time_glider[0,okg[0]]
timestampg = timestamp_glider[okg[0]]
latg = np.asarray(latglider[0,okg[0]])
long = np.asarray(longlider[0,okg[0]])


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

#%% Typhoon Soulik best track

lonSl = np.array([136.4,135.5,134.5,133.3,132.1,130.9,129.8,128.7,127.6,126.8,\
                  126.2,125.7,125.5,125.7,126.1,127.3])
latSl = np.array([25.8,26.1,26.4,27.0,27.7,28.3,28.9,29.7,30.4,31.2,\
                  31.9,32.6,33.0,33.6,34.3,35.7])
tSl = ['2018/08/20/00/00','2018/08/20/06/00','2018/08/20/12/00','2018/08/20/18/00',\
       '2018/08/21/00/00','2018/08/21/06/00','2018/08/21/12/00','2018/08/21/18/00',\
       '2018/08/22/00/00','2018/08/22/06/00','2018/08/22/12/00','2018/08/22/18/00',\
       '2018/08/23/00/00','2018/08/23/06/00','2018/08/23/12/00','2018/08/23/18/00']

# Convert time to UTC
#pst = pytz.timezone('America/New_York') # time zone
#utc = pytz.UTC 

timeSl = [None]*len(tSl) 
for x in range(len(tSl)):
    timeSl[x] = datetime.strptime(tSl[x], '%Y/%m/%d/%H/%M')

#%% 
    
fig, ax = plt.subplots(figsize=(7, 5))
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.plot(long,latg,'.r')
ax.text(np.mean(long),np.mean(latg)-0.2,'RU22',size=10,fontweight='bold')

plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.yticks([])
plt.xticks([])
plt.axis([lon_lim[0],lon_lim[1],lat_lim[0],lat_lim[1]])


ax.plot(lonSl,latSl,'o-',markersize = 8,label = 'Soulik Track',\
        color = 'dimgray',linewidth=3)

for x in range(8,16):   
    ax.text(lonSl[x]-2.2,latSl[x]-0.3,timeSl[x].strftime('%d, %H:%M'),size = 10,fontweight='bold')
legend = ax.legend(loc='upper right',fontsize = 14)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_Yellow_Sea_SOULIK.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
