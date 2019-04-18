#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 12:07:36 2019

@author: aristizabal
"""

import netCDF4
#from netCDF4 import Dataset
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import xarray as xr
#from datetime import datetime
#from datetime import datetime
#import pytz
#import matplotlib.dates as mdates

#%% User input

# Directories where increments files reside 
Dir= '/Volumes/aristizabal/GOFS/seatmp_20181002-20181014/' #Michael period

#%% Reading and plotting ncoda data

ncoda_files = sorted(glob.glob(os.path.join(Dir,'*.nc')))

# increment North Atlantic Surface Florence
    
siz=12

for l in ncoda_files:
    print(l)
    ncncoda = xr.open_dataset(l,decode_times=False) 
    temp_incr = ncncoda.pot_temp[0]

    time_ncoda = ncncoda.MT # Michael
    time_ncoda = np.transpose(netCDF4.num2date(time_ncoda[:],time_ncoda.units))
    depth_ncoda = ncncoda.Depth
    lat_ncoda = ncncoda.Latitude
    lon_ncoda = ncncoda.Longitude

    # Get rid off very high values
    tincr = np.asarray(temp_incr[0,:,:])
    tincr[tincr < -4.0] = np.nan 
    tincr[tincr > 4.0] = np.nan 

    fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
    #ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
    #ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
    cs = ax.contourf(lon_ncoda-360,lat_ncoda,tincr,20,cmap=plt.get_cmap('seismic'), vmin=-4.0, vmax=4.0)
    cbar = plt.colorbar(cs)
    cbar.set_label('($^o$C)',rotation=270,size = 18,labelpad = 20)
    #plt.colorbar()
    #ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
    #plt.axis('equal')
    plt.axis([260-360,350-360,0,50])
    #ax.plot(lonFl[1:-2],latFl[1:-2],'o-',markersize = 5,color = 'dimgray') #label = 'Florence Track',
    plt.title('{0} {1}'.format('Temperature Increments at Surface on',time_ncoda[0]))
    ax.set_facecolor('lightgrey')
    file = '{0} {1}'.format('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_surf_Atlant',\
        time_ncoda[0])
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0) 
    
    
#%%

ncoda_files = sorted(glob.glob(os.path.join(Dir,'*.nc')))

# increment North Atlantic Surface Florence
siz=12
l=ncoda_files[3]
ncncoda = xr.open_dataset(l,decode_times=False) 
temp_incr = ncncoda.pot_temp[0]
#ncncoda = Dataset(l)
#temp_incr = ncncoda.variables['pot_temp'][:]

time_ncoda = ncncoda.MT # Michael
time_ncoda = np.transpose(netCDF4.num2date(time_ncoda[:],time_ncoda.units))
depth_ncoda = ncncoda.Depth
lat_ncoda = ncncoda.Latitude
lon_ncoda = ncncoda.Longitude

# Get rid off very high values
tincr = np.asarray(temp_incr[0,:,:])
tincr[tincr < -4.0] = np.nan 
tincr[tincr > 4.0] = np.nan 

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
cs = ax.contourf(lon_ncoda-360,lat_ncoda,tincr,20,cmap=plt.get_cmap('seismic'), vmin=-4.5, vmax=4.5)
cbar = plt.colorbar(cs)
cbar.set_label('($^o$C)',rotation=270,size = 18, labelpad = 20)
plt.axis([260-360,350-360,0,50])
plt.title('{0} {1}'.format('Temperature Increments at Surface on',time_ncoda[0]))
ax.set_facecolor('lightgrey')
file = '{0} {1}'.format('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_surf_Atlant',\
        time_ncoda[0])
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0)     






