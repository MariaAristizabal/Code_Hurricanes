#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:51:06 2020

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
min_time = '2019-06-01T00:00:00Z'
max_time = '2019-11-30T00:00:00Z'

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/gebco_2020_n50.0_s0.0_w-100.0_e0.0.nc'
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# storm track files
track_folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/'
basin = 'al'
year = '2019'
fname = '_best_track.kmz' 

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import cmocean
import glob
import os
from bs4 import BeautifulSoup
from zipfile import ZipFile
#import cartopy
#import cartopy.feature as cfeature
#from datetime import datetime

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
plt.title('Glider Tracks and Storm Tracks \n Hurricane Season 2019',fontsize=30)


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
  
plt.legend(loc='upper left',bbox_to_anchor=[-0.32,1.0])     
plt.axis('scaled') 
#plt.axis([-100,-50,10,50])
plt.axis([-100,-57,10,50])
plt.savefig("/Users/aristizabal/Desktop/map_gliders_hurric_track_season_2019.png",\
            bbox_inches = 'tight',pad_inches = 0.1)
    