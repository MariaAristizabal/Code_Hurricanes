#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:53:44 2019

@author: aristizabal
"""

#%% Download kmz files

import requests
import urllib.request
from bs4 import BeautifulSoup
from datetime import datetime,timedelta
import numpy as np
import os
import glob
from zipfile import ZipFile

import matplotlib.pyplot as plt
import xarray as xr

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

#%% Get time bounds for the previous day

ti = datetime.today() - timedelta(1)
tini = datetime(ti.year,ti.month,ti.day)

#%%

url = 'https://www.nhc.noaa.gov/gis/'

r = requests.get(url)
data = r.text

soup = BeautifulSoup(data,"lxml")

#latest_cone = 'https://www.nhc.noaa.gov/storm_graphics/api/AL052019_CONE_latest.kmz'
#latest_track = 'https://www.nhc.noaa.gov/storm_graphics/api/AL052019_TRACK_latest.kmz'
#prel_best_track ='https://www.nhc.noaa.gov/gis/best_track/al052019_best_track.kmz'

for i,s in enumerate(soup.find_all("a")):
    ff = s.get('href')
    if type(ff) == str:
        if np.logical_and('kmz' in ff, str(tini.year) in ff):
            if 'CONE_latest' in ff:
                file_name = ff.split('/')[3] 
                print(ff, file_name) 
                urllib.request.urlretrieve(url[:-4] + ff , file_name)
            if 'TRACK_latest' in ff:
                file_name = ff.split('/')[3]
                print(ff, file_name) 
                urllib.request.urlretrieve(url[:-4] + ff ,file_name)
            if 'best_track' in ff:
                file_name = ff.split('/')[1]
                print(ff,file_name)
                urllib.request.urlretrieve(url + ff ,file_name)

#%%
       
kmz_files = glob.glob('*.kmz')
         
# NOTE: UNTAR  the .kmz FILES AND THEN RUN FOLLOWING CODE
for f in kmz_files: 
    os.system('cp ' + f + ' ' + f[:-3] + 'zip')              
    os.system('tar -xvf ' + f)
    
    
#%% get names zip and kml files

zip_files = glob.glob('*.zip')    
kml_files = glob.glob('*.kml')

#%%             

#filename = '/Users/aristizabal/Desktop/AL052019_TRACK_latest.kmz'
#filename = '/Users/aristizabal/Desktop/AL052019_TRACK_latest.zip'
for i,f in enumerate(zip_files):
    kmz = ZipFile(f, 'r')
    if 'TRACK' in f:
        kml_f = [f for f in kml_files if 'TRACK' in f]
        kml_track = kmz.open(kml_f[0], 'r').read()
    else:
        if 'CONE' in f:
            kml_f = [f for f in kml_files if 'CONE' in f]
            kml_cone = kmz.open(kml_f[0], 'r').read()
        else:
            kml_f = [f for f in kml_files if len(f.split('_')) == 1]
        kml_best_track = kmz.open(kml_f[0], 'r').read()

#%% Get TRACK coordinates
        
filename = '/Users/aristizabal/Desktop/AL052019_TRACK_latest.zip'    
kmz = ZipFile(filename, 'r')       
kml = kmz.open('al052019_025Aadv_TRACK.kml', 'r').read() 

'''
filename = '/Users/aristizabal/Desktop/AL052019_CONE_latest.zip'
kmz = ZipFile(filename, 'r')
kml = kmz.open('al052019_025Aadv_CONE.kml', 'r').read()    
 '''         

soup = BeautifulSoup(kml_track,'lxml')  

lon_forec_track = np.empty(len(soup.find_all("point")))
lon_forec_track[:] = np.nan
lat_forec_track = np.empty(len(soup.find_all("point")))
lat_forec_track[:] = np.nan
for i,s in enumerate(soup.find_all("point")):
    print(s.get_text("coordinates"))
    lon_forec_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
    lat_forec_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])
    
# Get time stamp
    
time = []
for i,s in enumerate(soup.find_all("td")):
    #print(s.get_text(""))
    if len(s.get_text(' ').split('Valid at:'))>1:
        time.append(s.get_text(' ').split('Valid at:')[1])
    
#%% CONE coordinates
'''
filename = '/Users/aristizabal/Desktop/AL052019_CONE_latest.zip'
kmz = ZipFile(filename, 'r')
kml = kmz.open('al052019_025adv_CONE.kml', 'r').read()    
 '''   

soup = BeautifulSoup(kml_cone,'lxml')  

lon_forec_cone = []
lat_forec_cone = []
for i,s in enumerate(soup.find_all("coordinates")):
    #print(s.get_text('coordinates'))
    coor = s.get_text('coordinates').split(',0')
    for st in coor[1:-1]:
        lon_forec_cone.append(st.split(',')[0])
        lat_forec_cone.append(st.split(',')[1])
        
lon_forec_cone = np.asarray(lon_forec_cone).astype(float)
lat_forec_cone = np.asarray(lat_forec_cone).astype(float)

#%% best track coordinates



#%%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(lon_forec_cone,lat_forec_cone,'.b')

        
#%%

# Reading bathymetry data

lon_lim2 = [-100.0,-60.0]
lat_lim2 = [5.0,45.0]

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

# Getting subdomain for plotting glider track on bathymetry
oklatbath = np.logical_and(bath_lat >= lat_lim2[0],bath_lat <= lat_lim2[1])
oklonbath = np.logical_and(bath_lon >= lon_lim2[0],bath_lon <= lon_lim2[1])
        
bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.plot(lon_forec_track, lat_forec_track,'.-k')
plt.plot(lon_forec_cone,lat_forec_cone,'.b')
#plt.plot(lon_forec_cone,lat_forec_cone,'.b')
#for tind,t in enumerate(time):
#    print(tind)
    #plt.text(lon_forec_track[tind], lat_forec_track[tind],'a')
    