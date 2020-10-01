#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:10:02 2020

@author: aristizabal
"""

#%%

cycle = '2019082800'

file_shear = 'dorian05l.2019082800.hwrfprs.storm.0p015.f003_shear.nc'

folder_pom19 =  '/scratch2/NOS/nosofs/Maria.Aristizabal/HWRF2019_POM_Dorian/'
folder_pom_oper = folder_pom19 + 'HWRF2019_POM_dorian05l.' + cycle + '_oper/'
hwrf_pom_track_oper = folder_pom_oper + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'

folder_hwrf19_pom_oper = folder_pom19 + 'HWRF2019_POM_dorian05l.' + cycle + '_grb2_to_nc_oper/'

#%%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#from datetime import datetime, timedelta
#import matplotlib.dates as mdates
import os
import glob

#%% Get storm track from HWRF/POM output

def get_storm_track_POM(file_track):

    ff = open(file_track,'r')
    f = ff.readlines()
    
    latt = []
    lont = []
    lead_time = []
    for l in f:
        lat = float(l.split(',')[6][0:4])/10
        if l.split(',')[6][4] == 'N':
            lat = lat
        else:
            lat = -lat
        lon = float(l.split(',')[7][0:5])/10
        if l.split(',')[7][4] == 'E':
            lon = lon
        else:
            lon = -lon
        latt.append(lat)
        lont.append(lon)
        lead_time.append(int(l.split(',')[5][1:4]))
    
    latt = np.asarray(latt)
    lont = np.asarray(lont)
    lead_time, ind = np.unique(lead_time,return_index=True)
    lat_track = latt[ind]
    lon_track = lont[ind]  

    return lon_track, lat_track, lead_time

#%% Get Dorian track from models

lon_forec_track_pom_oper, lat_forec_track_pom_oper, lead_time_pom_oper = get_storm_track_POM(hwrf_pom_track_oper)

#%% Get list HWRF files

HWRF_POM_oper = sorted(glob.glob(os.path.join(folder_hwrf19_pom_oper,'*shear.nc')))

#%%

shear_mag = np.empty((len(HWRF_POM_oper)))
shear_mag[:] = np.nan

for f,file_shear in enumerate(HWRF_POM_oper[12:]):
    
    print(file_shear)
    HWRF = xr.open_dataset(file_shear)
    
    lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
    lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
    time_hwrf = np.asarray(HWRF.variables['time'][:])
    UGRD850_hwrf = np.asarray(HWRF.variables['UGRD_850mb'][0,:,:])
    VGRD850_hwrf = np.asarray(HWRF.variables['VGRD_850mb'][0,:,:])
    UGRD200_hwrf = np.asarray(HWRF.variables['UGRD_200mb'][0,:,:])
    VGRD200_hwrf = np.asarray(HWRF.variables['VGRD_200mb'][0,:,:])
    
    Vwind200 = np.sqrt(UGRD200_hwrf**2 + VGRD200_hwrf**2)
    Vwind850 = np.sqrt(UGRD850_hwrf**2 + VGRD850_hwrf**2)
    cos200 = np.divide(UGRD200_hwrf,Vwind200)
    sin200 = np.divide(VGRD200_hwrf,Vwind200)
    cos850 = np.divide(UGRD850_hwrf,Vwind850)
    sin850 = np.divide(VGRD850_hwrf,Vwind850)
    
    shear = np.sqrt((Vwind200*cos200-Vwind850*cos850)**2 + \
                        (Vwind200*sin200-Vwind850*sin850)**2)
        
    oklon = int(np.round(np.interp(lon_forec_track_pom_oper[1],lon_hwrf,np.arange(len(lon_hwrf)))))
    oklat = int(np.round(np.interp(lat_forec_track_pom_oper[1],lat_hwrf,np.arange(len(lat_hwrf)))))
    
    shear_mag[f] = shear[oklat,oklon]

#%%




plt.figure()
plt.contourf(lon_hwrf,lat_hwrf,cos200)
plt.plot(lon_forec_track_pom_oper[1],lat_forec_track_pom_oper[1],'*k')
plt.colorbar()

plt.figure()
plt.contourf(lon_hwrf,lat_hwrf,Vwind850)
plt.plot(lon_forec_track_pom_oper[1],lat_forec_track_pom_oper[1],'*k')
plt.colorbar()

plt.figure()
plt.contourf(lon_hwrf,lat_hwrf,shear_mag)
plt.plot(lon_forec_track_pom_oper[1],lat_forec_track_pom_oper[1],'*k')
plt.colorbar()

