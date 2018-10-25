#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:54:23 2018

@author: aristizabal
"""
#%% User input

wrf_folder = '/Volumes/coolgroup/ru-wrf/test_real-time/processed/9km/'

dateini = '2018/09/10/00/00'
dateend = '2018/09/17/00/00'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

# folder to save figures
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Florence/'

#ncfile = '/Volumes/coolgroup/ru-wrf/test_real-time/processed/9km/20180909/wrfproc_9km_20180909_00Z_H000.nc'
#ncfile = '/Volumes/coolgroup/ru-wrf/test_real-time/processed/9km/20180909/wrfproc_9km_20180909_00Z_H120.nc'

#%%

import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import os

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

#%% Reading nc files

all_folders=os.listdir(wrf_folder)

datei = datetime.strptime(dateini, '%Y/%m/%d/%H/%M') #Time already in UTC
datee = datetime.strptime(dateend, '%Y/%m/%d/%H/%M') #Time already in UTC

datei_stamp = time.mktime(datei.timetuple())
datee_stamp = time.mktime(datee.timetuple())
day_in_sec = 3600 *24

ndays = int((datee_stamp -datei_stamp)/(3600*24))

dat_vec = [datei_stamp]
for i in range(ndays):
    dat_vec = dat_vec + [dat_vec[-1] + day_in_sec]

#%%
date = datei_stamp
for i in range(len(all_folders)):  
    #print(i)
    if str.isdigit(all_folders[i]):
        dat = datetime.strptime(all_folders[i], '%Y%m%d')
        dat_stamp = time.mktime(dat.timetuple())
        #print(dat_stamp)
        #matching = [x for x in datt if dat_stamp == int(x)] 
        for x in dat_vec:
            if int(dat_stamp) == int(x):
                print(i)
                ncfile = wrf_folder + all_folders[i] + '/' + 'wrfproc_9km_' + all_folders[i] + '_00Z_H' + '000' + '.nc'
                ncwrf = Dataset(ncfile)
                timwrf = ncwrf.variables['Time_1']
                timwrf = netCDF4.num2date(timwrf[:],timwrf.units) 
                latwrf = ncwrf.variables['XLAT'][:]
                lonwrf = ncwrf.variables['XLONG'][:]
                slpwrf = ncwrf.variables['SLP'][:]
                
                plt.figure()
                plt.contour(bath_lon,bath_lat,bath_elev,colors='k')
                c=plt.contourf(lonwrf[0,:,:],latwrf[0,:,:],slpwrf[0,:,:],50, cmap=plt.cm.Spectral_r)
                cbar = plt.colorbar(c)
                plt.clim(960,1040)
                cbar.set_label('HPa', fontsize=20)
                cbar.set_ticks(np.arange(960,1040,10))
                plt.title(ncwrf.variables['SLP'].description + '  ' + str(datetime.strptime(all_folders[i], '%Y%m%d')),fontsize = 16)
                plt.ylim(31,47)
                plt.xlim(-83,-57)
                plt.savefig(folder + 'ru-wrf_' + all_folders[i] + '_00Z_H' + '000' + '.png')
                