#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:47:15 2020

@author: aristizabal
"""

#%% User input

cycle = '20200701'

# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

folder_RTOFS_DA = '/home/aristizabal/RTOFS-DA/rtofs.20200701/'
prefix_RTOFS_DA = 'rtofs_glo_3dz_f'
    
# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

folder_figs = '/home/aristizabal/Figures/'

#%% 
from erddapy import ERDDAP
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from datetime import datetime, timedelta
import os
import os.path
import glob

import sys
sys.path.append('/home/aristizabal/NCEP_scripts/')
#from utils4HYCOM_orig import readBinz 
from utils4HYCOM import readgrids, readdepth, readVar

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading bathymetry data
ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

#%%  Reading RTOFS-DA nc files

ncfiles_RTOFS_DA = sorted(glob.glob(os.path.join(folder_RTOFS_DA,prefix_RTOFS_DA+'*.nc')))

file = ncfiles_RTOFS_DA[0]
ncRTOFS_DA = xr.open_dataset(file)
lonRTOFS_DA = np.asarray(ncRTOFS_DA.Longitude[:])
latRTOFS_DA = np.asarray(ncRTOFS_DA.Latitude[:])

for file in ncfiles_RTOFS_DA:
    print(file)
    ncRTOFS_DA = xr.open_dataset(file)
    depthRTOFS_DA = np.asarray(ncRTOFS_DA.Depth[:])
    timeRTOFS_DA = np.asarray(ncRTOFS_DA.MT[:])
    tempRTOFS_DA = np.asarray(ncRTOFS_DA.temperature[:])

