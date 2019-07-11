#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:13:37 2019

@author: aristizabal
"""

#%% User input

url_RTOFS = 'https://nomads.ncep.noaa.gov:9090/dods/rtofs/rtofs_global'

#%%

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean

from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import netCDF4

#%% Get time bounds for the previous day

te = datetime.today() 
tend = datetime(te.year,te.month,te.day)

#ti = datetime.today() - timedelta(1)
ti = datetime.today() - timedelta(1)
tini = datetime(ti.year,ti.month,ti.day)

#%% Read RTOFS output

print('Retrieving coordinates from RTOFS')

url_RTOFS_temp = url_RTOFS + tend.strftime('%Y%m%d') + '/rtofs_glo_3dz_nowcast_daily_temp'
url_RTOFS_salt = url_RTOFS + tend.strftime('%Y%m%d') + '/rtofs_glo_3dz_nowcast_daily_salt'
RTOFS_temp = xr.open_dataset(url_RTOFS_temp ,decode_times=False)
RTOFS_salt = xr.open_dataset(url_RTOFS_salt ,decode_times=False)
    
latRTOFS = RTOFS_temp.lat[:]
lonRTOFS = RTOFS_temp.lon[:]
depthRTOFS = RTOFS_temp.lev[:]
ttRTOFS = RTOFS_temp.time[:]
tRTOFS = netCDF4.num2date(ttRTOFS[:],ttRTOFS.units) 
tempRTOFS = RTOFS_temp.temperature[-1,0,:,:]

#%%
fig, ax = plt.subplots(figsize=(9, 3))
cs = plt.contourf(lonRTOFS,latRTOFS,tempRTOFS,cmap=cmocean.cm.thermal)
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
plt.title('Surface Temperature RTOFS ' + date)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'surf_temp_RTOFS_global_' + date 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)