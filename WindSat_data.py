#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:02:15 2019

@author: aristizabal
"""

#%% User input

WindSat_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/GDS2/L3U/WindSat/REMSS/v7.0.1a/'

#uu = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L3U/WindSat/REMSS/v7.0.1a/2017/001/20170101000000-REMSS-L3U_GHRSST-SSTsubskin-WSAT-wsat_20170101v7.0.1-v02.0-fv01.0.nc';

#WindSat_url = 'https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/WindSat-REMSS-L3U-v7.0.1a.nc'

date_ini = '2018-07-17T00:00:00Z'
date_end = '2018-09-18T00:00:00Z'

lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

#%%

import matplotlib.pyplot as plt
import xarray as xr
import netCDF4
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

#%% Download WindSat data

2018/001/20180101000000-REMSS-L3U_GHRSST-SSTsubskin-WSAT-wsat_20180101v7.0.1-v02.0-fv01.0.nc'

WindSat = xr.open_dataset(WindSat_url)
#%%
WS_time = np.asarray(WindSat.variables['time'][:])
WS_lat = np.asarray(WindSat.variables['lat'][:])
WS_lon = np.asarray(WindSat.variables['lon'][:])

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

ok_time = np.where(np.logical_and(mdates.date2num(WS_time) >= mdates.date2num(tini),\
                         mdates.date2num(WS_time) <= mdates.date2num(tend)))

ok_lon = np.where(np.logical_and(WS_lon >= lon_lim[0],WS_lon <= lon_lim[1]))
ok_lat = np.where(np.logical_and(WS_lat >= lat_lim[0],WS_lat <= lat_lim[1]))

#%%

WS_sst = np.asarray(WindSat.variables['sea_surface_temperature'][ok_time[0][0],ok_lat[0],ok_lon[0]])
