#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 13:50:37 2019

@author: aristizabal
"""

#%% User input

# RU22
lon_lim = [120,134]
lat_lim = [30,40]
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc'

#Time window
date_ini = '2018/08/17/00/00'
date_end = '2018/08/17/00/00'

# url for GOFS 3.1
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_90.0_0.0_180.0_45.0.nc'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
import datetime

#%% Download GOFS 3.1 output

df = xr.open_dataset(catalog31,decode_times=False)

#%%
### Build the bbox for the xy data
oklat31 = np.where(np.logical_and(df.lat >= lat_lim[0], df.lat <= lat_lim[-1]))[0]
oklon31 = np.where(np.logical_and(df.lon >= lon_lim[0], df.lon <= lon_lim[-1]))[0]

#lat31= df.lat[oklat31]
#lon31= df.lon[oklon31]
lat31= df.lat
lon31= df.lon

## Find the dates of import
dini = datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H/%M')
dend = datetime.datetime.strptime(date_end,'%Y/%m/%d/%H/%M')

tt31 = netCDF4.num2date(df.time[:],df.time.units)
oktime31 = np.where(np.logical_and(tt31 >= dini,tt31 <=dend))[0][0]
time31 = tt31[oktime31]

#%% Figure grid

meshlon31 = np.meshgrid(lon31,lat31)
meshlat31 = np.meshgrid(lat31,lon31)
#%%
fig, ax = plt.subplots(figsize=(7,6), dpi=80, facecolor='w', edgecolor='w')
ax.plot(meshlon31[0][oklat31.min():oklat31.max(),oklon31.min():oklon31.max()],\
        meshlat31[0][oklon31.min():oklon31.max(),oklat31.min():oklat31.max()].T,'.k')
plt.xlim(120,121)
plt.ylim(30,31)

