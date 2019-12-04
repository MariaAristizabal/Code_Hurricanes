#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:49:31 2019

@author: root
"""
#%% User input

# files for HWRF-POM
ncfolder = '/Volumes/aristizabal/ncep_model/HWRF-POM_Michael/'
# POM grid file name
grid_file = 'michael14l.2018100718.pom.grid.nc'

# Server erddap url IOOS glider dap
server = 'https://data.ioos.us/gliders/erddap'

#gliders sg666, sg665, sg668, silbo
gdata_ng665 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG665-20190718T1155/SG665-20190718T1155.nc3.nc'

#Time window
date_ini = '2019/08/28/00'
date_end = '2019/09/02/12'

#%%

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import sys

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

from read_glider_data import read_glider_data_thredds_server

#%% Reading glider data
    
url_glider = gdata_ng665
#url_glider = gdata_ng666
#url_glider = gdata_ng668
#url_glider = gdata_silbo

var = 'temperature'

scatter_plot = 'no'
kwargs = dict(date_ini=date_ini,date_end=date_end)

varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
tempg = varg
timestampg = mdates.date2num(timeg)

#%% Reading POM grid files

pom_grid = xr.open_dataset(ncfolder + grid_file)

lonc = np.asarray(pom_grid['east_e'][:])
latc = np.asarray( pom_grid['north_e'][:])
#lonu, latu = pom_grid['east_u'][:], pom_grid['north_u'][:]
#lonv, latv = pom_grid['east_v'][:], pom_grid['north_v'][:]
zlevc = np.asarray(pom_grid['zz'][:])
topoz = np.asarray(pom_grid['h'][:])  

#%% Getting oklonpom and oklatpom from POM grid

# Getting time of track
t0 = datetime.strptime(date_ini,'%Y/%m/%d/%H')
time_pom = [t0 + timedelta(hours=int(hrs)) for hrs in np.arange(0,132,6)]
timestamp_pom = mdates.date2num(time_pom)

oklonpom = []
oklatpom = []
for t,tt in enumerate(time_pom):
    print(t)

    # Interpolating latg and longlider into RTOFS grid
    sublonpom = np.interp(timestamp_pom[t],timestampg,long)
    sublatpom = np.interp(timestamp_pom[t],timestampg,latg)
    oklonpom.append(np.int(np.round(np.interp(sublonpom,lonc[0,:],np.arange(len(lonc[0,:]))))))
    oklatpom.append(np.int(np.round(np.interp(sublatpom,latc[:,0],np.arange(len(latc[:,0]))))))
    print(sublonpom)
    print(sublatpom)
    
oklonpom = np.asarray(oklonpom)
oklatpom = np.asarray(oklatpom)    