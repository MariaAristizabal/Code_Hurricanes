#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:43:16 2020

@author: aristizabal
"""

#%%
url_GOFS1 = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'
url_GOFS2 = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest'

import xarray as xr
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

GOFS31_1 = xr.open_dataset(url_GOFS1,decode_times=False)
latGOFS_1 = np.asarray(GOFS31_1.lat[:])
lonGOFS_1 = np.asarray(GOFS31_1.lon[:])
depthGOFS_1 = np.asarray(GOFS31_1.depth[:])
ttGOFS_1 = GOFS31_1['time']
tGOFS_1 = netCDF4.num2date(ttGOFS_1[:],ttGOFS_1.units) 

GOFS31_2 = xr.open_dataset(url_GOFS2,decode_times=False)
ttGOFS_2 = GOFS31_2['time']
tGOFS_2 = netCDF4.num2date(ttGOFS_2[:],ttGOFS_2.units) 
latGOFS_2 = np.asarray(GOFS31_2.lat[:])
lonGOFS_2 = np.asarray(GOFS31_2.lon[:])
depthGOFS_2 = np.asarray(GOFS31_2.depth[:])

#%%

temp_GOFS1 = np.asarray(GOFS31_1['water_temp'][4829,:,2977,3576])
temp_GOFS2 = np.asarray(GOFS31_1['water_temp'][47,:,2977,3576])

temp_GOFS1 = np.asarray(GOFS31_1['water_temp'][4829,:,2505,3663])
temp_GOFS2 = np.asarray(GOFS31_1['water_temp'][47,:,2505,3663])


#%%
plt.figure()
plt.plot(temp_GOFS1,-depthGOFS_1,'.-')
plt.plot(temp_GOFS2,-depthGOFS_2,'.-')
plt.ylim(-200,0)