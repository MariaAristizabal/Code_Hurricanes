#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:09:09 2018

@author: aristizabal
"""

#%% User input

# SAB + MAB
lon_lim = [-81,-70]
lat_lim = [30,42]

#Initial and final date

dateini = '2018/09/10/00/00'
dateend = '2018/09/17/00/00'

# Soulik
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc'

#%%

import netCDF4
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import datetime
import pytz
import numpy as np

#%% Reading glider data

ncglider = Dataset(gdata)
latg = ncglider.variables['latitude'][:]
long = ncglider.variables['longitude'][:]
presg = ncglider.variables['pressure'][:]
tempg = ncglider.variables['temperature'][:,:,:]
timeg = ncglider.variables['time']
timeg = netCDF4.num2date(timeg[:],timeg.units)

#%% Figure

t = np.T(np.matlib.repmat(timeg,presg.shape[2],1))
tt = tt.reshape((1,t.shape[0]*t.shape[1]))
PP = presg.reshape(1,tt.shape[0]*tt.shape[1])
TT = tempg.reshape(1,tt.shape[0]*tt.shape[1])

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.pcolor(timeg,presg, tempg, cmap='RdYlBu_r')

#s = ax.scatter(timeg,presg, tempg, cmap='RdYlBu_r')
#%%
ax = plt.subplot()
ax.plot(presg[0,0,:],'.')