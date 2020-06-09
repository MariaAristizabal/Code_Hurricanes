#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:56:19 2020

@author: aristizabal
"""
#%%

#url_thredds  = '/Users/Aristizabal/Desktop/SG665-20190718T1155.nc3.nc'
url_thredds  = '/Users/Aristizabal/Desktop/ru29-20191010T1932.nc3.nc'

#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#%%
gdata = xr.open_dataset(url_thredds,decode_times=False)

temp_qc = np.asarray(gdata.temperature_qc[:])[0,:,:]
time_qc = np.asarray(gdata.temperature_qc.time)[0,:]
depth_qc = np.asarray(gdata.temperature_qc.depth)[0,:,:]

matrix_time_qc = np.tile(time_qc,(temp_qc.shape[1],1)).T

ttg = np.ravel(matrix_time_qc)
dg = np.ravel(depth_qc)
teg = np.ravel(temp_qc)

kw = dict(c=teg, marker='o', s=5, edgecolor='none')

fig, ax = plt.subplots(figsize=(10, 3))
cs = ax.scatter(ttg,-dg,cmap='Reds',**kw)
#fig.colorbar(cs)
#ax.set_xlim(timeg[0], timeg[-1])

ax.set_ylabel('Depth (m)',fontsize=14)
cbar = plt.colorbar(cs)
#cbar.ax.set_ylabel(clabel,fontsize=14)
#ax.set_title(dataset_id,fontsize=16)
#xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
#ax.xaxis.set_major_formatter(xfmt)
plt.ylim([-np.nanmax(dg),0])

#%%

gdata = xr.open_dataset(url_thredds,decode_times=False)

var_qc = np.asarray(gdata.qartod_salinity_spike_flag)[0,:,:]
time_qc = np.asarray(gdata.qartod_salinity_spike_flag.time)[0,:]
depth_qc = np.asarray(gdata.qartod_salinity_spike_flag.depth)[0,:,:]

matrix_time_qc = np.tile(time_qc,(var_qc.shape[1],1)).T

ttg = np.ravel(matrix_time_qc)
dg = np.ravel(depth_qc)
teg = np.ravel(var_qc)

kw = dict(c=teg, marker='o', s=5, edgecolor='none')

fig, ax = plt.subplots(figsize=(10, 3))
cs = ax.scatter(ttg,-dg,**kw)
plt.title(gdata.qartod_salinity_spike_flag.long_name)
#fig.colorbar(cs)
#ax.set_xlim(timeg[0], timeg[-1])

ax.set_ylabel('Depth (m)',fontsize=14)
cbar = plt.colorbar(cs)
#cbar.ax.set_ylabel(clabel,fontsize=14)
#ax.set_title(dataset_id,fontsize=16)
#xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
#ax.xaxis.set_major_formatter(xfmt)
plt.ylim([-np.nanmax(dg),0])

