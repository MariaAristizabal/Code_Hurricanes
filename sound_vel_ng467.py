#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:43:23 2019

@author: aristizabal
"""

#%% User input

mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/nc467_Aug_01_Aug_09_2018.mat'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

url_glider = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc'

# Initial and final time for glider profile
date_ini = '2018/08/01/00' # year/month/day/hour
date_end = '2018/08/09/00' # year/month/day/hour

#%%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import xarray as xr
import netCDF4

import scipy.io as sio
ng467 = sio.loadmat(mat_file)

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')
from process_glider_data import grid_glider_data_thredd

import seawater as sw

#%% Read glider data

gdata = xr.open_dataset(url_glider,decode_times=False)
    
inst_id = gdata.id.split('_')[0]

temperature =gdata.temperature[0][:]
salinity =gdata.salinity[0][:]
latitude = gdata.latitude[0]
longitude = gdata.longitude[0]
depth = gdata.depth[0]
pressure = gdata.pressure[0]
    
time = gdata.time[0]
time = netCDF4.num2date(time,time.units)
    
# Find time window of interest    
if date_ini=='':
    tti = time[0]
else:
    tti = datetime.strptime(date_ini,'%Y/%m/%d/%H')
        
if date_end=='':
    tte = time[-1]
else:
    tte = datetime.strptime(date_end,'%Y/%m/%d/%H')
        
oktimeg = np.logical_and(time >= tti,time <= tte)
        
# Fiels within time window
tempg =  temperature[oktimeg,:]
saltg =  salinity[oktimeg,:]
latg = latitude[oktimeg]
long = longitude[oktimeg]
depthg = depth[oktimeg,:]
pressg = pressure[oktimeg,:]
timeg = time[oktimeg]

#%% Read model output

time_matlab31 =  ng467['timem'][:,0]
depth31 = ng467['depthm'][:,0]
temp31 = ng467['temp31'][:]
salt31 = ng467['salt31'][:]

time31 = [] 
for i in np.arange(len(time_matlab31)):
    print(i)
    time31.append(datetime.fromordinal(int(time_matlab31[i])) + \
        timedelta(days=time_matlab31[i]%1) - timedelta(days = 366))      
        
time31 = np.asarray(time31)
timestamp31 = mdates.date2num(time31)

#%%
var_name = 'Temperature'

depthg_gridded, tempg_gridded, timegg = \
grid_glider_data_thredd(timeg,latg,long,depthg,tempg,var_name,inst_id)

#%%
var_name = 'Salinity'

depthg_gridded, saltg_gridded, timegg = \
grid_glider_data_thredd(timeg,latg,long,depthg,saltg,var_name,inst_id)

#%% Calculate sound velocity glider

svelg = sw.svel(saltg,tempg,pressg)
var_name = 'Sound Velocity'

depthg_gridded, svelg_gridded, timegg = \
grid_glider_data_thredd(timeg,latg,long,depthg,svelg,var_name,inst_id,\
                        contour_plot='no')

#%% Calculate sound velocity GOFS 3.1 

svel31 = sw.svel(salt31,temp31,np.tile(depth31,(len(timestamp31),1)).T)

#%% Fig temperature glider
var_name = 'Temperature'

fig, ax=plt.subplots(figsize=(10, 3), facecolor='w', edgecolor='w')

nlevels = np.round(np.nanmax(tempg_gridded)) - np.round(np.nanmin(tempg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(tempg_gridded)),\
                               np.round(np.nanmax(tempg_gridded)),nlevels))
#plt.contour(timeg,-depthg_gridded,varg_gridded.T,levels=26,colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded,tempg_gridded.T,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(tti, tte)
#ax.set_ylim(-50, 0)

xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16); 

file = folder + inst_id.split('-')[0] + '-' + var_name + str(tti) + '-' + str(tte) + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Fig salinity glider
var_name = 'Salinity'

fig, ax=plt.subplots(figsize=(10, 3), facecolor='w', edgecolor='w')

#nlevels = np.round(np.nanmax(varg_gridded)) - np.round(np.nanmin(varg_gridded)) + 1
#kw = dict(levels = np.linspace(np.round(np.nanmin(varg_gridded)),\
#                               np.round(np.nanmax(varg_gridded)),nlevels))
kw = dict(levels = np.linspace(34.8,37.6,15))
cs = plt.contourf(timegg,-depthg_gridded,saltg_gridded.T,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(tti, tte)
#ax.set_ylim(-50, 0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16); 

file = folder + inst_id.split('-')[0] + '-' + var_name + str(tti) + '-' + str(tte) + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure sound velocity glider
var_name = 'Sound Velocity'

fig, ax=plt.subplots(figsize=(10, 3), facecolor='w', edgecolor='w')

nlevels = np.round(np.nanmax(svelg_gridded)) - np.round(np.nanmin(svelg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(svelg_gridded)),\
                               np.round(np.nanmax(svelg_gridded)),nlevels))
plt.contour(timeg,-depthg_gridded,varg_gridded.T,levels=26,colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded,svelg_gridded.T,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(tti, tte)
#ax.set_ylim(-50, 0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16); 

file = folder + inst_id.split('-')[0] + '-' + var_name + str(tti) + '-' + str(tte) + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure sound velocity GOFS 3.1
var_name = 'Sound Velocity'

fig, ax=plt.subplots(figsize=(10, 3), facecolor='w', edgecolor='w')

#nlevels = np.round(np.nanmax(svelg_gridded)) - np.round(np.nanmin(svelg_gridded)) + 1
#kw = dict(levels = np.linspace(np.round(np.nanmin(svelg_gridded)),\
#                               np.round(np.nanmax(svelg_gridded)),nlevels))

nlevels = np.round(1544 - 1529) + 1
kw = dict(levels = np.linspace(1529,1544,nlevels))
cs = plt.contourf(time31,-1*depth31,svel31,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(tti, tte)
ax.set_ylim(-190, 0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel(var_name,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16); 
file = folder + 'GOFS3.1-' + inst_id.split('-')[0] + '-' + var_name + str(tti) + '-' + str(tte) + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 