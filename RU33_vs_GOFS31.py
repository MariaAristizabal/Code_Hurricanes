#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:52:08 2019

@author: aristizabal
"""

#%% User input

# ru33
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc'

# date limits
date_ini = '2018-08-01T00:00:00Z'
date_end = '2018-08-09T00:00:00Z'

mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Ru33_08-01_08_09.mat'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta  
import xarray as xr
import netCDF4
from matplotlib.dates import date2num 
import cmocean

import scipy.io as sio
ru33 = sio.loadmat(mat_file)

ru33.keys()

#%% Downloding data

depthg_vec_ru33 = ru33['depthg_vec'][0,:]
tempg_matrix_ru33 = ru33['tempg_matrix'][:]
saltg_matrix_ru33 = ru33['saltg_matrix'][:]
tstamp_glider_ru33 = ru33['timeg_vec'][:,0]

depth31 = ru33['depthm'][:,0]
temp31 = ru33['temp31'][:]
salt31 = ru33['salt31'][:]
tstamp31 = ru33['timem'][:,0]

#%% Changing timestamps to datenum

timeglid = []
for i in np.arange(len(tstamp_glider_ru33)):
    timeglid.append(datetime.fromordinal(int(tstamp_glider_ru33[i])) + \
        timedelta(days=tstamp_glider_ru33[i]%1) - timedelta(days = 366))
timeglider_ru33 = np.asarray(timeglid)
    
timeglid = []
for i in np.arange(len(tstamp31)):
    timeglid.append(datetime.fromordinal(int(tstamp31[i])) + \
        timedelta(days=tstamp31[i]%1) - timedelta(days = 366))
time31 = np.asarray(timeglid)


#%% Reading glider data
'''
ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)
tempglider = np.array(ncglider.temperature[0,:,:])
saltglider = np.array(ncglider.salinity[0,:,:])
depthglider = np.array(ncglider.depth[0,:,:])

timestamp_glider = date2num(time_glider)[0]

#%%
tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

okg = np.where(np.logical_and(time_glider.T >= tmin, time_glider.T <= tmax))

timeg = time_glider[0,okg[0]]
timestampg = timestamp_glider[okg[0]]
latg = np.asarray(latglider[0,okg[0]])
long = np.asarray(longlider[0,okg[0]])
depthg = depthglider[okg[0],:]
tempg = tempglider[okg[0],:]
saltg = saltglider[okg[0],:]
  '''        
#%% RU33 temperature contour plot

timegg = timeglider_ru33
depthg_gridded = depthg_vec_ru33
varg_gridded = tempg_matrix_ru33
inst_id = 'RU33'

nlevels =21
kw = dict(levels = np.linspace(7,27,nlevels))

fig, ax=plt.subplots(figsize=(10,1.7))
plt.contour(timegg,-depthg_gridded.T,varg_gridded,[26],colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded.T,varg_gridded,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timegg[0], timegg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel('Temperature ($^oC$)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_ylabel('Depth (m)',fontsize=16);
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
#plt.xlim([datetime(2018,7,18),datetime(2018,9,16)])
plt.ylim([-50,0]) 
#plt.xticks([datetime(2018,9,8),datetime(2018,9,10),datetime(2018,9,12),\
#            datetime(2018,9,14),datetime(2018,9,16)])
plt.yticks([-50,-30,-10])                                      

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU33_temp.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% RU33 salinity contour plot

timegg = timeglider_ru33
depthg_gridded = depthg_vec_ru33
varg_gridded = saltg_matrix_ru33
inst_id = 'RU33'

nlevels =7
kw = dict(levels = np.linspace(30,33,nlevels))

fig, ax=plt.subplots(figsize=(10,1.7))
cs = plt.contourf(timegg,-depthg_gridded.T,varg_gridded,cmap=cmocean.cm.haline,**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timegg[0], timegg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel('Salinity',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_ylabel('Depth (m)',fontsize=16);
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
#plt.xlim([datetime(2018,7,18),datetime(2018,9,16)])
plt.ylim([-50,0]) 
#plt.xticks([datetime(2018,9,8),datetime(2018,9,10),datetime(2018,9,12),\
#            datetime(2018,9,14),datetime(2018,9,16)])
plt.yticks([-50,-30,-10])                                      

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU33_salt.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% GOFS 3.1 temperature

nlevels =21
kw = dict(levels = np.linspace(7,27,nlevels))
    
fig, ax=plt.subplots(figsize=(10, 1.7))
plt.contour(time31,-1*depth31.T,temp31,[26],colors = 'k')
cs = plt.contourf(time31,-1*depth31.T,temp31,cmap='RdYlBu_r',**kw)
plt.title('GOFS 3.1',fontsize=20)
#ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%d-%b\n %Y')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel('Temperature ($^oC$)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_ylabel('Depth (m)',fontsize=16);
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.ylim([-50,0]) 
plt.yticks([-50,-30,-10]) 
#plt.xticks([datetime(2018,9,8),datetime(2018,9,10),datetime(2018,9,12),\
#            datetime(2018,9,14),datetime(2018,9,16),datetime(2018,9,18)])                                   

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ru33_GOFS31_temp.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% GOFS 3.1 salinity

nlevels =7
kw = dict(levels = np.linspace(30,33,nlevels))
    
fig, ax=plt.subplots(figsize=(10, 1.7))
cs = plt.contourf(time31,-1*depth31.T,salt31,cmap=cmocean.cm.haline,**kw)
plt.title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%d-%b\n %Y')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel('Salinity',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_ylabel('Depth (m)',fontsize=16);
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.ylim([-50,0]) 
plt.yticks([-50,-30,-10]) 
#plt.xticks([datetime(2018,9,8),datetime(2018,9,10),datetime(2018,9,12),\
#            datetime(2018,9,14),datetime(2018,9,16),datetime(2018,9,18)])
                           

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ru33_GOFS31_salt.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
