#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:15:25 2019

@author: aristizabal
"""

#%% User input

# ramses
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc'

# date limits
date_ini = '2018-09-08T00:00:00Z'
date_end = '2018-09-18T00:00:00Z'

mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Ramses_GOFS31.mat'

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
ramses = sio.loadmat(mat_file)

ramses.keys()

#%% Downloding data

depthg_vec_ram = ramses['depthg_vec'][0,:]
tempg_matrix_ram = ramses['tempg_matrix'][:]
tstamp_glider_ram = ramses['timeg_vec'][:,0]

depth31 = ramses['depthm'][:,0]
temp31 = ramses['tempm'][:]
salt31 = ramses['salm'][:]
tstamp31 = ramses['timem'][:,0]

#%% Changing timestamps to datenum

timeglid = []
for i in np.arange(len(tstamp_glider_ram)):
    timeglid.append(datetime.fromordinal(int(tstamp_glider_ram[i])) + \
        timedelta(days=tstamp_glider_ram[i]%1) - timedelta(days = 366))
timeglider_ram = np.asarray(timeglid)
    
timeglid = []
for i in np.arange(len(tstamp31)):
    timeglid.append(datetime.fromordinal(int(tstamp31[i])) + \
        timedelta(days=tstamp31[i]%1) - timedelta(days = 366))
time31 = np.asarray(timeglid)

#%% Reading glider data

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
          
#%% ramses contour plot

timegg = timeglider_ram
depthg_gridded = depthg_vec_ram
varg_gridded = tempg_matrix_ram
inst_id = 'Ramses'

nlevels =21
kw = dict(levels = np.linspace(9,29,nlevels))

fig, ax=plt.subplots(figsize=(12, 3))
plt.contour(timegg,-depthg_gridded.T,varg_gridded,[26],colors = 'k')
cs = plt.contourf(timegg,-depthg_gridded.T,varg_gridded,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timegg[0], timegg[-1])
xfmt = mdates.DateFormatter('%d-%b\n %Y')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel('Temperature ($^oC$)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_ylabel('Depth (m)',fontsize=16);
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
#plt.xlim([datetime(2018,7,18),datetime(2018,9,16)])
plt.ylim([-40,0]) 
plt.xticks([datetime(2018,9,8),datetime(2018,9,10),datetime(2018,9,12),\
            datetime(2018,9,14),datetime(2018,9,16)])
plt.yticks([-40,-30,-20,-10,0])                                      

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ramses_temp.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% GOFS 3.1 temperature

nlevels =17
kw = dict(levels = np.linspace(13,29,nlevels))
    
fig, ax=plt.subplots(figsize=(10, 1.7))
plt.contour(time31,-1*depth31.T,temp31,[26],colors = 'k')
cs = plt.contourf(time31,-1*depth31.T,temp31,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018,9,14,00),len(np.arange(-40,0))),np.arange(-40,0),'--k')
plt.title('GOFS 3.1',fontsize=20)
ax.set_xlim(timeg[0], timeg[-1])
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
plt.ylim([-40,0]) 
plt.yticks([-40,-20,0]) 
plt.xticks([datetime(2018,9,8),datetime(2018,9,10),datetime(2018,9,12),\
            datetime(2018,9,14),datetime(2018,9,16),datetime(2018,9,18)])
#plt.yticks([-40,-30,-20,-10,0])                                      

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ramses_GOFS31_temp.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% GOFS 3.1 salnity

nlevels =11
kw = dict(levels = np.linspace(26,36,nlevels))
    
fig, ax=plt.subplots(figsize=(10, 1.7))
cs = plt.contourf(time31,-1*depth31.T,salt31,cmap=cmocean.cm.haline,**kw)
plt.plot(np.tile(datetime(2018,9,14,00),len(np.arange(-40,0))),np.arange(-40,0),'--k')
plt.title('GOFS 3.1',fontsize=20)
ax.set_xlim(timeg[0], timeg[-1])
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
plt.ylim([-40,0]) 
plt.yticks([-40,-20,0]) 
plt.xticks([datetime(2018,9,8),datetime(2018,9,10),datetime(2018,9,12),\
            datetime(2018,9,14),datetime(2018,9,16),datetime(2018,9,18)])
                           

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ramses_GOFS31_salt.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% ramses scatter temperature

timeg_matrix = np.transpose(np.tile(timeg.T,(depthg.shape[1],1)))
ttg = np.ravel(timeg_matrix)
dg = np.ravel(depthg)
teg = np.ravel(tempg)

kw = dict(c=teg, marker='*', edgecolor='none')

fig, ax = plt.subplots(figsize=(10, 1.7))
cs = ax.scatter(ttg,-dg,cmap='RdYlBu_r',**kw)
plt.contour(timegg,-depthg_gridded.T,varg_gridded,[26],colors = 'k')
plt.plot(np.tile(datetime(2018,9,14,00),len(np.arange(-40,0))),np.arange(-40,0),'--k')
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel('Temperature ($^oC$)',fontsize=16)
cbar.set_ticks(np.arange(13,31,2))
cbar.ax.tick_params(labelsize=14)

cbar.set_clim(13,29)
ax.set_title(inst_id.split('-')[0],fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
plt.xticks([datetime(2018,9,8),datetime(2018,9,10),datetime(2018,9,12),\
            datetime(2018,9,14),datetime(2018,9,16),datetime(2018,9,18)])
    
file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ramses_temp_scatter.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% ramses scatter salinity

timeg_matrix = np.transpose(np.tile(timeg.T,(depthg.shape[1],1)))
ttg = np.ravel(timeg_matrix)
dg = np.ravel(depthg)
seg = np.ravel(saltg)

kw = dict(c=seg, marker='*', edgecolor='none')

fig, ax = plt.subplots(figsize=(10, 1.7))
cs = ax.scatter(ttg,-dg,cmap=cmocean.cm.haline,**kw)
plt.plot(np.tile(datetime(2018,9,13,18),len(np.arange(-40,0))),np.arange(-40,0),'--k')
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylabel('Depth (m)',fontsize=16)

cbar = plt.colorbar(cs)
cbar.ax.set_ylabel('Salinity',fontsize=16)
cbar.set_ticks(np.arange(26,38,2))
cbar.ax.tick_params(labelsize=14)
cbar.set_clim(26,36)

ax.set_title(inst_id.split('-')[0],fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
plt.xticks([datetime(2018,9,8),datetime(2018,9,10),datetime(2018,9,12),\
            datetime(2018,9,14),datetime(2018,9,16),datetime(2018,9,18)])
    
file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ramses_salt_scatter.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)     