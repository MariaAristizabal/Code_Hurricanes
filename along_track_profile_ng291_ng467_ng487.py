#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:48:04 2019

@author: aristizabal
"""
#%% User input

url_glider291 = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng291-20180701T0000/ng291-20180701T0000.nc3.nc'
url_glider467 = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc'
url_glider487 = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng487-20180701T0000/ng487-20180701T0000.nc3.nc'

# Inputs
var = 'temperature'
date_ini = '2018/07/17/00' # year/month/day/hour
date_end = '2018/09/17/00' # year/month/day/hour
scatter_plot = 'no'
contour_plot='no'

mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/along_track_temp_prof_ng291_ng467_ng487.mat'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta   

import scipy.io as sio
ngVI = sio.loadmat(mat_file)

#%% Downloding data

depthg_vec291 = ngVI['depthg_vec291'][0,:]
tempg_matrix291 = ngVI['tempg_matrix291'][:]
tstamp_glider291 = ngVI['timeg_vec291'][:,0]

depthg_vec487 = ngVI['depthg_vec487'][0,:]
tempg_matrix487 = ngVI['tempg_matrix487'][:]
tstamp_glider487 = ngVI['timeg_vec487'][:,0]

depthg_vec467 = ngVI['depthg_vec467'][0,:]
tempg_matrix467 = ngVI['tempg_matrix467'][:]
tstamp_glider467 = ngVI['timeg_vec467'][:,0]


#%% Changing timestamps to datenum

timeglid = []
for i in np.arange(len(tstamp_glider291)):
    timeglid.append(datetime.fromordinal(int(tstamp_glider291[i])) + \
        timedelta(days=tstamp_glider291[i]%1) - timedelta(days = 366))
timeglider291 = np.asarray(timeglid)
    
timeglid = []
for i in np.arange(len(tstamp_glider487)):
    timeglid.append(datetime.fromordinal(int(tstamp_glider487[i])) + \
        timedelta(days=tstamp_glider487[i]%1) - timedelta(days = 366))
timeglider487 = np.asarray(timeglid)

timeglid = []
for i in np.arange(len(tstamp_glider467)):
    timeglid.append(datetime.fromordinal(int(tstamp_glider467[i])) + \
        timedelta(days=tstamp_glider467[i]%1) - timedelta(days = 366))    
timeglider467 = np.asarray(timeglid)

#%%
'''
import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python/') 

from read_glider_data import read_glider_data_thredds_server
from process_glider_data import grid_glider_data_thredd

#%% ng291 

kwargs = dict(date_ini=date_ini,date_end=date_end)

varg291, latg291, long291, depthg291, timeg291, inst_id291 = \
             read_glider_data_thredds_server(url_glider291,var,scatter_plot,**kwargs)

depthg_gridded291, varg_gridded291 = \
                    grid_glider_data_thredd(timeg291,latg291,long291,depthg291,varg291,var,inst_id291)
                                  
#%% ng487

scatter_plot = 'no'
kwargs = dict(date_ini=date_ini,date_end=date_end)

varg487, latg487, long487, depthg487, timeg487, inst_id487 = \
             read_glider_data_thredds_server(url_glider487,var,scatter_plot,**kwargs)

contour_plot='no'
depthg_gridded487, varg_gridded487 = \
                    grid_glider_data_thredd(timeg487,latg487,long487,depthg487,varg487,var,inst_id487)
                    
#%% ng467

scatter_plot = 'no'
kwargs = dict(date_ini=date_ini,date_end=date_end)

varg467, latg467, long467, depthg467, timeg467, inst_id467 = \
             read_glider_data_thredds_server(url_glider467,var,scatter_plot,**kwargs)

contour_plot='no'
depthg_gridded467, varg_gridded467 = \
                    grid_glider_data_thredd(timeg467,latg467,long467,depthg467,varg291,var,inst_id291)                    
'''                    
#%% ng291
    
fig, ax=plt.subplots(figsize=(12, 3), facecolor='w', edgecolor='w')

timeg = timeglider291
depthg_gridded = depthg_vec291
varg_gridded = tempg_matrix291
inst_id = 'ng291'
       
nlevels = np.round(np.nanmax(varg_gridded)) - np.round(np.nanmin(varg_gridded))\
         + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(varg_gridded)),\
                               np.round(np.nanmax(varg_gridded)),nlevels))
plt.contour(timeg,-depthg_gridded.T,varg_gridded,[26],colors = 'k')
cs = plt.contourf(timeg,-depthg_gridded.T,varg_gridded,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%d-%b\n %Y')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel('Temperature ($^oC$)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_ylabel('Depth (m)',fontsize=16);
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
plt.xlim([datetime(2018,7,18),datetime(2018,9,16)])
plt.ylim([-200,0]) 
plt.xticks([datetime(2018,7,17),datetime(2018,7,24),datetime(2018,7,31),\
            datetime(2018,8,7),\
            datetime(2018,8,14),datetime(2018,8,21),datetime(2018,8,28),\
            datetime(2018,9,4),datetime(2018,9,11)]) 
plt.yticks([-200,-150,-100,-50,0])                                      

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng291_temp.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% ng287
    
fig, ax=plt.subplots(figsize=(12, 3), facecolor='w', edgecolor='w')

timeg = timeglider487
depthg_gridded = depthg_vec487
varg_gridded = tempg_matrix487
inst_id = 'ng487'
       
nlevels = np.round(np.nanmax(tempg_matrix291)) - np.round(np.nanmin(tempg_matrix291))\
         + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(tempg_matrix291)),\
                               np.round(np.nanmax(tempg_matrix291)),nlevels))
plt.contour(timeg,-depthg_gridded.T,varg_gridded,[26],colors = 'k')
cs = plt.contourf(timeg,-depthg_gridded.T,varg_gridded,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%d-%b\n %Y')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel('Temperature ($^oC$)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_ylabel('Depth (m)',fontsize=16);
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
plt.xlim([datetime(2018,7,18),datetime(2018,9,16)])
plt.ylim([-200,0]) 
plt.xticks([datetime(2018,7,17),datetime(2018,7,24),datetime(2018,7,31),\
            datetime(2018,8,7),\
            datetime(2018,8,14),datetime(2018,8,21),datetime(2018,8,28),\
            datetime(2018,9,4),datetime(2018,9,11)]) 
plt.yticks([-200,-150,-100,-50,0])                                      

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng487_temp.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% ng267
    
fig, ax=plt.subplots(figsize=(12, 3), facecolor='w', edgecolor='w')

timeg = timeglider467
depthg_gridded = depthg_vec467
varg_gridded = tempg_matrix467
inst_id = 'ng467'
       
nlevels = np.round(np.nanmax(tempg_matrix291)) - np.round(np.nanmin(tempg_matrix291))\
         + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(tempg_matrix291)),\
                               np.round(np.nanmax(tempg_matrix291)),nlevels))
plt.contour(timeg,-depthg_gridded.T,varg_gridded,[26],colors = 'k')
cs = plt.contourf(timeg,-depthg_gridded.T,varg_gridded,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%d-%b\n %Y')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel('Temperature ($^oC$)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_ylabel('Depth (m)',fontsize=16);
plt.xticks(fontsize=16)
plt.yticks(fontsize=14) 
plt.xlim([datetime(2018,7,18),datetime(2018,9,16)])
plt.ylim([-200,0]) 
plt.xticks([datetime(2018,7,17),datetime(2018,7,24),datetime(2018,7,31),\
            datetime(2018,8,7),\
            datetime(2018,8,14),datetime(2018,8,21),datetime(2018,8,28),\
            datetime(2018,9,4),datetime(2018,9,11)]) 
plt.yticks([-200,-150,-100,-50,0])                                      

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng467_temp.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 