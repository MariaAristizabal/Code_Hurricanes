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

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%
from read_glider_data import read_glider_data_thredds_server
from process_glider_data import grid_glider_data_thredd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates   

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
                    
#%%
    
fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

timeg = timeg291
depthg_gridded = depthg_gridded291
varg_gridded = varg_gridded291
inst_id = inst_id291
       
nlevels = np.round(np.nanmax(varg_gridded)) - np.round(np.nanmin(varg_gridded))\
         + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(varg_gridded)),\
                               np.round(np.nanmax(varg_gridded)),nlevels))
plt.contour(timeg,-depthg_gridded,varg_gridded.T,levels=26,colors = 'k')
cs = plt.contourf(timeg,-depthg_gridded,varg_gridded.T,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel(var,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16);                                         

#%%

fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

timeg = timeg487
depthg_gridded = depthg_gridded487
varg_gridded = varg_gridded487
inst_id = inst_id487
       
nlevels = np.round(np.nanmax(varg_gridded)) - np.round(np.nanmin(varg_gridded))\
         + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(varg_gridded)),\
                               np.round(np.nanmax(varg_gridded)),nlevels))
plt.contour(timeg,-depthg_gridded,varg_gridded.T,levels=26,colors = 'k')
cs = plt.contourf(timeg,-depthg_gridded,varg_gridded.T,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel(var,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16);  

#%%

fig, ax=plt.subplots(figsize=(10, 6), facecolor='w', edgecolor='w')

timeg = timeg467
depthg_gridded = depthg_gridded467
varg_gridded = varg_gridded467
inst_id = inst_id467
       
nlevels = np.round(np.nanmax(varg_gridded)) - np.round(np.nanmin(varg_gridded))\
         + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(varg_gridded)),\
                               np.round(np.nanmax(varg_gridded)),nlevels))
plt.contour(timeg,-depthg_gridded,varg_gridded.T,levels=26,colors = 'k')
cs = plt.contourf(timeg,-depthg_gridded,varg_gridded.T,cmap='RdYlBu_r',**kw)
plt.title(inst_id.split('-')[0],fontsize=20)

ax.set_xlim(timeg[0], timeg[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical') 
cbar.ax.set_ylabel(var,fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16);      