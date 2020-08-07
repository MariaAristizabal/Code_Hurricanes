#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:54:51 2020

@author: aristizabal
"""

#%% Cell #5: Search for glider data sets given a 
#   latitude and longitude box and time window, choose one those data sets 
#   (glider_id), plot a scatter plot of the chosen glider transect, grid 
#   and plot a contour plot of the chosen glider transect 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean
from datetime import datetime, timedelta

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')
       
from read_glider_data import retrieve_dataset_id_erddap_server
from read_glider_data import read_glider_data_erddap_server
from process_glider_data import grid_glider_data

# Server location
url_erddap = 'https://data.ioos.us/gliders/erddap'

# SAB
lon_lim = [-90.0,-75.0]
lat_lim = [25.0,33.0]

# date limits
date_ini = '2020/07/01/00'
date_end = '2020/08/05/00'

#folder_fig = ''

gliders = retrieve_dataset_id_erddap_server(url_erddap,lat_lim,lon_lim,date_ini,date_end)
print(gliders)

#%% Sam

dataset_id = gliders[4]

kwargs = dict(date_ini=date_ini,date_end=date_end)
scatter_plot = 'no'

tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                   lat_lim,lon_lim,scatter_plot,**kwargs)
    
#%%
contour_plot = 'no' # default value is 'yes'
delta_z = 0.4     # default value is 0.3    
    
tempg_gridded, timegg, depthg_gridded = \
                    grid_glider_data('temperature',dataset_id,tempg,timeg,latg,long,depthg,delta_z,contour_plot)
                    
saltg_gridded, timegg, depthg_gridded = \
                    grid_glider_data('salinity',dataset_id,saltg,timeg,latg,long,depthg,delta_z,contour_plot)
                    
#%%    
# variable to retrieve
var_name = 'temperature'
varg = tempg
     
if var_name == 'temperature':
    color_map = cmocean.cm.thermal
    clabel = '($^oC$)'
else:
    if var_name == 'salinity':
        color_map = cmocean.cm.haline
        clabel = ' '
    else:
        color_map = 'RdBu_r'

timeg_matrix = np.tile(timeg.T,(depthg.shape[0],1))
ttg = np.ravel(timeg_matrix)
dg = np.ravel(depthg)
teg = np.ravel(varg)
tisaias = datetime(2020,8,4,0)

kw = dict(c=teg, marker='*', edgecolor='none')

fig, ax = plt.subplots(figsize=(10, 3))
cs = ax.scatter(ttg,-dg,cmap=color_map,**kw)
plt.contour(timegg,-depthg_gridded,tempg_gridded,levels=[26],colors = 'k')
plt.plot(np.tile(tisaias,len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylabel('Depth (m)',fontsize=14)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel(clabel,fontsize=14)
ax.set_title('Temperature Transect ' + dataset_id,fontsize=16)

ti = datetime(2020,7,17)
xvec = [ti + dt*timedelta(2) for dt in np.arange(10)]
plt.xticks(xvec,fontsize=12)
xfmt = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.ylim([-np.nanmax(dg),0])

#%%   
var_name = 'salinity'    
varg = saltg 

if var_name == 'temperature':
    color_map = cmocean.cm.thermal
    clabel = '($^oC$)'
else:
    if var_name == 'salinity':
        color_map = cmocean.cm.haline
        clabel = ' '
    else:
        color_map = 'RdBu_r'

timeg_matrix = np.tile(timeg.T,(depthg.shape[0],1))
ttg = np.ravel(timeg_matrix)
dg = np.ravel(depthg)
teg = np.ravel(varg)
tteg = np.copy(teg)
tteg[tteg>36.7] = np.nan

kw = dict(c=tteg, marker='*', edgecolor='none')

fig, ax = plt.subplots(figsize=(10, 3))
cs = ax.scatter(ttg,-dg,cmap=color_map,**kw)
plt.contour(timegg,-depthg_gridded,saltg_gridded,levels=[30],colors = 'k')
plt.plot(np.tile(tisaias,len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylabel('Depth (m)',fontsize=14)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel(clabel,fontsize=14)
ax.set_title('Salinity Transect ' + dataset_id,fontsize=16)

ti = datetime(2020,7,17)
xvec = [ti + dt*timedelta(2) for dt in np.arange(10)]
plt.xticks(xvec,fontsize=12)
xfmt = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.ylim([-np.nanmax(dg),0])                       