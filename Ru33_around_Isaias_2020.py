#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:02:45 2020

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
from glider_transect_model_com import get_glider_transect_from_GOFS

# Server location
url_erddap = 'https://data.ioos.us/gliders/erddap'
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'

folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%% RU33

# SAB
lon_lim = [-76,-70]
lat_lim = [35,42]

# date limits
date_ini = '2020/07/31/00'
date_end = '2020/08/08/00'

#folder_fig = ''

gliders = retrieve_dataset_id_erddap_server(url_erddap,lat_lim,lon_lim,date_ini,date_end)
print(gliders)


dataset_id = gliders[6]

kwargs = dict(date_ini=date_ini,date_end=date_end)
scatter_plot = 'no'

tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                   lat_lim,lon_lim,scatter_plot,**kwargs)
    
#%%
contour_plot = 'no' # default value is 'yes'
delta_z = 0.4     # default value is 0.3    
    
tempg_gridded, timegg, depthg_gridded = \
                    grid_glider_data('temperature',dataset_id,tempg,timeg,depthg,delta_z,contour_plot)
                    
#saltg_gridded, timegg, depthg_gridded = \
#                    grid_glider_data('salinity',dataset_id,saltg,timeg,depthg,delta_z,contour_plot)

#%% RU33
# variable to retrieve
var_name = 'temperature'
varg = tempg
color_map = cmocean.cm.thermal
clabel = '($^oC$)'

timeg_matrix = np.tile(timeg.T,(depthg.shape[0],1))
ttg = np.ravel(timeg_matrix)
dg = np.ravel(depthg)
teg = np.ravel(varg)
tisaias = datetime(2020,8,4,16)

kw = dict(c=teg, marker='*', edgecolor='none')

fig, ax = plt.subplots(figsize=(10, 3))
cs = ax.scatter(ttg,-dg,cmap=color_map,**kw)
plt.contour(timegg,-depthg_gridded,tempg_gridded,levels=[26],colors = 'k')
plt.plot(np.tile(tisaias,len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylabel('Depth (m)',fontsize=14)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel(clabel,fontsize=14)
#ax.set_title('Temperature Transect ' + dataset_id,fontsize=16)

ti = datetime(2020,7,31)
xvec = [ti + dt*timedelta(2) for dt in np.arange(4)]
plt.xticks(xvec,fontsize=12)
xfmt = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.ylim([-np.nanmax(dg),0])
plt.xlim(datetime(2020,7,31),timeg[-1])

file = folder_fig + 'RU33_temp_transect_around_Isaias_2020_1'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)


#%% RU33 
# variable to retrieve
var_name = 'temperature'
varg = tempg
color_map = cmocean.cm.thermal
clabel = '($^oC$)'

timeg_matrix = np.tile(timeg.T,(depthg.shape[0],1))
ttg = np.ravel(timeg_matrix)
dg = np.ravel(depthg)
teg = np.ravel(varg)
tisaias = datetime(2020,8,4,16)

kw = dict(levels = np.arange(6,31,1))

fig, ax = plt.subplots(figsize=(10, 3))
cs = plt.contourf(timegg,-depthg_gridded,tempg_gridded,cmap=cmocean.cm.thermal,**kw)
plt.contour(timegg,-depthg_gridded,tempg_gridded,levels=[26],colors = 'k')
plt.plot(np.tile(tisaias,len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylabel('Depth (m)',fontsize=14)
cbar = plt.colorbar(cs)
cbar.ax.set_ylabel(clabel,fontsize=14)
ax.set_title('Temperature Transect ' + dataset_id,fontsize=16)

ti = datetime(2020,7,30)
xvec = [ti + dt*timedelta(1) for dt in np.arange(10)]
plt.xticks(xvec,fontsize=12)
xfmt = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.ylim([-np.nanmax(dg),0])
plt.xlim(datetime(2020,7,31),timeg[-1])

file = folder_fig + 'RU33_temp_transect_around_Isaias_2020_2'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Get temperature transect from model RU33
                    
# model variable name
model_name = 'GOFS 3.1'
var_name_model = 'water_temp'
var_name_glider = 'temperature' 
contour_plot = 'yes'                   
                    
temp_GOFS, time_GOFS, depth_GOFS, lat_GOFS, lon_GOFS = \
              get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot)

#%% 
varg = tempg
url_model = url_GOFS 

kw = dict(levels = np.arange(6,31,1))

tisaias = datetime(2020,8,4,16)
fig, ax = plt.subplots(figsize=(10, 3))
cs = plt.contourf(time_GOFS,-depth_GOFS,temp_GOFS,cmap=color_map,**kw)
plt.contour(time_GOFS,-depth_GOFS,temp_GOFS,[26],colors='k')
plt.plot(np.tile(tisaias,len(depthg_gridded)),-depthg_gridded,'--k')
cs = fig.colorbar(cs, orientation='vertical')
cs.ax.set_ylabel(clabel,fontsize=14,labelpad=15)

ti = datetime(2020,7,30)
xvec = [ti + dt*timedelta(1) for dt in np.arange(10)]
plt.xticks(xvec,fontsize=12)
xfmt = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.ylim([-np.nanmax(dg),0])
plt.xlim(datetime(2020,7,31),timeg[-1])
ax.set_ylabel('Depth (m)',fontsize=14)
plt.title('Along Track Temperature Profile ' + model_name,fontsize=16)

file = folder_fig + 'GOFS31_along_track_RU33_around_Isaias_2020'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% SG649

# SAB
lon_lim = [-90,-60]
lat_lim = [10,25]

# date limits
date_ini = '2020/07/01/00'
date_end = '2020/08/07/00'

gliders = retrieve_dataset_id_erddap_server(url_erddap,lat_lim,lon_lim,date_ini,date_end)
print(gliders)

#%%
dataset_id = gliders[7]

kwargs = dict(date_ini=date_ini,date_end=date_end)
scatter_plot = 'no'

tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                   lat_lim,lon_lim,scatter_plot,**kwargs)
    
#%%
contour_plot = 'no' # default value is 'yes'
delta_z = 0.4     # default value is 0.3    
    
tempg_gridded, timegg, depthg_gridded = \
                    grid_glider_data('temperature',dataset_id,tempg,timeg,latg,long,depthg,delta_z,contour_plot)
                    
#saltg_gridded, timegg, depthg_gridded = \
#                    grid_glider_data('salinity',dataset_id,saltg,timeg,latg,long,depthg,delta_z,contour_plot)

#%%    
# variable to retrieve
var_name = 'temperature'
varg = tempg
color_map = cmocean.cm.thermal
clabel = '($^oC$)'

timeg_matrix = np.tile(timeg.T,(depthg.shape[0],1))
ttg = np.ravel(timeg_matrix)
dg = np.ravel(depthg)
teg = np.ravel(varg)
okd = dg <= 500
ddg = dg[okd]
tteg = teg[okd]
ttgg = ttg[okd]
tisaias = datetime(2020,7,30,12)

kw = dict(c=tteg, marker='*', edgecolor='none')

#kw = dict(levels = np.arange(10,31,2))

fig, ax = plt.subplots(figsize=(10, 3))
cs = ax.scatter(ttgg,-ddg,cmap=color_map,**kw)
#cs=plt.contourf(timegg,-depthg_gridded,tempg_gridded,cmap=cmocean.cm.thermal,**kw)
plt.contour(timegg,-depthg_gridded,tempg_gridded,levels=[26],colors = 'k')
#plt.plot(np.tile(tisaias,len(depthg_gridded)),-depthg_gridded,'--k')
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
plt.ylim(-500,0)
ax.set_xlim(timeg[0],datetime(2020,8,5))

file = folder_fig + 'SG649_temp_transect_around_Isaias_2020'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%    
# variable to retrieve
var_name = 'temperature'
varg = tempg
color_map = cmocean.cm.thermal
clabel = '($^oC$)'

timeg_matrix = np.tile(timeg.T,(depthg.shape[0],1))
ttg = np.ravel(timeg_matrix)
dg = np.ravel(depthg)
teg = np.ravel(varg)
okd = dg <= 500
ddg = dg[okd]
tteg = teg[okd]
ttgg = ttg[okd]
tisaias = datetime(2020,7,30,12)

kw = dict(levels = np.arange(10,31,2))

fig, ax = plt.subplots(figsize=(10, 3))
cs=plt.contourf(timegg,-depthg_gridded,tempg_gridded,cmap=cmocean.cm.thermal,**kw)
plt.contour(timegg,-depthg_gridded,tempg_gridded,levels=[26],colors = 'k')
#plt.plot(np.tile(tisaias,len(depthg_gridded)),-depthg_gridded,'--k')
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
plt.ylim(-500,0)
ax.set_xlim(timeg[0],datetime(2020,8,5))

file = folder_fig + 'SG649_temp_transect_around_Isaias_2020_2'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Get temperature transect from model  
                    
# model variable name
model_name = 'GOFS 3.1'
var_name_model = 'water_temp'
var_name_glider = 'temperature' 
contour_plot = 'yes'                   
                    
temp_GOFS, time_GOFS, depth_GOFS, lat_GOFS, lon_GOFS = \
              get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot)

#%%

kw = dict(levels = np.arange(10,31,2))

fig, ax = plt.subplots(figsize=(10, 3))
cs = plt.contourf(time_GOFS,-depth_GOFS,temp_GOFS,cmap=color_map,**kw)
plt.contour(time_GOFS,-depth_GOFS,temp_GOFS,[26],colors='k')

cs = fig.colorbar(cs, orientation='vertical')
cs.ax.set_ylabel(clabel,fontsize=14,labelpad=15)

ax.set_xlim(timeg[0],datetime(2020,8,5))
ax.set_ylim(-np.nanmax(depthg), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ti = datetime(2020,7,17)
xvec = [ti + dt*timedelta(2) for dt in np.arange(10)]
plt.xticks(xvec,fontsize=12)
xfmt = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Along Track Temperature' +\
          ' Profile ' + model_name,fontsize=16)
plt.ylim(-500,0)

file = folder_fig + 'GOFS31_along_track_SG649_around_Isaias_2020'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
