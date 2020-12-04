#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:50:04 2020

@author: aristizabal
"""

#%% Search for glider data sets given a latitude and longitude box and time window, 
# choose one those data sets(dataset_id), 
# get the glider transect in the AmSeas grid, and plot it

from read_glider_data import retrieve_dataset_id_erddap_server
from read_glider_data import read_glider_data_erddap_server
from process_glider_data import grid_glider_data
from glider_transect_model_com import get_glider_transect_from_Amseas
from glider_transect_model_com import get_glider_transect_from_GOFS

# Servers location
url_erddap = 'https://data.ioos.us/gliders/erddap'
url_erddap_Rutgers = "http://slocum-data.marine.rutgers.edu/erddap"
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'
url_amseas = 'https://www.ncei.noaa.gov/thredds-coastal/dodsC/amseas/amseas_20130405_to_current/' #'20190901/ncom_relo_amseas_u_2019090100_t003.nc'

# Caribbean
lon_lim = [-80,-60.0]
lat_lim = [10.0,30.0]

# date limits
date_ini = '2020/10/17/20 00:00:00'
date_end = '2020/10/18/20 16:00:00'
#date_ini = '2020/07/01/00'
#date_end = '2020/10/28/00'
kwargs = dict(date_ini=date_ini,date_end=date_end)

scatter_plot = 'yes'
delta_z = 0.4     # default value is 0.3

var_name_glider = 'temperature'

folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%
gliders = retrieve_dataset_id_erddap_server(url_erddap,lat_lim,lon_lim,date_ini,date_end)

dataset_id = [idg for idg in gliders if idg[0:4] == 'ru29'][0]

#%% Read glider data and plot scatter plot

tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                   lat_lim,lon_lim,scatter_plot,**kwargs)
    
#%% Grid glider data
    
contour_plot = 'no'
tempg_gridded, timeg, depthg_gridded = \
                    grid_glider_data(var_name_glider,dataset_id,tempg,timeg,latg,long,depthg,delta_z,contour_plot)     

#%% Retrive temp transect from GOFS

# model variable name
model_name = 'GOFS'
var_name_model = 'water_temp'

temp_GOFS, time_GOFS, depth_GOFS, lat_GOFS, lon_GOFS = \
              get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot)
                  
#%% Retrive velocity transect from GOFS

# model variable name
model_name = 'GOFS'
url_GOFS = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'

var_name_model = 'water_u'
u_GOFS, time_GOFS, depth_GOFS, lat_GOFS, lon_GOFS = \
              get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot)
                  
var_name_model = 'water_v'
v_GOFS, time_GOFS, depth_GOFS, lat_GOFS, lon_GOFS = \
              get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot)
                  
#%% Retrive temp transect from AmSeas

# model variable name
model_name = 'Amseas'
var_name_model = 'water_temp'

temp_amseas, time_amseas, depth_amseas, lat_amseas, lon_amseas = \
              get_glider_transect_from_Amseas(url_amseas,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot)

#%% Retrive velocity  transect from AmSeas                  

# model variable name
model_name = 'Amseas'

var_name_model = 'water_u'
u_amseas, time_amseas, depth_amseas, lat_amseas, lon_amseas = \
              get_glider_transect_from_Amseas(url_amseas,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot)
                  
var_name_model = 'water_v'
v_amseas, time_amseas, depth_amseas, lat_amseas, lon_amseas = \
              get_glider_transect_from_Amseas(url_amseas,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot)

#%% Read velocity from Rutgers erddap server

from erddapy import ERDDAP
import matplotlib.pyplot as plt
import numpy as np

date_ini = kwargs.get('date_ini', None)
date_end = kwargs.get('date_end', None)

# Find time window of interest    
if np.logical_or(date_ini==None,date_end==None):
    constraints = {
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }
else:
    constraints = {
        'time>=': date_ini,
        'time<=': date_end,
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }
 
variables = [
        'depth',
        'latitude',
        'longitude',
        'time',
        'temperature',
        'salinity',
        'u',
        'v'
        ]

e = ERDDAP(
        server=url_erddap_Rutgers,
        protocol='tabledap',
        response='nc'
        )

e.dataset_id = dataset_id + '-profile-sci-rt'
e.constraints = constraints
e.variables = variables
    
# Converting glider data xarray
ds = e.to_xarray()

#%% Get rid of nans

ttu, indt = np.unique(ds['time'].values,return_index='true')

oku = np.isfinite(ds['u'].values[indt])
ug = ds['u'].values[indt][oku]
time_ug = ds['time'].values[indt][oku]

okv = np.isfinite(ds['v'].values[indt])
vg = ds['v'].values[indt][okv]
time_vg = ds['time'].values[indt][okv]

#%% Plot

import matplotlib.dates as mdates
import cmocean
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime

plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

okg = depthg <= np.nanmax(depthg)
okG = depth_GOFS <= np.nanmax(depthg)
okA = depth_amseas <= np.nanmax(depthg)
min_val = np.int(np.floor(np.min([np.nanmin(tempg[okg]),np.nanmin(temp_GOFS[okG]),np.nanmin(temp_amseas[okA])])))
max_val = np.int(np.ceil(np.max([np.nanmax(tempg[okg]),np.nanmax(temp_GOFS[okG]),np.nanmax(temp_amseas[okA])])))
nlevels = int(max_val - min_val + 1)
kw = dict(levels = np.linspace(min_val,max_val,nlevels)) 
max_depth = -np.nanmax(depthg)
clabel = ' ($^oC$)'

fig,ax = plt.subplots(figsize=(22, 8))
grid = plt.GridSpec(3, 3, wspace=0.1, hspace=0.1)

#####
ax = plt.subplot(grid[0:2,0])
cl = plt.contour(timeg,-depthg_gridded,tempg_gridded,levels=[26],colors = 'k')
ax.clabel(cl,inline=1,fontsize=10,fmt='%i$^o$')
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=cmocean.cm.thermal,**kw)
plt.title('Temperature ' + dataset_id,fontsize=18)

ax.set_xlim(time_GOFS[0], time_GOFS[-1])
ax.set_ylim(max_depth, 0)
xfmt = mdates.DateFormatter('%b-%d \nH:%H')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticks([datetime(2020,10,17,4),datetime(2020,10,17,14),\
               datetime(2020,10,18,0),datetime(2020,10,18,10)])
ax.set_xticklabels([])
ax.set_ylabel('Depth (m)',fontsize=14)

#add color bar below chart
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size='10%', pad=0.8)
#fig.add_axes(cax)
#cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
#cbar.ax.set_ylabel(clabel,fontsize=14)

#######
ax = plt.subplot(grid[0:2,1])
cl = plt.contour(time_GOFS,-depth_GOFS,temp_GOFS,levels=[26],colors = 'k')
ax.clabel(cl,inline=1,fontsize=10,fmt='%i$^o$')
cs = plt.contourf(time_GOFS,-depth_GOFS,temp_GOFS,cmap=cmocean.cm.thermal,**kw)
plt.title('Temperature GOFS 3.1',fontsize=18)

ax.set_xlim(time_GOFS[0], time_GOFS[-1])
ax.set_ylim(max_depth, 0)
ax.set_yticklabels([' '])
xfmt = mdates.DateFormatter('%b-%d \nH:%H')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticks([datetime(2020,10,17,4),datetime(2020,10,17,14),\
               datetime(2020,10,18,0),datetime(2020,10,18,10)])
ax.set_xticklabels([])
#ax.set_ylabel('Depth (m)',fontsize=14)

#add color bar below chart
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size='10%', pad=0.8)
fig.add_axes(cax)
cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
cbar.ax.set_ylabel(clabel,fontsize=16)

#######
ax = plt.subplot(grid[0:2,2])
cl = plt.contour(time_amseas,-depth_amseas,temp_amseas,levels=[26],colors = 'k')
ax.clabel(cl,inline=1,fontsize=10,fmt='%i$^o$')
cs = plt.contourf(time_amseas,-depth_amseas,temp_amseas,cmap=cmocean.cm.thermal,**kw)
plt.title('Temperature AmSeas',fontsize=18)

ax.set_xlim(time_GOFS[0], time_GOFS[-1])
ax.set_ylim(max_depth, 0)
ax.set_yticklabels([' '])
xfmt = mdates.DateFormatter('%b-%d \nH:%H')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticks([datetime(2020,10,17,4),datetime(2020,10,17,14),\
               datetime(2020,10,18,0),datetime(2020,10,18,10)])
ax.set_xticklabels([])
#ax.set_ylabel('Depth (m)',fontsize=14)

#add color bar below chart
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size='10%', pad=0.8)
#fig.add_axes(cax)
#cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
#cbar.ax.set_ylabel(clabel,fontsize=14)

#####
ax = plt.subplot(grid[2,0])
plt.plot(time_ug,ug,'o-',markeredgecolor='k')
plt.plot(time_vg,vg,'o-',markeredgecolor='k')
plt.plot(time_ug,np.tile(0,len(time_ug)),'--k')
xfmt = mdates.DateFormatter('%b-%d \nH:%H')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticks([datetime(2020,10,17,4),datetime(2020,10,17,14),\
               datetime(2020,10,18,0),datetime(2020,10,18,10)])
ax.set_xlim(time_GOFS[0], time_GOFS[-1])
ax.set_ylim(-0.32,0.32)
ax.set_ylabel('Velocity (m/s)',fontsize=16)


#####
okd1000 = depth_GOFS < 1000
umean_GOFS = np.nanmean(u_GOFS[okd1000,:],axis=0)
vmean_GOFS = np.nanmean(v_GOFS[okd1000,:],axis=0)

ax = plt.subplot(grid[2,1])
plt.plot(time_GOFS,umean_GOFS,'o-',markeredgecolor='k',\
         label='Eastward Depth-Averaged Current')
plt.plot(time_GOFS,vmean_GOFS,'o-',markeredgecolor='k',
         label='Northward Depth-Averaged Current')
plt.plot(time_ug,np.tile(0,len(time_ug)),'--k')
xfmt = mdates.DateFormatter('%b-%d \nH:%H')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticks([datetime(2020,10,17,4),datetime(2020,10,17,14),\
               datetime(2020,10,18,0),datetime(2020,10,18,10)])
ax.set_yticklabels([])
ax.set_xlim(time_GOFS[0], time_GOFS[-1])
ax.set_ylim(-0.32,0.32)
ax.legend(fontsize=12)

#####
okd1000 = depth_amseas < 1000
umean_amseas = np.nanmean(u_amseas[okd1000,:],axis=0)
vmean_amseas = np.nanmean(v_amseas[okd1000,:],axis=0)

ax = plt.subplot(grid[2,2])
plt.plot(time_amseas,umean_amseas,'o-',markeredgecolor='k')
plt.plot(time_amseas,vmean_amseas,'o-',markeredgecolor='k')
plt.plot(time_ug,np.tile(0,len(time_ug)),'--k')
xfmt = mdates.DateFormatter('%b-%d \nH:%H')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticks([datetime(2020,10,17,4),datetime(2020,10,17,14),\
               datetime(2020,10,18,0),datetime(2020,10,18,10)])
ax.set_xlim(time_GOFS[0], time_GOFS[-1])
ax.set_ylim(-0.32,0.32)
ax.set_yticklabels([])

file = folder_fig + dataset_id + '_vs_GOFS_vs_AmSeas'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Plot trayectory

import matplotlib.pyplot as plt
import cartopy
import cartopy.feature as cfeature
#import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig, ax = plt.subplots(figsize=(7, 3),subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
coast = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
ax.add_feature(coast, edgecolor='black', facecolor='none')
ax.add_feature(cfeature.BORDERS)  # adds country borders
ax.add_feature(cfeature.STATES)    # adds statet borders
plt.axis([-70,-60,15,20])
gl = ax.gridlines(crs=cartopy.crs.PlateCarree(),draw_labels=True)
gl.xlabels_top = False
gl.ylabels_left = False
#gl.xlines = False
#gl.ylines = False
#gl.xlocator = mticker.FixedLocator([-70, -75, -60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.plot(long,latg,'.')

#%% plot lat as a function of time

plt.figure()
plt.plot(timeg,latg,'.-')

plt.figure()
plt.plot(latg,'.-')


