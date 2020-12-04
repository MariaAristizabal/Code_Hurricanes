#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:50:04 2020

@author: aristizabal
"""

#%% cell #7: Search for glider data sets given a
#    latitude and longitude box and time window, choose one those data sets
#    (dataset_id), grid in the vertical the glider transect, get the glider
#    transect in the AmSeas grid, and plot both the transect from the glider
#    deployment and the AmSeas output

from read_glider_data import read_glider_data_erddap_server
from read_glider_data import retrieve_dataset_id_erddap_server
from process_glider_data import grid_glider_data
from glider_transect_model_com import get_glider_transect_from_Amseas

# Servers location
url_erddap = 'https://data.ioos.us/gliders/erddap'
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

# model variable name
model_name = 'Amseas'
var_name_model = 'water_temp'
var_name_glider = 'temperature'

gliders = retrieve_dataset_id_erddap_server(url_erddap,lat_lim,lon_lim,date_ini,date_end)

dataset_id = [idg for idg in gliders if idg[0:4] == 'ru29'][0]

#%% Read glider data and plot scatter plot
tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                   lat_lim,lon_lim,scatter_plot,**kwargs)

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

#gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
#                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
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

#%% Retrive gilder transect from AmSeas

temp_amseas, time_amseas, depth_amseas, lat_amseas, lon_amseas = \
              get_glider_transect_from_Amseas(url_amseas,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot='yes')













#%% Choose time window

from datetime import datetime
import matplotlib.dates as mdates
import numpy as np

ti = mdates.date2num(datetime(2020,10,15))
te = mdates.date2num(datetime(2020,10,27))

okt = np.where(np.logical_and(mdates.date2num(timeg) >= ti,\
                          mdates.date2num(timeg) <= te))[0]

tempgg = tempg[:,okt]
timegg = timeg[okt]
latgg = latg[okt]
longg = long[okt]
depthgg = depthg[:,okt]

tempg_gridded, timeg_gridded, depthg_gridded = \
                    grid_glider_data(var_name_glider,dataset_id,tempgg,timegg,latgg,longg,depthgg,delta_z,contour_plot='yes')

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

#gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
#                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl = ax.gridlines(crs=cartopy.crs.PlateCarree(),draw_labels=True)
gl.xlabels_top = False
gl.ylabels_left = False
#gl.xlines = False
#gl.ylines = False
#gl.xlocator = mticker.FixedLocator([-70, -75, -60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.plot(longg,latgg,'.')

#%% Plot just one transect across

okl = np.arange(5,29)
#okl = np.arange(28,50)

timet = timegg[okl]
tempt = tempgg[:,okl]
latt = latgg[okl]
lont= longg[okl]
deptht = depthgg[:,okl]

plt.figure()
plt.plot(timet,latt,'.-')

tempg_gridded, timeg_gridded, depthg_gridded = \
                    grid_glider_data(var_name_glider,dataset_id,tempt,timet,latt,lont,deptht,delta_z,contour_plot='yes')

temp_amseas, time_amseas, depth_amseas, lat_amseas, lon_amseas = \
              get_glider_transect_from_Amseas(url_amseas,var_name_model,model_name,\
                                        tempt,timet,latt,lont,deptht,contour_plot='yes')
