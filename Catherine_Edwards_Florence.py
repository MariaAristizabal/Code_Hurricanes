#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:49:15 2020

@author: root
"""
#%% User input

# Servers location
url_erddap = 'https://data.ioos.us/gliders/erddap'
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
#url_model30 = 'http://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_91.2/ts3z'
#url_glider = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG668-20190819T1217/SG668-20190819T1217.nc3.nc'
url_thredds = 'http://gliders.ioos.us/thredds/dodsC/deployments/secoora/bass-20180808T0000/bass-20180808T0000.nc3.nc'

#date_ini = '2018/09/01/00' # year/month/day/hour
#date_end = '2018/11/30/00' # year/month/day/hour
date_ini = '2018/08/10/00'
date_end = '2018/08/13/00'
scatter_plot = 'no'
contour_plot = 'yes'

# SAB
lon_lim = [-81.75,-75] 
lat_lim = [28,36.2]
#lon_lim = [-85.0,-75.0]
#lat_lim = [25.0,35.0]

# glider variable to retrieve
var_name_glider = 'temperature'
#var_glider = 'salinity'
delta_z = 0.4 # bin size in the vertical when gridding the variable vertical profile 
              # default value is 0.3  

# model variable name
model_name = 'GOFS 3.1'
#model_name = 'GOFS 3.0'
var_name_model = 'water_temp'
#var_model = 'salinity'

# Directories where increments files reside 
#dir_increm = '/Volumes/aristizabal/GOFS/'
dir_increm = '/home/aristizabal/GOFS/'

# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'


#%%
import sys
sys.path
#sys.path.append('/home/aristizabal/glider_model_comparisons_Python/')
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

from read_glider_data import retrieve_dataset_id_erddap_server
from read_glider_data import read_glider_data_erddap_server
from process_glider_data import grid_glider_data
from glider_transect_model_com import get_glider_transect_from_GOFS

import xarray as xr
import netCDF4
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

#%%
url_server = url_erddap

gliders = retrieve_dataset_id_erddap_server(url_server,lat_lim,lon_lim,date_ini,date_end)
print('The gliders found are ')
print(gliders)   

#%%                    
    
dataset_id = gliders[0]
print(dataset_id)
kwargs = dict(date_ini=date_ini,date_end=date_end)

tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                   lat_lim,lon_lim,scatter_plot) #,**kwargs)
    
tempg_gridded, timegg, depthg_gridded = \
                    grid_glider_data(var_name_glider,dataset_id,tempg,timeg,latg,long,depthg,delta_z,contour_plot)

# Get temperature transect from model    
temp_GOFS, time_GOFS, depth_GOFS, lat_GOFS, lon_GOFS = \
              get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                        tempg,timeg,latg,long,depthg,contour_plot='yes')                  
    
#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Reading ncoda data

ncoda_files = sorted(glob.glob(os.path.join(dir_increm,'*seatmp*')))

ncncoda = xr.open_dataset(ncoda_files[0],decode_times=False)
#ncncoda = Dataset(ncoda_files[2]) # Michael
temp_incr = ncncoda.variables['pot_temp'][:]

time_ncoda = ncncoda.MT 
time_ncoda = np.transpose(netCDF4.num2date(time_ncoda[:],time_ncoda.units))
depth_ncoda = ncncoda.variables['Depth'][:]
lat_ncoda = ncncoda.variables['Latitude'][:]
lon_ncoda = ncncoda.variables['Longitude'][:]   

#%%

oklonncoda = np.where(np.logical_and(lon_ncoda-360 > lon_lim[0], lon_ncoda-360 < lon_lim[-1]))
oklatncoda = np.where(np.logical_and(lat_ncoda > lat_lim[0], lat_ncoda < lat_lim[-1]))

fig, ax = plt.subplots() 
#ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)

kw = dict(levels = np.linspace(-3,3,13))
cs = ax.contourf(lon_ncoda[oklonncoda]-360,lat_ncoda[oklatncoda],\
                 temp_incr[0,0,oklatncoda[0][0]:oklatncoda[0][-1]+1,oklonncoda[0][0]:oklonncoda[0][-1]+1],\
                 cmap=plt.get_cmap('seismic'),**kw)
cs=plt.colorbar(cs)
cs.ax.set_ylabel('$(^oc)$',fontsize=16,labelpad=15)
cs.ax.tick_params(labelsize=14) 

#ax.grid(False)
ax.set_ylim([lat_lim[0],lat_lim[-1]])
ax.set_xlim([lon_lim[0],lon_lim[-1]])
plt.title('NCODA Temperature Increments at Surface \n on '+str(time_ncoda[0])[0:10] ,size = 16)
    