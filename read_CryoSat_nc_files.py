#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:02:28 2019

@author: aristizabal
"""

#%% User input

#cryosat_url = 'https://science-pds.cryosat.esa.int/#Cry0Sat2_data%2FSIR_GOP_P2P%2F2018%2F10'
cryosat_folder = '/Volumes/aristizabal/CryoSat_data/'
target_date = '20180911'

#%%
import matplotlib.pyplot as plt
import xarray as xr
import os
import netCDF4
import numpy as np

import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean

#%% Access files

nc_list = os.listdir(cryosat_folder)

nc_listsub = []
for l in nc_list:
    if l.split('_')[6][0:8] == target_date:
       nc_listsub.append(l) 

#%%
'''
cryosat_file = cryosat_folder+ nc_list[0]
nccryosat = xr.open_dataset(cryosat_file , decode_times=False) 
latcryo=nccryosat.lat_20_ku[:]
loncryo=nccryosat.lon_20_ku[:]
'''

#%%

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())

for l in nc_listsub:
    nccryosat = xr.open_dataset(cryosat_folder + l, decode_times=False) 
    #mssh = ncjason.mean_sea_surface
    ssha = nccryosat.ssha_20_ku
    lat_cryosat = nccryosat.lat_20_ku
    lon_cryosat = nccryosat.lon_20_ku
    
    kw = dict(s=30, c=ssha, marker='*', edgecolor='none',vmin=-0.5,vmax=0.5)
    cs = ax.scatter(lon_cryosat, lat_cryosat, **kw, cmap=cmocean.cm.balance)

time_cryos = nccryosat.time_20_ku
time_cryosat = np.transpose(netCDF4.num2date(time_cryos[:],time_cryos.units))
cbar = fig.colorbar(cs, orientation='vertical')
cbar.set_label('(m)',rotation=270,size = 18,labelpad = 20)
plt.title('{0} {1}-{2}-{3}'.format('Sea Surface Height Anomaly from CryoSat',\
         time_cryosat[0].year,time_cryosat[0].month,time_cryosat[0].day))
   
# Draw coastlines
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='lightblue', alpha=0.4)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}.png'.format('SSHA CryoSat',\
          time_cryosat[0].year,time_cryosat[0].month,time_cryosat[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
