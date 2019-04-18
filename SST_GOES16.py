#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:22:38 2019

@author: aristizabal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:50:55 2019

@author: aristizabal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:45:54 2019

@author: aristizabal
"""

import matplotlib.pyplot as plt
import xarray as xr
import cmocean
from bs4 import BeautifulSoup
import requests
import netCDF4
import numpy as np

import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#%% User input

# GHRSST Level 3C sub-skin Sea Surface Temperature from the Geostationary Operational 
#Environmental Satellites (GOES 16) Advanced Baseline Imager (ABI) 
#in East position (GDS V2) produced by OSI SAF
AVHRR_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L3C/AMERICAS/GOES16/OSISAF/v1/'

# Florence
#year = '2018'
#day_of_year = '254' # set 11 2018
#date = '20180911'

# Michael
year = '2018'
day_of_year = '283' # set 11 2018
date = '20181010'

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

lon_lim = [-100,-10]
lat_lim = [0,50]

#%% Find url list

#works for 
# AVHRR_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L3C/AMERICAS/GOES16/OSISAF/v1/'

r = requests.get(AVHRR_url + year + '/' + day_of_year + '/')
data = r.text
soup = BeautifulSoup(data,"lxml")

fold = []
for s in soup.find_all("a"):
    fold.append(s.get("href").split('/')[0])
 
nc_list = []
for f in fold:
    elem = f.split('.')
    for l in elem:
        if l[0:8] == date:
            nc_list.append(f.split('.nc')[0]+'.nc')
nc_list = list(set(nc_list)) 

#%% America map

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())
cs = ax.contourf(lon_avhrr, lat_avhrr,sst-273.15,cmap=plt.cm.Spectral_r)  
cbar = fig.colorbar(cs, orientation='vertical')
cbar.set_label('(m)',rotation=270,size = 18,labelpad = 20)
plt.title('{0} {1}-{2}-{3}'.format('Sea Surface Temperature',\
          time_avhrr[0].year,time_avhrr[0].month,time_avhrr[0].day))   
 
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
file = folder + '{0}_{1}_{2}_{3}.png'.format('SST',\
          time_avhrr[0].year,time_avhrr[0].month,time_avhrr[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 