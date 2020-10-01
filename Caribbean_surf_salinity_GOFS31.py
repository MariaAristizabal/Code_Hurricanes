#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:04:30 2020

@author: aristizabal
"""

lon_lim = [-88.0,-60.0]
lat_lim = [8.0,30.0]

date_ini = '2020/07/28/00'
date_end = '2020/07/28/00'

# url for GOFS 3.1
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'

file_EEZs = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/World_EEZ_v11_20191118/eez_boundaries_v11.shp'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%

import xarray as xr
import netCDF4
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import cmocean

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file,decode_times=False)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% GOGF 3.1

GOFS31 = xr.open_dataset(catalog31,decode_times=False)
#GOFS31 = Dataset(catalog31,decode_times=False)
    
latt31 = GOFS31['lat'][:]
lonn31 = GOFS31['lon'][:]
depth31 = GOFS31['depth'][:]
tt31= GOFS31['time']
t31 = netCDF4.num2date(tt31[:],tt31.units) 

tmin = datetime.strptime(date_ini,'%Y/%m/%d/%H')
tmax = datetime.strptime(date_end,'%Y/%m/%d/%H')

oktime31 = np.where(np.logical_and(t31 >= tmin, t31 <= tmax))[0]
time31 = t31[oktime31]

# Conversion from glider longitude and latitude to GOFS convention
lon_limG = np.empty((len(lon_lim),))
lon_limG[:] = np.nan
for i in range(len(lon_lim)):
    if lon_lim[i] < 0: 
        lon_limG[i] = 360 + lon_lim[i]
    else:
        lon_limG[i] = lon_lim[i]
lat_limG = lat_lim

### Build the bbox for the xy data
botm  = int(np.where(latt31 == lat_limG[0])[0][0])
top   = int(np.where(latt31 == lat_limG[1])[0][0])

left  = np.where(lonn31 > lon_limG[0])[0][0]
right = np.where(lonn31 > lon_limG[1])[0][0]
lat31= latt31[botm:top]
lon31= lonn31[left:right]

# Conversion from GOFS convention to glider longitude and latitude
lon31g= np.empty((len(lon31),))
lon31g[:] = np.nan
for i in range(len(lon31)):
    if lon31[i] > 180: 
        lon31g[i] = lon31[i] - 360 
    else:
        lon31g[i] = lon31[i]
lat31g = lat31

#%% Figures surface salinity

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31):
    print(i)
    salt_GOFS = GOFS31['salinity'][ind,0,botm:top,left:right]

    fig, ax = plt.subplots(figsize=(9, 6),subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
    coast = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    ax.add_feature(coast, edgecolor='black', facecolor='none')
    ax.add_feature(cfeature.BORDERS)  # adds country borders  
    ax.add_feature(cfeature.STATES)
    
    shape_feature = cfeature.ShapelyFeature(Reader(file_EEZs).geometries(),
                                   ccrs.PlateCarree(),edgecolor='grey',facecolor='none')
    ax.add_feature(shape_feature,zorder=1)

    plt.title('Salinity  \n GOFS 3.1 on ' + str(time31[0])[0:13],fontsize = 16)
    
    #ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    #ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    kw = dict(levels=np.arange(31,37.1,0.5))
                  
    plt.contourf(lon31g,lat31g,salt_GOFS, cmap=cmocean.cm.haline, **kw)

    #ax.plot(np.asarray(long),np.asarray(latg),'.k')
    #okt = np.where(timeg > time31[i])
    #ax.plot(long[okt[0][0]],latg[okt[0][0]],'*',color='k')
    #print(long[okt[0][0]],latg[okt[0][0]])
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    #cb.set_label('Salinity',rotation=270, labelpad=25, fontsize=12)
    
    #plt.xlim(-90,-80)
    #plt.ylim(20,30)
    
    file = folder + 'Caribbean_surf_salinity_' + str(time31[i])[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 