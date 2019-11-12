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
#from bs4 import BeautifulSoup
#import requests
import netCDF4
import numpy as np

#import cartopy.crs as ccrs
#import cartopy
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% User input

# GHRSST Level 3C sub-skin Sea Surface Temperature from the Geostationary Operational 
#Environmental Satellites (GOES 16) Advanced Baseline Imager (ABI) 
#in East position (GDS V2) produced by OSI SAF
#AVHRR_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L3C/AMERICAS/GOES16/OSISAF/v1/'

# AVHRR Individual Passes (Unmasked, 2013-2018)
#http://tds.maracoos.org/thredds/dodsC/AVHRR/2018/Unmasked/Files/20181231.365.2318.n18.EC1.nc
#AVHRR_url = 'http://tds.maracoos.org/thredds/dodsC/AVHRR/2018/Unmasked/Files/'

#year = '2018'
#day_of_year = '254' # set 11 2018
#date = '20180911'

year = '2019'
day_of_year = '246' # 20190827
date = '20190903'

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

lon_lim = [-100,-10]
lat_lim = [0,50]

#lon_lim = [-80.0,-60.0]
#lat_lim = [15.0,30.0]

#%% Find url list
'''
#AVHRR_url + year + day_of_year

r = requests.get(url + year + '/' + day_of_year + '/')
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
'''

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.lat
bath_lon = ncbath.lon
bath_elev = ncbath.elevation

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevsub = bath_elev[oklatbath,oklonbath]

#%% Loadng data

#AVHRR_file = 'http://tds.maracoos.org/thredds/dodsC/AVHRR/2018/Unmasked/Files/20180911.254.2036.n19.EC1.nc'

#GHRSST L3C global sub-skin Sea Surface Temperature from the Advanced Very High 
#Resolution Radiometer (AVHRR) on Metop satellites (currently Metop-B) (GDS V2) 
#produced by OSI SAF

# Florence
#AVHRR_file = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/GDS2/L3C/GLOB/AVHRR_SST_METOP_B_GLB/OSISAF/v1/2018/254/20180911120000-OSISAF-L3C_GHRSST-SSTsubskin-AVHRR_SST_METOP_B_GLB-sstglb_metop01_20180911_120000-v02.0-fv01.0.nc'
AVHRR_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/nc_20180911120000-OSISAF-L3C_GHRSST-SSTsubskin-AVHRR_SST_METOP_B_GLB-sstglb_metop01_20180911_120000-v02.0-fv01.0.nc.nc4'

# Michael 2018/10/10 00
#AVHRR_file = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/GDS2/L3C/GLOB/AVHRR_SST_METOP_B_GLB/OSISAF/v1/2018/282/20181010000000-OSISAF-L3C_GHRSST-SSTsubskin-AVHRR_SST_METOP_B_GLB-sstglb_metop01_20181010_000000-v02.0-fv01.0.nc'
#AVHRR_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/nc_20181010000000-OSISAF-L3C_GHRSST-SSTsubskin-AVHRR_SST_METOP_B_GLB-sstglb_metop01_20181010_000000-v02.0-fv01.0.nc.nc4'

#MIchale 2018/10/09/ 00
#AVHRR_file = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/GDS2/L3C/GLOB/AVHRR_SST_METOP_B_GLB/OSISAF/v1/2018/281/20181009000000-OSISAF-L3C_GHRSST-SSTsubskin-AVHRR_SST_METOP_B_GLB-sstglb_metop01_20181009_000000-v02.0-fv01.0.nc'
#AVHRR_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/nc_20181009000000-OSISAF-L3C_GHRSST-SSTsubskin-AVHRR_SST_METOP_B_GLB-sstglb_metop01_20181009_000000-v02.0-fv01.0.nc.nc4'

# MIchael 2018/10/11 00
#AVHRR_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/nc_20181011000000-OSISAF-L3C_GHRSST-SSTsubskin-AVHRR_SST_METOP_B_GLB-sstglb_metop01_20181011_000000-v02.0-fv01.0.nc.nc4'

# Michael 2018/10/12 00
#AVHRR_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/nc_20181012000000-OSISAF-L3C_GHRSST-SSTsubskin-AVHRR_SST_METOP_B_GLB-sstglb_metop01_20181012_000000-v02.0-fv01.0.nc.nc4'

ncavhrr = xr.open_dataset(AVHRR_file, decode_times=False) 
#nc = xr.open_dataset(AVHRR_file + year + '/' + day_of_year + '/' + nc_list[0] + '.bz2'\
#                          , decode_times=False) 

#xr.open_dataset('https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/L4/GLOB/ABOM/GAMSSA_28km/2019/239/20190827-ABOM-L4LRfnd-GLOB-v01-fv01_0-GAMSSA_28km.nc.bz2')

#mcsst = ncavhrr.mcsst[0,:,:]
#sst = nc.sea_surface_temperature[0,:,:]
sst = np.asarray(ncavhrr.analysed_sst[0,:,:])
lat = np.asarray(ncavhrr.lat[:])
lon = np.asarray(ncavhrr.lon[:])
time = ncavhrr.time
time = np.transpose(netCDF4.num2date(time[:],time.units))

#%% Global map

'''
fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())
cs = ax.contourf(lon_avhrr, lat_avhrr,sst,cmap=plt.cm.Spectral_r)  
cbar = fig.colorbar(cs, orientation='vertical')
cbar.set_label('($^oC$)',rotation=270,size = 18,labelpad = 20)
plt.title('{0} {1}-{2}-{3}'.format('Sea Surface Temperature',\
          time_avhrr[0].year,time_avhrr[0].month,time_avhrr[0].day))   

# Draw coastlines
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='lightblue', alpha=0.4)
#gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER
#gl.xlabels_top = False
#gl.ylabels_right = False
#gl.xlabel_style = {'size': 14}
#gl.ylabel_style = {'size': 14}

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}.png'.format('SST_AVHRR',\
          time_avhrr[0].year,time_avhrr[0].month,time_avhrr[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
'''

#%% Atlantic map

fig = plt.figure(figsize=(13,8))
ax = plt.subplot()

oklon = np.logical_and(lon > lon_lim[0],lon < lon_lim[-1])
oklat = np.logical_and(lat > lat_lim[0],lat < lat_lim[-1])

lonsub = lon[oklon]
latsub = lat[oklat]
sstsu = sst[oklat,:]
sstsub = sstsu[:,oklon]

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

cs = ax.contourf(lonsub, latsub, sstsub-273.15, \
                 cmap=cmocean.cm.thermal,levels=np.linspace(21,33,25))
cbar = fig.colorbar(cs, orientation='vertical')
cbar.set_label('($^o$C)',rotation=270,size = 18,labelpad = 20)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=20)
ax.set_aspect(1)

#plt.xlim(-100,-10)
#plt.ylim(0,50)
#plt.grid('on')

#plt.title('{0} {1}-{2}-{3}'.format('Sea Surface Temperature',\
#          time[0].year,time[0].month,time[0].day),\
#          fontsize=20)

plt.title('Sea Surface Temperature'+\
          str(time[0].year) + '-' + str(time[0].month) + '-' + str(time[0].day) +\
          '\n GHRSST Level 4',fontsize=20)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}.png'.format('SST_AVHRR_Atlantic',\
          time[0].year,time[0].month,time[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% GoM map
'''
plt.figure(figsize=(10,8))

oklon = np.logical_and(lon_avhrr > lon_lim[0],lon_avhrr < lon_lim[-1])
oklat = np.logical_and(lat_avhrr > lat_lim[0],lat_avhrr < lat_lim[-1])

sstsub = sst[oklat,oklon]
lonsub = lon_avhrr[oklon]
latsub = lat_avhrr[oklat]

plt.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

plt.xlim(-98,-79.5)
plt.ylim(15,32.5)

cs = plt.contourf(lonsub, latsub, sstsub-273.15, \
                 levels=np.linspace(26,31,11),\
                 cmap=plt.cm.Spectral_r)
cbar = plt.colorbar(cs, orientation='vertical')
cbar.set_label('($^o$C)',size = 18,labelpad = 20)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=20)


plt.grid('on')

plt.title('Sea Surface Temperature '+str(time_avhrr[0]),fontsize=20)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'SST_AVHRR_GoM_' + str(time_avhrr[0])
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
'''