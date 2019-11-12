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
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% User input
 
#AVHRR_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L4/GLOB/NCEI/AVHRR_OI/v2/'

url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/L4/GLOB/UKMO/OSTIA/'

year = '2019'
date = '20190828'
day_of_year = '240'

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

#lon_lim = [-100,-10]
#lat_lim = [0,50]

lon_lim = [-80.0,-60.0]
lat_lim = [15.0,30.0]

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

#%% Find url list

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

#%% Loadng data

nc = xr.open_dataset(url + year + '/' + day_of_year + '/' + nc_list[0] + '.bz2'\
                          , decode_times=False) 

sst = np.asarray(nc.analysed_sst[0,:,:])
lat = np.asarray(nc.lat[:])
lon = np.asarray(nc.lon[:])
time = nc.time
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

fig, ax = plt.subplots(figsize=(7,5)) 

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
#cbar.set_label('($^o$C)',rotation=270,size = 18,labelpad = 18)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels()) #,fontsize=18)
ax.set_aspect(1)

#plt.xlim(-100,-10)
#plt.ylim(0,50)
#plt.grid('on')

plt.title('Sea Surface Temperature'+\
          str(time[0].year) + '-' + str(time[0].month) + '-' + str(time[0].day) +\
          '\n GHRSST Level 4')#,fontsize=20)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}.png'.format('SST_AVHRR_Atlantic',\
          time[0].year,time[0].month,time[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Getting time series of temperature at a specific lat and lon

date_ini = '20190825'
date_end = '20190908'

tini = datetime.strptime(date_ini,'%Y%m%d')
tend = datetime.strptime(date_end,'%Y%m%d')

day_year_ini = tini.timetuple().tm_yday
day_year_end = tend.timetuple().tm_yday

#%% Find url list

#all_nc_list = []
sst_time_series = []
t_time_series = []
for t in np.arange((tend-tini).days+1):
#for t in np.arange(2):    
    year = (tini + timedelta(float(t))).year
    day_of_year = (tini + timedelta(float(t))).timetuple().tm_yday
    date = (tini + timedelta(float(t))).strftime('%Y%m%d')
    print(day_of_year)
    
    r = requests.get(url + str(year) + '/' + str(day_of_year) + '/')
    data = r.text
    soup = BeautifulSoup(data,"lxml")

    fold = []
    for s in soup.find_all("a"):
        fold.append(s.get("href").split('/')[0])
 
    nc_file = []
    for f in fold:
        elem = f.split('.')
        for l in elem:
            if l[0:8] == date:
                nc_file.append(f.split('.nc')[0]+'.nc')
                #print(nc_file.append(f.split('.nc')[0]+'.nc'))
    nc_file = list(set(nc_file))[0]
    #print(nc_file)
    
    nc = xr.open_dataset(url + str(year) + '/' + str(day_of_year) + '/' + nc_file + '.bz2'\
                          , decode_times=False) 

    sst = np.asarray(nc.analysed_sst[0,:,:])
    lat = np.asarray(nc.lat[:])
    lon = np.asarray(nc.lon[:])
    time = nc.time
    time = np.transpose(netCDF4.num2date(time[:],time.units))
    
    indx = np.where(lon > -77.4)[0][0]
    indy = np.where(lat > 27.0)[0][0]

    sst_time_series.append(sst[indy,indx]-273.15)
    t_time_series.append(time)
    
sst_time_series = np.asarray(sst_time_series)
t_time_series = np.asarray(t_time_series)

#%% Time series at point underneath Dorian when cat 5

fig,ax1 = plt.subplots(figsize=(12, 4))
plt.plot(t_time_series,sst_time_series,'o-',linewidth=2)

t0 = tini
deltat= timedelta(1)
xticks = [t0+nday*deltat for nday in np.arange(15)]
xticks = np.asarray(xticks)
plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.xlim([tini,tend])
plt.ylabel('$^oC$',fontsize = 14)
tDorian = np.tile(datetime(2019,9,2,0),len(np.arange(22,30,0.1)))
plt.plot(tDorian,np.arange(22,30,0.1),'--k')
plt.title('Surface Temperature GOFS 3.1',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(0.6,1.3))

file = folder + ' ' + 'sst_GHRSST_Dorian'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

