#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:23:45 2019

@author: aristizabal
"""

#%% User input

# RU22
lon_lim = [120,134]
lat_lim = [30,40]
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc'

#Time window
date_ini = '2018/08/17/00/00'
date_end = '2018/08/18/00/00'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_90.0_0.0_180.0_45.0.nc'

# url for GOFS 3.1
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from netCDF4 import Dataset

import datetime
import matplotlib.dates as mdates

#%% Glider data
        
dglider = xr.open_dataset(gdata,decode_times=False) 

inst_id = dglider.id.split('_')[0]
inst_name = inst_id.split('-')[0]  

latitude = dglider.latitude[0] 
longitude = dglider.longitude[0]

## Change time into standardized mdates datenums 
seconds_since1970 = dglider.time[0]
timei = datetime.datetime.strptime(dglider.time.time_origin,'%d-%b-%Y %H:%M:%S')
timei + datetime.timedelta(seconds=int(seconds_since1970[0]))
time = np.empty(len(seconds_since1970))
for ind, hrs in enumerate(seconds_since1970):
    time[ind] = mdates.date2num(timei + datetime.timedelta(seconds=int(hrs)))

# Find time window of interest
tti = time[0] 
tte = time[-1]   
oktimeg = np.logical_and(time >= tti,time <= tte)

# Fiels within time window
timeg = time[oktimeg]
latg = latitude[oktimeg]
long = longitude[oktimeg]

# Change glider lot and lat to GOFS 3.1 convention
target_lon = np.empty((len(long),))
target_lon[:] = np.nan
for i in range(len(long)):
    if long[i] < 0: 
        target_lon[i] = 360 + long[i]
    else:
        target_lon[i] = long[i]
target_lat = latg

#%% GOGF 3.1

df = xr.open_dataset(catalog31,decode_times=False)

#%%
## Decode the GOFS3.1 time into standardized mdates datenums 
hours_since2000 = df.time
time_naut       = datetime.datetime(2000,1,1)
time31 = np.ones_like(hours_since2000)
for ind, hrs in enumerate(hours_since2000):
    time31[ind] = mdates.date2num(time_naut+datetime.timedelta(hours=int(hrs)))

## Find the dates of import
dini = mdates.date2num(datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H/%M')) # October 7, 2018
dend = mdates.date2num(datetime.datetime.strptime(date_end,'%Y/%m/%d/%H/%M')) # October 11, 2018
formed  = int(np.where(time31 >= dini)[0][0])
dissip  = int(np.where(time31 >= dend)[0][0])
oktime31 = np.arange(formed,dissip+1,dtype=int)

lat31 = df.lat
lon31 = df.lon
depth31 = df.depth

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
botm  = int(np.where(df.lat == lat_limG[0])[0][0])
top   = int(np.where(df.lat == lat_limG[1])[0][0])
left  = np.where(df.lon > lon_limG[0])[0][0]
right = np.where(df.lon > lon_limG[1])[0][0]
#oklat31 = np.where(np.logical_and(df.lat >= lat_limG[0], df.lat <= lat_lim[-1]))[0]
#oklon31 = np.where(np.logical_and(df.lon >= lon_limG[0], df.lon <= lon_lim[-1]))[0]
lat31= df.lat[botm:top]
lon31= df.lon[left:right]

# Conversion from GOFS convention to glider longitude and latitude
lon31g= np.empty((len(lon31),))
lon31g[:] = np.nan
for i in range(len(lon31)):
    if lon31[i] > 180: 
        lon31g[i] = lon31[i] - 360 
    else:
        lon31g[i] = lon31[i]
lat31g = lat31

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Typhoon Soulik best track

lonSl = np.array([136.4,135.5,134.5,133.3,132.1,130.9,129.8,128.7,127.6,126.8,\
                  126.2,125.7,125.5,125.7,126.1,127.3])
latSl = np.array([25.8,26.1,26.4,27.0,27.7,28.3,28.9,29.7,30.4,31.2,\
                  31.9,32.6,33.0,33.6,34.3,35.7])
tSl = ['2018/08/20/00/00','2018/08/20/06/00','2018/08/20/12/00','2018/08/20/18/00',\
       '2018/08/21/00/00','2018/08/21/06/00','2018/08/21/12/00','2018/08/21/18/00',\
       '2018/08/22/00/00','2018/08/22/06/00','2018/08/22/12/00','2018/08/22/18/00',\
       '2018/08/23/00/00','2018/08/23/06/00','2018/08/23/12/00','2018/08/23/18/00']

# Convert time to UTC
#pst = pytz.timezone('America/New_York') # time zone
#utc = pytz.UTC 

timeSl = [None]*len(tSl) 
for x in range(len(tSl)):
    timeSl[x] = datetime.datetime.strptime(tSl[x], '%Y/%m/%d/%H/%M')


#%% Figures surface salinity

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31):
    S31 = df.salinity[ind,0,botm:top,left:right]
    var = S31

    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
    plot_date = mdates.num2date(time31[ind])
    plt.title('Salinity  \n GOFS 3.1 on {}'.format(plot_date))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(31,35,9), 
              cmap='RdYlBu_r')
        
    plt.contour(lon31g,lat31g, var,'--',levels=33,colors='k')#**kw)
    plt.contourf(lon31g,lat31g, var, **kw)

    okt = np.where(timeg > time31[ind])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'*',color='k')
    
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('Salinity',rotation=270, labelpad=25, fontsize=12)
    
    file = folder + '{0}_{1}.png'.format('Salt_GOFS31_Yellow_sea',\
                     mdates.num2date(time31[ind]))
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
      
#%% Figure with grid points on top 

ind = oktime31[4]

meshlon31 = np.meshgrid(lon31,lat31)
meshlat31 = np.meshgrid(lat31,lon31)

S31 = df.salinity[ind,0,botm:top,left:right]
var = S31

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
plot_date = mdates.num2date(time31[ind])
plt.title('Salinity  \n GOFS 3.1 on {}'.format(plot_date),fontsize=20)
    
ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

max_v = np.nanmax(abs(var))
kw = dict(levels=np.linspace(31,35,9), 
           cmap='RdYlBu_r')
        
plt.contour(lon31g,lat31g, var,'--',levels=33,colors='k')
plt.contourf(lon31,lat31, var, **kw)

cb = plt.colorbar()
cb.set_label('Salinity',rotation=270, labelpad=25, fontsize=16)

ax.plot(meshlon31[0],meshlat31[0].T,'.k')

ax.plot(lonSl,latSl,'o-',markersize = 15,label = 'Soulik Track',\
        color = 'dimgray',linewidth=5)

#for x in range(0, len(tSl)):
for x in range(11,14):    
    ax.text(lonSl[x],latSl[x],timeSl[x].strftime('%d, %H:%M'),size = 14,fontweight='bold')
legend = ax.legend(loc='upper right',fontsize = 14)

ax.plot(np.asarray(long),np.asarray(latg),'.k')
okt = np.where(timeg > time31[ind])
ax.plot(long[okt[0][0]],latg[okt[0][0]],'*',color='red')

plt.xlim(125,128)
plt.ylim(32,34)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
    
file = folder + '{0}_{1}.png'.format('Salt_GOFS31_Yellow_sea',\
                 mdates.num2date(time31[ind]))
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)     
    
    
#%% Figures temperature at 60 m

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31):
    T31 = df.water_temp[ind,15,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
    plot_date = mdates.num2date(time31[ind])
    plt.title('Temperature  \n GOFS 3.1 on {}'.format(plot_date))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(10,17,8), 
              cmap='RdYlBu_r')
    plt.contour(lon31,lat31, var,**kw)
    plt.contourf(lon31,lat31, var, **kw)

    okt = np.where(timeg > time31[ind])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'o',markeredgecolor='k',color='white')
    
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('$(^oC)$',rotation=270, labelpad=25, fontsize=12)
    
    file = folder + '{0}_{1}.png'.format('Temp_GOFS31_Yellow_sea',\
                     mdates.num2date(time31[ind]))
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0)     


#%% Figures temperature at surface

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31):
    T31 = df.water_temp[ind,0,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
    plot_date = mdates.num2date(time31[ind])
    plt.title('Temperature  \n GOFS 3.1 on {}'.format(plot_date))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(20,30,11), 
              cmap='RdYlBu_r')
    plt.contour(lon31g,lat31g, var,**kw)
    plt.contourf(lon31g,lat31g, var, **kw)

    okt = np.where(timeg > time31[ind])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'o',markeredgecolor='k',color='white')
    
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('$(^oC)$',rotation=270, labelpad=25, fontsize=12)
    
    file = folder + '{0}_{1}.png'.format('Temp_GOFS31_MAB',\
                     mdates.num2date(time31[ind]))
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0)     

#%% Figures bottom temperature

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31):
    T31 = df.water_temp_bottom[ind,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
    plot_date = mdates.num2date(time31[ind])
    plt.title('Bottom Temperature  \n GOFS 3.1 on {}'.format(plot_date))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(4,30,25), 
              cmap='RdYlBu_r')
    plt.contour(lon31g,lat31g, var,**kw)
    plt.contour(lon31g,lat31g, var,levels = 10,color ='black',cmap=plt.cm.jet)
    plt.contourf(lon31g,lat31g, var, **kw)
    
    okt = np.where(timeg > time31[ind])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'o',markeredgecolor='k',color='white')
    
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('$(^oC)$',rotation=270, labelpad=25, fontsize=12)
    
    file = folder + '{0}_{1}.png'.format('Bott_Temp_GOFS31_Yellow_Sea',\
                     mdates.num2date(time31[ind]))
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)    
    