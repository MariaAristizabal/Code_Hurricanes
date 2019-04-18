#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:40:34 2019

@author: aristizabal
"""

#%% User input

# RU33 (MAB + SAB)
lon_lim = [-81,-70]
lat_lim = [30,42]
gdata = 'https://data.ioos.us//thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc'
#gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180704T0000/ramses-20180704T0000.nc3.nc'
# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

#Time window
date_ini = '2018/09/11/00/00'
date_end = '2018/09/13/00/00'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

# url for GOFS 3.1
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import netCDF4 
import matplotlib.dates as mdates

import datetime

#%% Glider data
        
#dglider = xr.open_dataset(gdata,decode_times=False) 
dglider = Dataset(gdata,decode_times=False) 

inst_id = dglider.id.split('_')[0]
inst_name = inst_id.split('-')[0]  

latitude = dglider['latitude'][:] 
longitude = dglider['longitude'][:]

ttg = dglider['time']
tg = netCDF4.num2date(ttg[:],ttg.units) 

tmin = datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H/%M')
tmax = datetime.datetime.strptime(date_end,'%Y/%m/%d/%H/%M')
oktimeg = np.where(np.logical_and(tg >= tmin, tg <= tmax))

# Fiels within time window
timeg = tg[oktimeg]
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

#GOFS31 = xr.open_dataset(catalog31,decode_times=False)
GOFS31 = Dataset(catalog31,decode_times=False)
    
latt31 = GOFS31['lat'][:]
lonn31 = GOFS31['lon'][:]
depth31 = GOFS31['depth'][:]
tt31= GOFS31['time']
t31 = netCDF4.num2date(tt31[:],tt31.units) 

tmin = datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H/%M')
tmax = datetime.datetime.strptime(date_end,'%Y/%m/%d/%H/%M')

oktime31 = np.where(np.logical_and(t31 >= tmin, t31 <= tmax))
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

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
#oklatbath = oklatbath[:,np.newaxis]
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])
#oklonbath = oklonbath[:,np.newaxis]

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
#bath_elevsub = bath_elev[oklatbath,oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Figures surface salinity

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31[0]):
    print(i)
    S31 = GOFS31['salinity'][ind,0,botm:top,left:right]
    var = S31

    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
    plt.title('Salinity  \n GOFS 3.1 on {}'.format(time31[i]))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(28,38,11), 
              cmap=plt.cm.jet)
                  
    plt.contour(lon31g,lat31g, var,**kw)
    plt.contourf(lon31g,lat31g, var, **kw)

    #ax.plot(np.asarray(long),np.asarray(latg),'.k')
    okt = np.where(timeg > time31[i])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'*',color='k')
    
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('Salinity',rotation=270, labelpad=25, fontsize=12)
    
    plt.xlim(-77,-74)
    plt.ylim(34,37)  
    
    file = folder + '{0}_{1}.png'.format('Salt_MAB_SAB',\
                     time31[i])
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
    
#%%
  '''  
i=0
ind = oktime31[0][0]
S31 = GOFS31['salinity'][ind,0,botm:top,left:right]
var = S31

fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
#lot_date = mdates.num2date(time31[i])
plt.title('Salinity  \n GOFS 3.1 on {}'.format(time31[i]))
    
ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

max_v = np.nanmax(abs(var))
kw = dict(levels=np.linspace(24,38,15), 
           cmap=plt.cm.jet)

plt.contour(lon31g,lat31g, var,**kw)
plt.contourf(lon31g,lat31g, var, **kw)                

#ax.plot(np.asarray(long),np.asarray(latg),'.k',markersize=1)
okt = np.where(timeg > time31[i])
ax.plot(long[okt[0][0]],latg[okt[0][0]],'*',color='k')
    
#cb = plt.colorbar(format='%d')
cb = plt.colorbar()

plt.xlim(-77,-74)
plt.ylim(34,37)  
'''
        
#%% Figures temperature at 60 m

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31[0]):
    T31 = GOFS31['water_temp'][ind,7,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='w') 
    plt.title('Temperature  \n GOFS 3.1 on {}'.format(time31[i]))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(13,29,9), 
              cmap=plt.cm.jet)
    plt.contour(lon31g,lat31g, var,**kw)
    plt.contourf(lon31g,lat31g, var, **kw)

    okt = np.where(timeg > time31[i])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'*',markeredgecolor='k',color='k')
    print(long[okt[0][0]],latg[okt[0][0]])
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('$(^oC)$',rotation=270, labelpad=25, fontsize=12)   
    
    plt.xlim(-77,-74)
    plt.ylim(34,37) 
    
    file = folder + '{0}_{1}.png'.format('Temp_GOFS31_MAB_SAB',\
                     time31[i])
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)     

#%%
    i=0
    T31 = GOFS31['water_temp'][ind,7,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='w') 
    plt.title('Temperature  \n GOFS 3.1 on {}'.format(time31[i]))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(13,29,9), 
              cmap=plt.cm.jet)
    plt.contour(lon31g,lat31g, var,**kw)
    plt.contourf(lon31g,lat31g, var, **kw)

    okt = np.where(timeg > time31[i])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'*',markeredgecolor='k',color='k')
    
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('$(^oC)$',rotation=270, labelpad=25, fontsize=12)   
    
    plt.xlim(-77,-74)
    plt.ylim(34,37) 

#%% Figures temperature at surface
'''
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31):
    #time_index = time31[oktime31] ==  time31[oktime31][0]
    T31 = df.water_temp[ind,0,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
    plot_date = mdates.num2date(time31[ind])
    plt.title('Temperature  \n GOFS 3.1 on {}'.format(plot_date))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(20,27,8), 
              cmap=plt.cm.jet)
    plt.contour(lon31g,lat31g, var,**kw)
    plt.contourf(lon31g,lat31g, var, **kw)

    #ax.plot(np.asarray(long),np.asarray(latg),'.k')
    okt = np.where(timeg > time31[ind])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'o',markeredgecolor='k',color='white')
    
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('$(^oC)$',rotation=270, labelpad=25, fontsize=12)
    
    plt.xlim(-75,-70)
    plt.ylim(36,42) 
    
    file = folder + '{0}_{1}.png'.format('Temp_GOFS31_MAB',\
                     mdates.num2date(time31[ind]))
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0)     
'''

#%% Figures bottom temperature

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

for i,ind in enumerate(oktime31[0]):
    print(i,ind)
    #time_index = time31[oktime31] ==  time31[oktime31][0]
    T31 = GOFS31.variables['water_temp_bottom'][ind,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
    plt.title('Bottom Temperature  \n GOFS 3.1 on {}'.format(time31[i]))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(0,28,29), 
              cmap=plt.cm.jet)
    plt.contour(lon31g,lat31g, var,**kw)
    plt.contour(lon31g,lat31g, var,levels = 10,colors ='black',linewidths=4)
    plt.contourf(lon31g,lat31g, var, **kw)
    
    #ax.plot(np.asarray(long),np.asarray(latg),'.k')
    ax.plot(long,latg,'*k')
    okt = np.where(timeg > time31[i])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'o',markeredgecolor='k',color='white')
    
    plt.xlim(-77,-74)
    plt.ylim(34,36) 
    
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('$(^oC)$',rotation=270, labelpad=25, fontsize=12) 
    
    file = folder + '{0}_{1}.png'.format('Bott_Temp_GOFS31_MAB_SAB_RAMSES',\
                     time31[i])
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
    
#%%
    
    i=0
    ind=oktime31[0][i]
  
    T31 = GOFS31.variables['water_temp_bottom'][ind,botm:top,left:right]
    var = T31

    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=80, facecolor='w', edgecolor='w') 
    plt.title('Bottom Temperature  \n GOFS 3.1 on {}'.format(time31[i]))
    
    ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
    ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

    max_v = np.nanmax(abs(var))
    kw = dict(levels=np.linspace(0,28,29), 
              cmap=plt.cm.jet)
    plt.contour(lon31g,lat31g, var,**kw)
    plt.contour(lon31g,lat31g, var,levels = 10,colors ='black',linewidths=4)
    plt.contourf(lon31g,lat31g, var, **kw)
    
    #ax.plot(np.asarray(long),np.asarray(latg),'.k')
    ax.plot(long,latg,'*k')
    okt = np.where(timeg > time31[i])
    ax.plot(long[okt[0][0]],latg[okt[0][0]],'o',markeredgecolor='k',color='white')
    
    plt.xlim(-77,-74)
    plt.ylim(34,36) 
    
    #cb = plt.colorbar(format='%d')
    cb = plt.colorbar()
    cb.set_label('$(^oC)$',rotation=270, labelpad=25, fontsize=12) 
    
    file = folder + '{0}_{1}.png'.format('Bott_Temp_GOFS31_MAB_SAB_RAMSES',\
                     time31[i])
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
  
