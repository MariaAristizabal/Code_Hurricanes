#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 09:50:07 2019

@author: aristizabal
"""

#%% User input

# Glider data url address

'''
#Gulf of Mexico
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';
# lat and lon of area
lon_lim = [-100,-80]
lat_lim = [  18, 32]
# Time window
date_ini = '2018/10/7/00/00'
date_end = '2018/10/13/00/00'
# time of hurricane passage
thurr = '2018/10/10/06/00'
'''

'''
# MAB + SAB
# RU33 (MAB + SAB)
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';
# lat and lon of area
lon_lim = [-81,-70]
lat_lim = [30,42]
#Time window
date_ini = '2018/09/06/00/00'
date_end = '2018/09/15/00/00'
# time of hurricane passage
thurr = '2018/09/14/18/00'
'''

# Caribbean
gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc'
# lat and lon of area
lon_lim = [-68,-64]
lat_lim = [15,20]
#Time window
date_ini = '2018/09/06/00/00'
date_end = '2018/09/15/00/00'
# time of hurricane passage
thurr = '2018/09/14/18/00'

# url for GOFS 3.1
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# Bathymetry data
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# In[1]:

from matplotlib import pyplot as plt
#import cmocean
import numpy as np
import xarray as xr
import matplotlib.dates as mdates
import datetime

#plt.style.use('seaborn-poster')
#plt.style.use('ggplot')

import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#%%

def dens0(s, t):
    s, t = list(map(np.asanyarray, (s, t)))
    T68 = T68conv(t)
    # UNESCO 1983 Eqn.(13) p17.
    b = (8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9)
    c = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
    d = 4.8314e-4
    return (smow(t) + (b[0] + (b[1] + (b[2] + (b[3] + b[4] * T68) * T68) *
            T68) * T68) * s + (c[0] + (c[1] + c[2] * T68) * T68) * s * s ** 0.5 + d * s ** 2)

def smow(t):
    t = np.asanyarray(t)
    a = (999.842594, 6.793952e-2, -9.095290e-3, 1.001685e-4, -1.120083e-6, 6.536332e-9)
    T68 = T68conv(t)
    return (a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * T68) * T68) * T68) * T68) * T68)
    
def T68conv(T90):
    T90 = np.asanyarray(T90)
    return T90 * 1.00024
    
#%% Glider data
        
dglider = xr.open_dataset(gdata,decode_times=False) 

inst_id = dglider.id.split('_')[0]
inst_name = inst_id.split('-')[0]  

latitude = dglider.latitude[0] 
longitude = dglider.longitude[0]
temperature = dglider.temperature[0]
salinity = dglider.salinity[0]
density = dglider.density[0]
depth = dglider.depth[0]

## Change time into standardized mdates datenums 
seconds_since1970 = dglider.time[0]
timei = datetime.datetime.strptime(dglider.time.time_origin,'%d-%b-%Y %H:%M:%S')
timei + datetime.timedelta(seconds=int(seconds_since1970[0]))
time = np.empty(len(seconds_since1970))
for ind, hrs in enumerate(seconds_since1970):
    time[ind] = mdates.date2num(timei + datetime.timedelta(seconds=int(hrs)))

# Find time window of interest
tti = mdates.date2num(datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H/%M'))     
tte = mdates.date2num(datetime.datetime.strptime(date_end,'%Y/%m/%d/%H/%M')) 
oktimeg = np.logical_and(time >= tti,time <= tte)

# Fiels within time window
timeg = time[oktimeg]
latg = latitude[oktimeg]
long = longitude[oktimeg]
tempg =  temperature[oktimeg,:]
saltg = salinity[oktimeg,:]
densg = density[oktimeg,:]
depthg = depth[oktimeg,:]

# Change glider lot and lat to GOFS 3.1 convention
target_lon = np.empty((len(long),))
target_lon[:] = np.nan
for i in range(len(long)):
    if long[i] < 0: 
        target_lon[i] = 360 + long[i]
    else:
        target_lon[i] = long[i]
target_lat = latg

#%% GOFS 3.1

df = xr.open_dataset(catalog31,decode_times=False)

## Decode the GOFS3.1 time into standardized mdates datenums 
hours_since2000 = df.time
time_naut       = datetime.datetime(2000,1,1)
time31 = np.ones_like(hours_since2000)
for ind, hrs in enumerate(hours_since2000):
    time31[ind] = mdates.date2num(time_naut+datetime.timedelta(hours=int(hrs)))
    
## Find the dates of import
dini = mdates.date2num(datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H/%M')) # October 7, 2018
dend = mdates.date2num(datetime.datetime.strptime(date_end,'%Y/%m/%d/%H/%M')) # October 11, 2018
formed  = int(np.where(time31 == dini)[0][0])
dissip  = int(np.where(time31 == dend)[0][0])
oktime31 = np.arange(formed,dissip+1,dtype=int)

lat31 = df.lat
lon31 = df.lon
depth31 = df.depth

### Build the bbox for the xy data
botm  = int(np.where(df.lat > lat_lim[0])[0][0])
top   = int(np.where(df.lat > lat_lim[1])[0][0])
half  = int(len(df.lon)/2)

left  = np.where(df.lon > lon_lim[0]+360)[0][0]
right = np.where(df.lon > lon_lim[1]+360)[0][0]
lat100= df.lat[botm:top]
lon100= df.lon[left:right]
X, Y = np.meshgrid(lon100,lat100)

#%%

# interpolating glider lon and lat to lat and lon on model time
sublon31 = np.interp(time31,timeg,target_lon)
sublat31 = np.interp(time31,timeg,target_lat)

# getting the model grid positions for sublon31 and sublat31
oklon31 = np.round(np.interp(sublon31,lon31,np.arange(len(lon31)))).astype(int)
oklat31 = np.round(np.interp(sublat31,lat31,np.arange(len(lat31)))).astype(int)

#%% Get glider transect from model

target_temp31 = np.empty((len(depth31),len(oktime31)))
target_temp31[:] = np.nan
target_salt31 = np.empty((len(depth31),len(oktime31)))
target_salt31[:] = np.nan
for i in range(len(oktime31)):
    target_temp31[:,i] = df.water_temp[oktime31[i],:,oklat31[i],oklon31[i]]
    target_salt31[:,i] = df.salinity[oktime31[i],:,oklat31[i],oklon31[i]]

target_temp31[target_temp31 < -100] = np.nan

#%% Surface Heat content for GOFS 3.1

# Heat capacity in J/(kg K)
cp = 3985 

## set a date with the time_index
#time_index = time31[michael] ==  mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0))
time_index = time31[oktime31] ==  mdates.date2num(datetime.datetime(2018, 9, 13, 0, 0))

T31 = df.water_temp[oktime31[time_index][0],:,botm:top,left:right]
S31 = df.salinity[oktime31[time_index][0],:,botm:top,left:right]
D31 = dens0(S31,T31)

OHC = np.empty((len(np.arange(botm,top)),len(np.arange(left,right))))
OHC[:] = np.nan
for j, index in enumerate(np.arange(left,right)):
    for i, index in enumerate(np.arange(botm,top)):
        print(i,' ' ,j)
        ok26 = T31[:,i,j] >= 26
        rho0 = np.nanmean(D31[ok26,i,j])
        OHC[i,j] = cp * rho0 * np.trapz(T31[ok26,i,j]-26,df.depth[ok26])
        
#%%  T100 for GOFS 3.1
        
dindex = np.where(depth31 <= 100)[0]        
        
## set a date with the time_index
#time_index = time31[michael] ==  mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0))
time_index = time31[oktime31] ==  mdates.date2num(datetime.datetime(2018, 9, 13, 0, 0))
T100 = df.water_temp[oktime31[time_index][0],dindex,botm:top,left:right]
T100m = np.nanmean(T100,axis=0)

#%%  S100 for GOFS 3.1

#time_index = time31[michael] ==  mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0))
time_index = time31[oktime31] ==  mdates.date2num(datetime.datetime(2018, 9, 13, 0, 0))
S100 = df.salinity[oktime31[time_index][0],dindex,botm:top,left:right]
S100m = np.nanmean(S100,axis=0)

#%% D100 for GOFS 3.1

#time_index = time31[michael] ==  mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0))
time_index = time31[oktime31] ==  mdates.date2num(datetime.datetime(2018, 9, 13, 0, 0))
D100  = dens0(S100,T100)
D100m = np.nanmean(D100,axis=0)

#%%
fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())
plot_date = mdates.num2date(time31[oktime31[time_index][0]])
plt.title('T_100  \n GOFS 3.1 on {}'.format(plot_date))

var = T100m

max_v = np.nanmax(abs(var))
kw = dict(levels=np.linspace(np.nanmin(var),np.nanmax(var),20), 
          cmap=plt.cm.Spectral_r,
          transform=ccrs.PlateCarree())

#plt.contourf( var)
plt.contourf(lon100,lat100, var, **kw)
#cb = plt.colorbar(format='%d')
cb = plt.colorbar()
#cb = plt.colorbar()
cb.set_label('Temperature (C)',rotation=270, labelpad=25, fontsize=12)

### High resolution coastline is set by resolution='10m'
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='gray', alpha=0.1)

# Michael path
#for x in range(0, len(tMc)-1, 2):
#    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = 12)
    
# glider positions
#okg = np.where(timeg>mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0)))
#ax.plot(long[okg[0][0]],latg[okg[0][0]],'*',color='k',markersize=20) 

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/T100_GOFS31_Caribb.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%
fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())
plot_date = mdates.num2date(time31[oktime31[time_index][0]])
plt.title('OHC  \n GOFS 3.1 on {}'.format(plot_date))

var = OHC

max_v = np.nanmax(abs(var))
kw = dict(levels=np.linspace(np.nanmin(var),np.nanmax(var),20), 
          cmap=plt.cm.Spectral_r,
          transform=ccrs.PlateCarree())

#plt.contourf( var)
plt.contourf(lon100,lat100, var, **kw)
cb = plt.colorbar(format='%.1e')
#cb = plt.colorbar()
cb.set_label('OHC (j/m^2)',rotation=270, labelpad=25, fontsize=12)

### High resolution coastline is set by resolution='10m'
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='gray', alpha=0.1)
#plt.ylim(30,42)

# Michael path
#for x in range(0, len(tMc)-1, 2):
#    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = 12)
    
# glider positions
#okg = np.where(timeg>mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0)))
#ax.plot(long[okg[0][0]],latg[okg[0][0]],'*',color='k',markersize=20) 

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/OHC_GOFS31_Caribb.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%
fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())
plot_date = mdates.num2date(time31[oktime31[time_index][0]])
plt.title('S_100  \n GOFS 3.1 on {}'.format(plot_date))

var = S100m
#rivers = S100m < 34
#hypers = S100m > 37
#var[rivers] = np.nan
#var[hypers] = np.nan

rivers = S100m < 28
var[rivers] = np.nan

max_v = np.nanmax(abs(var))
kw = dict(levels=np.linspace(np.nanmin(var),np.nanmax(var),20), 
          cmap=plt.cm.Spectral_r,
          transform=ccrs.PlateCarree())

#plt.contourf( var)
plt.contourf(lon100,lat100, var, **kw)
#cb = plt.colorbar(format='%.1e')
cb = plt.colorbar()
cb.set_label('Salinity',rotation=270, labelpad=25, fontsize=12)

### High resolution coastline is set by resolution='10m'
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='gray', alpha=0.1)

# Michael path
#ax.plot(lonMc,latMc,'o-',markersize = 10,label = 'Michael Track',color = 'dimgray')
#for x in range(0, len(tMc)-1, 2):
#    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = 12)
    
# glider positions
#okg = np.where(timeg>mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0)))
#ax.plot(long[okg[0][0]],latg[okg[0][0]],'*',color='k',markersize=20) 

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/S100_GOFS31_caribb.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%
fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())
plot_date = mdates.num2date(time31[oktime31[time_index][0]])
plt.title('D_100  \n GOFS 3.1 on {}'.format(plot_date))

var = D100m
rivers = D100m < 1020
#hypers = S100 > 37
var[rivers] = np.nan
#var[hypers] = np.nan

max_v = np.nanmax(abs(var))
kw = dict(levels=np.linspace(np.nanmin(var),np.nanmax(var),20), 
          cmap=plt.cm.Spectral_r,
          transform=ccrs.PlateCarree())

#plt.contourf( var)
plt.contourf(lon100,lat100, var, **kw)
#cb = plt.colorbar(format='%.1e')
cb = plt.colorbar()
cb.set_label('Density (kg/m^3)',rotation=270, labelpad=25, fontsize=12)

### High resolution coastline is set by resolution='10m'
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='gray', alpha=0.1)

# Michael path
#ax.plot(lonMc,latMc,'o-',markersize = 10,label = 'Michael Track',color = 'dimgray')
#for x in range(0, len(tMc)-1, 2):
#    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = 12)
    
# glider positions
#okg = np.where(timeg>mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0)))
#ax.plot(long[okg[0][0]],latg[okg[0][0]],'*',color='k',markersize=20) 

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/D100_GOFS31_Caribb.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 