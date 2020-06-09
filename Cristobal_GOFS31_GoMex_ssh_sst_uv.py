#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:11:41 2020

@author: aristizabal
"""
#%% User input

#GOFS3.1 output model location
url_GOFS_ts = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'

url_GOFS_uv = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'

url_GOFS_ssh = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ssh'
#https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ssh

url_GOFS_forecast = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'
# Bathymetry file
#bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# Gulf Mexico
lon_lim = [-100,-75]
lat_lim = [14,33]

# Jun1 -Jul1
#Dir_Argo = '/Users/Aristizabal/Desktop/DataSelection_20200604_190743_9994922/'
#Argo_nc = '/Users/Aristizabal/Desktop/ArgoFloats_48b0_9f3e_31b2.nc'
Argo_nc = '/Users/Aristizabal/Desktop/ArgoFloats_eb1e_1977_1b02.nc'

folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 
from datetime import datetime
import cmocean
from netCDF4 import Dataset

#%% GOGF 3.1

GOFS_ts = xr.open_dataset(url_GOFS_ts,decode_times=False)
GOFS_uv = xr.open_dataset(url_GOFS_uv,decode_times=False)
GOFS_ssh = xr.open_dataset(url_GOFS_ssh,decode_times=False)
GOFS_forecast =xr.open_dataset(url_GOFS_forecast,decode_times=False)

latt31 = np.asarray(GOFS_ts['lat'][:])
lonn31 = np.asarray(GOFS_ts['lon'][:])
tt31 = GOFS_ts['time']
t_GOFS = netCDF4.num2date(tt31[:],tt31.units) 

depth_GOFS = np.asarray(GOFS_ts['depth'][:])

tt = GOFS_forecast['time']
time_forec = netCDF4.num2date(tt[:],tt.units) 

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
botm  = int(np.where(latt31 > lat_limG[0])[0][0])
top   = int(np.where(latt31 > lat_limG[1])[0][0])
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

#%% Reading Argo data

ncargo = Dataset(Argo_nc)
argo_lat = np.asarray(ncargo.variables['latitude'][:])
argo_lon = np.asarray(ncargo.variables['longitude'][:])
argo_tim = ncargo.variables['time']#[:]
argo_time = netCDF4.num2date(argo_tim[:],argo_tim.units)  
argo_pres = np.asarray(ncargo.variables['pres'][:])
argo_temp = np.asarray(ncargo.variables['temp'][:])
argo_salt = np.asarray(ncargo.variables['psal'][:])
argo_id = ncargo.variables['platform_number'][:]

#%% Map Argo floats
 
lev = np.arange(-9000,9100,100)
plt.figure()
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo) 
#plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.plot(argo_lon,argo_lat,'s',color='g',markersize=3,markeredgecolor='k')
plt.title('Argo Floats ' + str(argo_time[0])[0:10]+'-'+str(argo_time[-1])[0:10],fontsize=14)
plt.axis('scaled')
plt.xlim(-98,-79.5)
plt.ylim(15,32.5)

file = folder_fig + 'ARGO_lat_lon_' + str(argo_time[0])[0:10]+'-'+str(argo_time[-1])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% GOFS  3.1
#t = datetime.strptime(date_enterGoM,'%Y/%m/%d/%H/%M')
#t = datetime.strptime(date_midGoM,'%Y/%m/%d/%H/%M')
t = datetime(2020,6,4,9)
oktime_GOFS = np.where(t_GOFS == t)[0][0]
time_GOFS = t_GOFS[oktime_GOFS]

# loading surface temperature and salinity
sst_GOFS = GOFS_ts['water_temp'][oktime_GOFS,0,botm:top,left:right]
sss_GOFS = GOFS_ts['salinity'][oktime_GOFS,0,botm:top,left:right]
ssh_GOFS = GOFS_ssh['surf_el'][oktime_GOFS,botm:top,left:right]
su_GOFS = GOFS_uv['water_u'][oktime_GOFS,0,botm:top,left:right]
sv_GOFS = GOFS_uv['water_v'][oktime_GOFS,0,botm:top,left:right]

#%% Figure argo float vs GOFS

oklon = np.where(np.logical_and(argo_lon > -90.5,\
                argo_lon < -89.5))

argo_ids = argo_id[oklon]   
argo_lons = argo_lon[oklon]
argo_lats = argo_lat[oklon]
argo_tts = argo_time[oklon]
argo_depths = argo_pres[oklon]
argo_temps = argo_temp[oklon]
argo_salts = argo_salt[oklon]

oklat = np.where(np.logical_and(argo_lats > 26.5,\
                argo_lats < 27))

argo_idss = argo_ids[oklat] 
#id_uniq, ind = np.unique(argo_id[oklon],return_index=True)
argo_idss = argo_ids[oklat]   
argo_lonss = argo_lons[oklat]
argo_latss = argo_lats[oklat]
argo_ttss = argo_tts[oklat]
argo_depthss = argo_depths[oklat]
argo_tempss = argo_temps[oklat]
argo_saltss = argo_salts[oklat]

oktt_GOFS = np.where(t_GOFS >= argo_ttss[0])[0][0]
oklat_GOFS = np.where(latt31 >= argo_latss[0])[0][0]
oklon_GOFS = np.where(lonn31 >= argo_lonss[0]+360)[0][0]
temp_GOFS = np.asarray(GOFS_ts['water_temp'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])
salt_GOFS = np.asarray(GOFS_ts['salinity'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])

# Figure temp
plt.figure(figsize=(5,6))
plt.plot(argo_tempss,-argo_depthss,'.-',linewidth=2,label='ARGO Float id'+argo_idss[0])
plt.plot(temp_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1')
#plt.ylim([-1000,0])
#plt.xlim([5,28])
plt.ylim([-200,0])
plt.xlim([15,28])
plt.title('Temperature Profile on '+ str(t_GOFS[oktt_GOFS])[0:13] +
          '\n [lon,lat] = [' \
          + str(np.round(argo_lonss[0],3)) +',' +\
              str(np.round(argo_latss[0],3))+']',\
              fontsize=16)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('$^oC$',fontsize=14)
plt.legend(loc='lower right',fontsize=14)

file = folder_fig + 'ARGO_vs_GOFS_' + str(t_GOFS[oktt_GOFS])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

# Figure salt
plt.figure(figsize=(5,6))
plt.plot(argo_saltss,-argo_depthss,'.-',linewidth=2,label='ARGO Float id'+argo_idss[0])
plt.plot(salt_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1')
plt.ylim([-1000,0])
#plt.xlim([5,28])
#plt.ylim([-200,0])
#plt.xlim([36,37])
plt.title('Salinity Profile on '+ str(t_GOFS[oktt_GOFS])[0:13] +
          '\n [lon,lat] = [' \
          + str(np.round(argo_lonss[0],3)) +',' +\
              str(np.round(argo_latss[0],3))+']',\
              fontsize=16)
plt.ylabel('Depth (m)',fontsize=14)
plt.legend(loc='lower right',fontsize=12)

file = folder_fig + 'salt_ARGO_vs_GOFS_' + str(t_GOFS[oktt_GOFS])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure argo float vs GOFS

oklon = np.where(np.logical_and(argo_lon > -90.5,\
                argo_lon < -89.5))

argo_ids = argo_id[oklon]   
argo_lons = argo_lon[oklon]
argo_lats = argo_lat[oklon]
argo_tts = argo_time[oklon]
argo_depths = argo_pres[oklon]
argo_temps = argo_temp[oklon]
argo_salts = argo_salt[oklon]

oklat = np.where(np.logical_and(argo_lats > 24.8,\
                argo_lats < 25))

argo_idss = argo_ids[oklat] 
#id_uniq, ind = np.unique(argo_id[oklon],return_index=True)
argo_idss = argo_ids[oklat]   
argo_lonss = argo_lons[oklat]
argo_latss = argo_lats[oklat]
argo_ttss = argo_tts[oklat]
argo_depthss = argo_depths[oklat]
argo_tempss = argo_temps[oklat]
argo_saltss = argo_salts[oklat]

oktt_GOFS = np.where(t_GOFS >= argo_ttss[0])[0][0]
oklat_GOFS = np.where(latt31 >= argo_latss[0])[0][0]
oklon_GOFS = np.where(lonn31 >= argo_lonss[0]+360)[0][0]
temp_GOFS = np.asarray(GOFS_ts['water_temp'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])
salt_GOFS = np.asarray(GOFS_ts['salinity'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])

# Figure temp
plt.figure(figsize=(5,6))
plt.plot(argo_tempss,-argo_depthss,'.-',linewidth=2,label='ARGO Float id'+argo_idss[0])
plt.plot(temp_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1')
#plt.ylim([-1000,0])
#plt.xlim([5,28])
plt.ylim([-200,0])
plt.xlim([15,28])
plt.title('Temperature Profile on '+ str(t_GOFS[oktt_GOFS])[0:13] +
          '\n [lon,lat] = [' \
          + str(np.round(argo_lonss[0],3)) +',' +\
              str(np.round(argo_latss[0],3))+']',\
              fontsize=16)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('$^oC$',fontsize=14)
plt.legend(loc='lower right',fontsize=14)

file = folder_fig + 'ARGO_vs_GOFS_' + str(t_GOFS[oktt_GOFS])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

# Figure salt
plt.figure(figsize=(5,6))
plt.plot(argo_saltss,-argo_depthss,'.-',linewidth=2,label='ARGO Float id'+argo_idss[0])
plt.plot(salt_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1')
#plt.ylim([-1000,0])
#plt.xlim([5,28])
plt.ylim([-200,0])
plt.xlim([36,37])
plt.title('Salinity Profile on '+ str(t_GOFS[oktt_GOFS])[0:13] +
          '\n [lon,lat] = [' \
          + str(np.round(argo_lonss[0],3)) +',' +\
              str(np.round(argo_latss[0],3))+']',\
              fontsize=16)
plt.ylabel('Depth (m)',fontsize=14)
plt.legend(loc='lower right',fontsize=12)

file = folder_fig + 'salt_ARGO_vs_GOFS_' + str(t_GOFS[oktt_GOFS])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 



#%% Figure sst

# GOFS 3.1
t = datetime(2020,6,4,9)
oktime_GOFS = np.where(t_GOFS == t)[0][0]
time_GOFS = t_GOFS[oktime_GOFS]

# loading surface temperature and salinity
sst_GOFS = GOFS_ts['water_temp'][oktime_GOFS,0,botm:top,left:right]
sss_GOFS = GOFS_ts['salinity'][oktime_GOFS,0,botm:top,left:right]
ssh_GOFS = GOFS_ssh['surf_el'][oktime_GOFS,botm:top,left:right]
su_GOFS = GOFS_uv['water_u'][oktime_GOFS,0,botm:top,left:right]
sv_GOFS = GOFS_uv['water_v'][oktime_GOFS,0,botm:top,left:right]

kw = dict(levels = np.linspace(24,30,16))

#plt.figure(figsize=(10, 8))
plt.figure()
plt.contourf(lon31g,lat31g,sst_GOFS[:,:],cmap=cmocean.cm.thermal,**kw)
plt.plot(np.tile(-90,len(lat31g)),lat31g,'-',color='k')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
plt.axis('equal')
plt.xlim(-98,-79.5)
plt.ylim(15,32.5)
#plt.title('GOFS 3.1 SST and surface velocity on '+str(time_GOFS)[0:13],size=22,y=1.03,fontsize=14)
plt.title('GOFS 3.1 SST on '+str(time_GOFS)[0:13],size=22,y=1.03,fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(np.arange(15,33,2.5),fontsize=12)

plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')

#q=plt.quiver(lon31g[::5],lat31g[::5],su_GOFS[::5,::5],sv_GOFS[::5,::5] ,scale=3,scale_units='inches',\
#           alpha=0.7)
#plt.quiverkey(q,-96.5,30.5,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})

file = folder_fig + 'GOFS_SST_GoMex_' + str(time_GOFS)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure sss

kw = dict(levels = np.linspace(32,37,11))

plt.figure(figsize=(10, 8))
plt.contourf(lon31g,lat31g,sss_GOFS[:,:],cmap=cmocean.cm.haline,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.axis('equal')
plt.xlim(-98,-79.5)
plt.ylim(15,32.5)
plt.title('GOFS 3.1 SSS and surface velocity on '+str(time_GOFS)[0:13],size=22,y=1.03)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')

#plt.quiver(lon31g[::4],lat31g[::4],su_GOFS[::4,::4],sv_GOFS[::4,::4] ,scale=3,scale_units='inches',\
#           alpha=0.7)

file = folder_fig + 'GOFS_SSH_GoMex_' + str(time_GOFS)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure ssh

kw = dict(levels = np.linspace(-0.6,0.6,25))

plt.figure(figsize=(10, 8))
plt.contourf(lon31g,lat31g,ssh_GOFS[:,:],cmap=cmocean.cm.curl,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.axis('equal')
plt.xlim(-98,-79.5)
plt.ylim(15,32.5)
plt.title('GOFS 3.1 SSH and surface velocity on '+str(time_GOFS)[0:13],size=22,y=1.03)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
cbar.ax.set_ylabel('meters',fontsize=16) 

plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')

plt.quiver(lon31g[::5],lat31g[::5],su_GOFS[::5,::5],sv_GOFS[::5,::5] ,scale=3,scale_units='inches',\
           alpha=0.7)

file = folder_fig + 'GOFS_SSH_GoMex_' + str(time_GOFS)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure sst at 200 meters

kw = dict(levels = np.linspace(10,25,31))
okdepth = np.where(depth_GOFS >= 200)[0][0]
temp_200_GOFS = np.asarray(GOFS_ts['water_temp'][oktime_GOFS,okdepth,botm:top,left:right])
su_GOFS = GOFS_uv['water_u'][oktime_GOFS,okdepth,botm:top,left:right]
sv_GOFS = GOFS_uv['water_v'][oktime_GOFS,okdepth,botm:top,left:right]

plt.figure(figsize=(10, 8))
plt.contourf(lon31g,lat31g,temp_200_GOFS,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.axis('equal')
plt.xlim(-98,-79.5)
plt.ylim(15,32.5)
plt.title('GOFS 3.1 Temp and Velocity at 200 m on '+str(time_GOFS)[0:13],size=22,y=1.03)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')

plt.quiver(lon31g[::5],lat31g[::5],su_GOFS[::5,::5],sv_GOFS[::5,::5] ,scale=3,scale_units='inches',\
           alpha=0.7)

file = folder_fig + 'GOFS_SST_GoMex_' + str(time_GOFS)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure salinity at 200 meters

kw = dict(levels = np.linspace(10,25,31))
okdepth = np.where(depth_GOFS >= 200)[0][0]
salt_200_GOFS = np.asarray(GOFS_ts['salinity'][oktime_GOFS,okdepth,botm:top,left:right])
su_GOFS = GOFS_uv['water_u'][oktime_GOFS,okdepth,botm:top,left:right]
sv_GOFS = GOFS_uv['water_v'][oktime_GOFS,okdepth,botm:top,left:right]

plt.figure(figsize=(10, 8))
plt.contourf(lon31g,lat31g,salt_200_GOFS,cmap=cmocean.cm.haline)#,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.axis('equal')
plt.xlim(-98,-79.5)
plt.ylim(15,32.5)
plt.title('GOFS 3.1 Temp and Velocity at 200 m on '+str(time_GOFS)[0:13],size=22,y=1.03)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')

#plt.quiver(lon31g[::5],lat31g[::5],su_GOFS[::5,::5],sv_GOFS[::5,::5] ,scale=3,scale_units='inches',\
#           alpha=0.7)

file = folder_fig + 'GOFS_salt_200_GoMex_' + str(time_GOFS)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure temp transect along Cristobal path

oklon = np.where(lonn31>=-90+360)[0][0]
temp_GOFS = GOFS_ts['water_temp'][oktime_GOFS,:,botm:top,oklon]
#%%

kw = dict(levels = np.linspace(12,32,21))
plt.figure()
plt.contourf(lat31,-depth_GOFS,temp_GOFS,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
plt.contour(lat31,-depth_GOFS,temp_GOFS,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.title('Temperature along Cristobal Path',fontsize=16)
plt.ylim([-200,0])
plt.xlim([20,30])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

file = folder_fig + 'GOFS_temp_along_Cristobal_' + str(time_GOFS)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)