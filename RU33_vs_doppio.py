#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:20:36 2019

@author: aristizabal
"""
#%% User input

#doppio_output_dir = '/Volumes/aristizabal/doppio_output/output_Aug_02_2018/'
#doppio_output_dir = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/'
doppio_output_dir = '/Volumes/aristizabal/doppio_output/output_Aug_02_Aug_07_2018/'
doppio_his = 'doppio_his.nc'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# date limits
date_ini = '2018/08/06/00'
date_end = '2018/08/07/00'

# Glider data 
url_glider = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc'

# folder to save figures
folder = '/Users/aristizabal/Desktop/4DVar_ROMS_Workshop/Doppio_figures/'

#%%
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy as np

import cmocean

import matplotlib.dates as mdates
import datetime

# Increase fontsize of labels globally 
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading glider data

gdata = xr.open_dataset(url_glider,decode_times=False)
    
inst_id = gdata.id.split('_')[0]
temperature = np.asarray(gdata.temperature[0][:])
salinity = np.asarray(gdata.salinity[0][:])
latitude = np.asarray(gdata.latitude[0])
longitude = np.asarray(gdata.longitude[0])
depth = np.asarray(gdata.depth[0])
    
time = gdata.time[0]
time = netCDF4.num2date(time,time.units)
tini = datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H')
tend= datetime.datetime.strptime(date_end,'%Y/%m/%d/%H')
oktimeg = np.logical_and(time >= tini,time <= tend)
        
# Fiels within time window
tempg =  temperature[oktimeg,:]
saltg =  salinity[oktimeg,:]
latg = latitude[oktimeg]
long = longitude[oktimeg]
depthg = depth[oktimeg,:]
timeg = time[oktimeg]

#%% Grid glider variables according to depth

depthg_gridded = np.arange(0,np.nanmax(depthg),0.5)
tempg_gridded = np.empty((len(timeg),len(depthg_gridded)))
tempg_gridded[:] = np.nan
saltg_gridded = np.empty((len(timeg),len(depthg_gridded)))
saltg_gridded[:] = np.nan

for t,tt in enumerate(timeg):
    depthu,oku = np.unique(depthg[t,:],return_index=True)
    tempu = tempg[t,oku]
    saltu = saltg[t,oku]
    okdd = np.isfinite(depthu)
    depth_fin = depthu[okdd]
    temp_fin = tempu[okdd]
    salt_fin = saltu[okdd]
    
    okt = np.isfinite(temp_fin)   
    if np.sum(okt) < 3:
        tempg_gridded[t,:] = np.nan
    else:
        okd = depthg_gridded < np.max(depth_fin[okt])
        tempg_gridded[t,okd] = np.interp(depthg_gridded[okd],depth_fin[okt],temp_fin[okt])
        
    oks = np.isfinite(salt_fin)   
    if np.sum(oks) < 3:
        saltg_gridded[t,:] = np.nan
    else:
        okd = depthg_gridded < np.max(depth_fin[oks])
        saltg_gridded[t,okd] = np.interp(depthg_gridded[okd],depth_fin[oks],salt_fin[oks])
    
        
#%% Get rid off of profiles with no data below 100 m

tempg_full = []
saltg_full = []
timeg_full = []
for t,tt in enumerate(timeg):
    okt = np.isfinite(tempg_gridded[t,:])
    oks = np.isfinite(saltg_gridded[t,:])
    if sum(depthg_gridded[okt] > 100) > 10:
        if tempg_gridded[t,0] != tempg_gridded[t,20]:
            tempg_full.append(tempg_gridded[t,:]) 
            timeg_full.append(tt) 
    if sum(depthg_gridded[oks] > 100) > 10:
        if saltg_gridded[t,0] != saltg_gridded[t,20]:
            saltg_full.append(saltg_gridded[t,:]) 
       
tempg_full = np.asarray(tempg_full)
saltg_full = np.asarray(saltg_full)
timeg_full = np.asarray(timeg_full)

#%% Read doppio output

doppio = xr.open_dataset(doppio_output_dir + doppio_his)

doppio_time = np.asarray(doppio.variables['ocean_time'][:])
doppio_lon_rho = np.asarray(doppio.variables['lon_rho'][:])
doppio_lat_rho = np.asarray(doppio.variables['lat_rho'][:])
doppio_lon_u = np.asarray(doppio.variables['lon_u'][:])
doppio_lat_u = np.asarray(doppio.variables['lat_u'][:])
doppio_lon_v = np.asarray(doppio.variables['lon_v'][:])
doppio_lat_v = np.asarray(doppio.variables['lat_v'][:])
doppio_s_rho = np.asarray(doppio.variables['s_rho'][:])
doppio_s_w = np.asarray(doppio.variables['s_w'][:])
#doppio_h = np.asarray(doppio.variables['h'][:])
#doppio_temp = np.asarray(doppio.variables['temp'][:])
#doppio_salt = np.asarray(doppio.variables['salt'][:])
#doppio_u = np.asarray(doppio.variables['u'][:])
#doppio_v = np.asarray(doppio.variables['v'][:])

doppio_oktime = np.where(np.logical_and(mdates.date2num(doppio_time) >= mdates.date2num(tini),\
                                        mdates.date2num(doppio_time) <= mdates.date2num(tend)))
doppio_subtime = doppio_time[doppio_oktime]

#%% Finding glider track in model

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
tstamp_model = [mdates.date2num(doppio_subtime[i]) for i in np.arange(len(doppio_subtime))]

# interpolating glider lon and lat to lat and lon on model time
sublonm = np.interp(tstamp_model,tstamp_glider,long)
sublatm = np.interp(tstamp_model,tstamp_glider,latg)

# getting the model grid positions for sublonm and sublatm
oklatm = np.empty((len(doppio_oktime[0])))
oklatm[:] = np.nan
oklonm = np.empty((len(doppio_oktime[0])))
oklonm[:] = np.nan
for t,tt in enumerate(doppio_oktime[0]):
    oklatmm = []
    oklonmm = []
    for pos_xi in np.arange(doppio_lat_rho.shape[1]):
        pos_eta = np.round(np.interp(sublatm[t],doppio_lat_rho[:,pos_xi],np.arange(len(doppio_lat_rho[:,pos_xi])),\
                                      left=np.nan,right=np.nan))
        if np.isfinite(pos_eta):
            oklatmm.append((pos_eta).astype(int))
            oklonmm.append(pos_xi)
            
    pos = np.round(np.interp(sublonm[t],doppio_lon_rho[oklatmm,oklonmm],np.arange(len(doppio_lon_rho[oklatmm,oklonmm])))).astype(int)    
    oklatm[t] = oklatmm[pos]
    oklonm[t] = oklonmm[pos]      

oklatm = oklatm.astype(int)
oklonm = oklonm.astype(int)

#%%       
# Getting glider transect from model
target_doppio_temp = np.empty((len(doppio_s_rho),len(doppio_oktime[0])))
target_doppio_temp[:] = np.nan
target_doppio_salt = np.empty((len(doppio_s_rho),len(doppio_oktime[0])))
target_doppio_salt[:] = np.nan
target_doppio_h = np.empty((len(doppio_oktime[0])))
target_doppio_h[:] = np.nan
for i in range(len(doppio_oktime[0])):
    print(len(doppio_oktime[0]),' ',i)
    target_doppio_temp[:,i] = doppio.variables['temp'][doppio_oktime[0][i],:,oklatm[i],oklonm[i]]
    target_doppio_salt[:,i] = doppio.variables['salt'][doppio_oktime[0][i],:,oklatm[i],oklonm[i]]
    target_doppio_h[i] = doppio.variables['h'][oklatm[i],oklonm[i]]

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= np.min(doppio_lat_rho),bath_lat <= np.max(doppio_lat_rho))
oklonbath = np.logical_and(bath_lon >= np.min(doppio_lon_rho),bath_lon <= np.max(doppio_lon_rho))

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath] 

#%% 

target_doppio_depth = np.matmul(doppio_s_rho.reshape(-1,1),target_doppio_h.reshape(1,-1))
target_doppio_time = np.tile(doppio_subtime,(len(doppio_s_rho),1))

#%%
fig,ax = plt.subplots()
plt.contourf(target_doppio_time,target_doppio_depth,target_doppio_temp,cmap=cmocean.cm.thermal)
plt.colorbar()
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

#%% Figure temperature

okg = depthg_gridded <= np.max(depthg_gridded) 
okm = target_doppio_depth <= np.max(depthg_gridded) 
min_val = np.floor(np.min([np.nanmin(tempg_gridded[:,okg]),np.nanmin(target_doppio_temp[okm])]))
max_val = np.ceil(np.max([np.nanmax(tempg_gridded[:,okg]),np.nanmax(target_doppio_temp[okm])]))
    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))

fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.subplot(211)        
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded.T,cmap=cmocean.cm.thermal,**kw)
plt.contour(timeg,-depthg_gridded,tempg_gridded.T,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('$^oC$',fontsize=14,labelpad=15)
#ax.set_xlim(df.index[0], df.index[-1])
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xticklabels(' ')
plt.title('Along Track Temperature' + ' Profile ' + inst_id.split('-')[0],fontsize=20)
    
ax = plt.subplot(212)        
plt.contour(target_doppio_time,target_doppio_depth,target_doppio_temp,colors = 'lightgrey')
cs = plt.contourf(target_doppio_time,target_doppio_depth,target_doppio_temp,cmap=cmocean.cm.thermal,**kw)
plt.contour(target_doppio_time,target_doppio_depth,target_doppio_temp,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('$^oC$',fontsize=14,labelpad=15)
#ax.set_xlim(df.index[0], df.index[-1])
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Along Track Profile Doppio',fontsize=20) 

file = folder + 'temp_transct_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%% Figure salinity

okg = depthg_gridded <= np.max(depthg_gridded) 
okm = target_doppio_depth <= np.max(depthg_gridded) 
min_val = np.floor(np.min([np.nanmin(saltg_gridded[:,okg]),np.nanmin(target_doppio_salt[okm])]))
max_val = np.ceil(np.max([np.nanmax(saltg_gridded[:,okg]),np.nanmax(target_doppio_salt[okm])]))
    
kw = dict(levels = np.arange(min_val,max_val+0.25,0.25))

fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.subplot(211)        
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,saltg_gridded.T,cmap=cmocean.cm.haline,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xticklabels(' ')
plt.title('Along Track Salinity' + ' Profile ' + inst_id.split('-')[0],fontsize=20)
    
ax = plt.subplot(212)        
plt.contour(target_doppio_time,target_doppio_depth,target_doppio_salt,colors = 'lightgrey')
cs = plt.contourf(target_doppio_time,target_doppio_depth,target_doppio_salt,cmap=cmocean.cm.haline,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
ax.set_ylim(-np.max(depthg_gridded), 0)
ax.set_ylabel('Depth (m)',fontsize=14)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.title('Along Track Profile Doppio',fontsize=20) 

file = folder + 'salt_transct_' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)    