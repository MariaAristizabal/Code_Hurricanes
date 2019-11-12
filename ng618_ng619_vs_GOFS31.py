#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:13:30 2019

@author: aristizabal
"""

#%% User input

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

# date limits
#date_ini = '2018-10-05T00:00:00Z'
#date_end = '2018-10-13T00:00:00Z'

lon_lim = [-78.0,-69.0]
lat_lim = [35.0,45.0]

# Glider data 
# ng618
#gdata = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng618-20190901T0000/ng618-20190901T0000.nc3.nc'

# ng619
gdata = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng619-20190901T0000/ng619-20190901T0000.nc3.nc'

# GOFS 3.1
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

#%%

import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import numpy as np
import matplotlib.dates as mdates
#from datetime import datetime
from matplotlib.dates import date2num
import xarray as xr
import netCDF4
import cmocean

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = np.array(ncglider.latitude[:])
longlider = np.array(ncglider.longitude[:])
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)
tempglider = np.array(ncglider.temperature[0,:,:]).T
saltglider = np.array(ncglider.salinity[0,:,:]).T
depthglider = np.array(ncglider.depth[0,:,:]).T
inst_id = ncglider.id.split('_')[0]

timestamp_glider = date2num(time_glider)[0]

#%%
#tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
#tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

tmin = time_glider[0][0]
tmax = time_glider[0][-1]

okg = np.where(np.logical_and(time_glider.T >= tmin, time_glider.T <= tmax))

timeg = time_glider[0,okg[0]]
timestampg = timestamp_glider[okg[0]]
latg = latglider[0,okg[0]]
long = longlider[0,okg[0]]
depthg = depthglider[:,okg[0]]
tempg = tempglider[:,okg[0]]
saltg = saltglider[:,okg[0]]

#%% Grid glider variables according to depth
             
depthg_gridded = np.arange(0,np.nanmax(depthg),0.5)
tempg_gridded = np.empty((len(depthg_gridded),len(timeg)))
tempg_gridded[:] = np.nan
saltg_gridded = np.empty((len(depthg_gridded),len(timeg)))
saltg_gridded[:] = np.nan

for t,tt in enumerate(timeg):
    #print(tt)
    depthu,oku = np.unique(depthg[:,t],return_index=True)
    tempu = tempg[oku,t]
    saltu = saltg[oku,t]
    okdd = np.isfinite(depthu)
    depthf = depthu[okdd]
    tempf = tempu[okdd]
    saltf = saltu[okdd]
 
    okt = np.isfinite(tempf)
    if np.sum(okt) < 3:
        tempg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[okt]),\
                            depthg_gridded < np.max(depthf[okt]))
        tempg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[okt],tempf[okt])
        
    oks = np.isfinite(saltf)
    if np.sum(oks) < 3:
        saltg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[okt]),\
                            depthg_gridded < np.max(depthf[okt]))
        saltg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[oks],saltf[oks])
        
#%% Get rid off of profiles with no data below 100 m
'''
tempg_full = []
timeg_full = []
for t,tt in enumerate(timeg):
    okt = np.isfinite(tempg_gridded[:,t])
    if sum(depthg_gridded[okt] > 100) > 10:
        if tempg_gridded[0,t] != tempg_gridded[20,t]:
            tempg_full.append(tempg_gridded[:,t]) 
            timeg_full.append(tt) 
       
tempg_full = np.asarray(tempg_full)
timeg_full = np.asarray(timeg_full)
'''
#%% Read GOFS 3.1 output

GOFS31 = xr.open_dataset(catalog31,decode_times=False)

lat31 = GOFS31.lat
lon31 = GOFS31.variables['lon'][:]
depth31 = GOFS31.variables['depth'][:]
tt31 = GOFS31.variables['time']
#t31 = netCDF4.num2date(tt31[:],tt31.units) 
t31 = netCDF4.num2date(tt31[:],'hours since 2000-01-01 00:00:00') 

#tmin = datetime.datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
#tmax = datetime.datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

oktime31 = np.where(np.logical_and(t31 >= tmin, t31 <= tmax))
time31 = t31[oktime31]

# Conversion from glider longitude and latitude to GOFS convention
target_lon = np.empty(len(long))
target_lon[:] = np.nan
for i in range(len(long)):
    if long[i] < 0: 
        target_lon[i] = 360 + long[i]
    else:
        target_lon[i] = long[i]
target_lat = latg

#%%

# Conversion from glider longitude and latitude to GOFS convention
target_lon = np.empty((len(long),))
target_lon[:] = np.nan
for i,ii in enumerate(long):
    if ii < 0: 
        target_lon[i] = 360 + ii
    else:
        target_lon[i] = ii
target_lat = latg

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
tstamp_model = [mdates.date2num(time31[i]) for i in np.arange(len(time31))]

# interpolating glider lon and lat to lat and lon on model time
sublon31=np.interp(tstamp_model,tstamp_glider,target_lon)
sublat31=np.interp(tstamp_model,tstamp_glider,target_lat)

# getting the model grid positions for sublonm and sublatm
oklon31=np.round(np.interp(sublon31,lon31,np.arange(len(lon31)))).astype(int)
oklat31=np.round(np.interp(sublat31,lat31,np.arange(len(lat31)))).astype(int)
    
# Getting glider transect from model
print('Getting glider transect from model. If it breaks is because GOFS 3.1 server is not responding')
target_temp31 = np.empty((len(depth31),len(oktime31[0])))
target_temp31[:] = np.nan
target_salt31 = np.empty((len(depth31),len(oktime31[0])))
target_salt31[:] = np.nan
for i in range(len(oktime31[0])):
    print(len(oktime31[0]),' ',i)
    target_temp31[:,i] = GOFS31.variables['water_temp'][oktime31[0][i],:,oklat31[i],oklon31[i]]
    target_salt31[:,i] = GOFS31.variables['salinity'][oktime31[0][i],:,oklat31[i],oklon31[i]]

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

#%% Map MAB with glider transects

# ng618
gdata_ng618 = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng618-20190901T0000/ng618-20190901T0000.nc3.nc'
# ng619
gdata_ng619 = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng619-20190901T0000/ng619-20190901T0000.nc3.nc'

ncglider_ng618 = xr.open_dataset(gdata_ng618,decode_times=False)
latglider_ng618 = np.array(ncglider_ng618.latitude[:])
longlider_ng618 = np.array(ncglider_ng618.longitude[:])
inst_id_ng618 = ncglider_ng618.id.split('_')[0]

ncglider_ng619 = xr.open_dataset(gdata_ng619,decode_times=False)
latglider_ng619 = np.array(ncglider_ng619.latitude[:])
longlider_ng619 = np.array(ncglider_ng619.longitude[:])
inst_id_ng619 = ncglider_ng619.id.split('_')[0]

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(6, 5))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
#plt.yticks([])
#plt.xticks([])
plt.plot(longlider_ng618,latglider_ng618,'.',color='darkorange')
ax.text(np.nanmean(longlider_ng618),np.nanmean(latglider_ng618),inst_id_ng618.split('-')[0],weight='bold',
        bbox=dict(facecolor='white',alpha=0.4,edgecolor='none'))
plt.plot(longlider_ng619,latglider_ng619,'.',color='darkorange')
ax.text(longlider_ng619[0,-1],latglider_ng619[0,-1],inst_id_ng619.split('-')[0],weight='bold',
        bbox=dict(facecolor='white',alpha=0.4,edgecolor='none'))

#plt.plot(longlider_ng619,latglider_ng619,'.',color='darkorange')
plt.title('Navy Glider Deployments ',fontsize=20)
ax.set_aspect(1)

file = folder + ' ' + 'Navy_gliders_MAB_2019 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Top 200 m temperature

color_map = cmocean.cm.thermal
       
okg = depthg_gridded <= 200
okm = depth31 <= 200 
min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp31[okm])]))
max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp31[okm])]))
    
nlevels = max_val - min_val + 1
kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    

# plot
fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.subplot(211)        
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=color_map,**kw)
plt.contour(timeg,-depthg_gridded,tempg_gridded,[26],colors='k')

cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
        
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
#xticks = [t0+nday*deltat for nday in np.arange(8)]
#xticks = np.asarray(xticks)
#plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticklabels(' ')
#tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(-1000,0)))
#tKaren = np.tile(datetime(2019,9,24,0),len(np.arange(-1000,0)))
#plt.plot(tKaren,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + inst_id)
#plt.legend()   

ax = plt.subplot(212)        
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(mdates.date2num(time31),-depth31,target_temp31,cmap=color_map,**kw)
plt.contour(mdates.date2num(time31),-depth31,target_temp31,[26],colors='k')
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
#xticks = [t0+nday*deltat for nday in np.arange(8)]
#xticks = np.asarray(xticks)
#plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
#plt.plot(tKaren,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Temperature' + ' Profile ' + 'GOFS 3.1')  
#plt.legend()   

file = folder + ' ' + 'along_track_temp_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Top 200 m salinity

color_map = cmocean.cm.haline
       
okg = depthg_gridded <= 200
okm = depth31 <= 200 
min_val = np.floor(np.min([np.nanmin(saltg_gridded[okg]),np.nanmin(target_salt31[okm])]))
max_val = np.ceil(np.max([np.nanmax(saltg_gridded[okg]),np.nanmax(target_salt31[okm])]))
    
nlevels = max_val - min_val + 1
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))
#kw = dict(levels = np.linspace(30,37,15)) #ng618
kw = dict(levels = np.linspace(30.6,36.0,10)) # ng619
    

# plot
fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.subplot(211)        
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,saltg_gridded,cmap=color_map,**kw)

cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
        
ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
#xticks = [t0+nday*deltat for nday in np.arange(8)]
#xticks = np.asarray(xticks)
#plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xticklabels(' ')
#tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(-1000,0)))
#tKaren = np.tile(datetime(2019,9,24,0),len(np.arange(-1000,0)))
#plt.plot(tKaren,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Salinity' + ' Profile ' + inst_id)
#plt.legend()   

ax = plt.subplot(212)        
#plt.contour(mdates.date2num(time31),-depth31,target_temp31,colors = 'lightgrey',**kw)
cs = plt.contourf(mdates.date2num(time31),-depth31,target_salt31,cmap=color_map,**kw)
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_xlim(timeg[0], timeg[-1])
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
#xticks = [t0+nday*deltat for nday in np.arange(8)]
#xticks = np.asarray(xticks)
#plt.xticks(xticks)
xfmt = mdates.DateFormatter('%d \n %b')
ax.xaxis.set_major_formatter(xfmt)
#plt.plot(tKaren,np.arange(-1000,0),'--k')
plt.title('Along Track ' + 'Salinity' + ' Profile ' + 'GOFS 3.1')  
#plt.legend()   

file = folder + ' ' + 'along_track_sal_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure GOFS 3.1 
'''
time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_temp_GOFS = temp31.T

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,10,5),datetime(2018,10,13))
ax.set_ylim(-260,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.tick_params(labelsize=14)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_Michael.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% Figure GOFS 3.1 with incremental insertion window

time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_temp_GOFS = temp31.T

dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,10,5),datetime(2018,10,13))
ax.set_ylim(-260,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.tick_params(labelsize=14)
plt.tick_params(labelsize=14)

patch = patches.Rectangle((date2num(datetime(2018,10,5,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,6,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,7,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,8,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,9,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,10,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,11,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,12,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_Michael2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% glider profile with less gaps

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(timeg_full,-depthg_gridded,tempg_full.T,colors = 'lightgrey',**kw)
plt.contour(timeg_full,-depthg_gridded,tempg_full.T,[26],colors = 'k')
plt.contourf(timeg_full,-depthg_gridded,tempg_full.T,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_title('ng288',fontsize=20)
ax.set_xlim(datetime(2018,10,5),datetime(2018,10,13))
ax.set_ylim(-260,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.tick_params(labelsize=14)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng288_Michael_vs_depth2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

plt.show()

#%% Time series at 10 m

d = 10
nz31 = np.where(depth31 >= d)[0][0]
nzg = np.where(depthg_gridded >= d)[0][0]
dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.plot(timeg_full,tempg_full[:,nzg],'o-',color='royalblue',label='ng288')
plt.plot(time31,target_temp_GOFS[:,nz31],'o-g',label='GOFS 3.1')
plt.legend(fontsize=14)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(np.arange(27.5,29.5,0.1))),\
         np.arange(27.5,29.5,0.1),'--k')
patch = patches.Rectangle((date2num(datetime(2018,10,5,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,6,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,7,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,8,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,9,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,10,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,11,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,12,9)),27.4),dt,2.6,\
                          facecolor = 'lightgray')
ax.add_patch(patch)

ax.set_ylabel('($^\circ$C)',fontsize=16)
ax.set_title('Time Series Temperature at 10 m',fontsize=20)
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
#cbar.ax.tick_params(labelsize=14)
plt.tick_params(labelsize=14)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_time_series_ng288_GOFS31_10m.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Time series at 100 m

d = 100
nz31 = np.where(depth31 >= d)[0][0]
nzg = np.where(depthg_gridded >= d)[0][0]
dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

fig, ax = plt.subplots(figsize=(8.8, 1.7))
plt.plot(timeg_full,tempg_full[:,nzg],'o-',color='royalblue',label='ng288')
plt.plot(time31,target_temp_GOFS[:,nz31],'o-g',label='GOFS 3.1')
plt.legend(fontsize=14,bbox_to_anchor = [0.45, 0.6])
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(np.arange(22,27.5,0.1))),\
         np.arange(22,27.5,0.1),'--k')

patch = patches.Rectangle((date2num(datetime(2018,10,5,9)),21.6),dt,270,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,6,9)),21.5),dt,270,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,7,9)),21.5),dt,270,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,8,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,9,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,10,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,11,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)
patch = patches.Rectangle((date2num(datetime(2018,10,12,9)),21.5),dt,6.2,\
                          facecolor = 'lightgray')
ax.add_patch(patch)

ax.set_ylabel('($^\circ$C)',fontsize=16)
ax.set_title('Time Series Temperature at 100 m',fontsize=20)
ax.set_xlim(datetime(2018,10,5),datetime(2018,10,13))
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.tick_params(labelsize=14)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_time_series_ng288_GOFS31_100m.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

'''