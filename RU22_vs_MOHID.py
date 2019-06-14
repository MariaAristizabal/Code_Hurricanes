#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:08:17 2019

@author: aristizabal
"""

#%% User input

# folder whre MOHID nc files reside
MOHID_folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/20180819-25_MOHIDDATA/'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# date limits
date_ini = '2018-08-19T12:00:00Z'
date_end = '2018-08-25T00:00:00Z'

# Glider data 

# ng288
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc'

#%%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.dates import date2num
import xarray as xr
import netCDF4
import os
import glob
import cmocean
import seawater as sw

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)
tempglider = np.array(ncglider.temperature[0,:,:])
saltglider = np.array(ncglider.salinity[0,:,:])
densglider = np.array(ncglider.density[0,:,:])
depthglider = np.array(ncglider.depth[0,:,:])

timestamp_glider = date2num(time_glider)[0]

tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

okg = np.where(np.logical_and(time_glider.T >= tmin, time_glider.T <= tmax))

timeg = time_glider[0,okg[0]]
timestampg = timestamp_glider[okg[0]]
latg = np.asarray(latglider[0,okg[0]])
long = np.asarray(longlider[0,okg[0]])
depthg = depthglider[okg[0],:]
tempg = tempglider[okg[0],:]
saltg = saltglider[okg[0],:]
densg = densglider[okg[0],:]

target_lat = latg
target_lon = long

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]

#%% Grid glider variables according to depth

depthg_gridded = np.arange(0,np.nanmax(depthg),0.5)
tempg_gridded = np.empty((len(timeg),len(depthg_gridded)))
tempg_gridded[:] = np.nan
saltg_gridded = np.empty((len(timeg),len(depthg_gridded)))
saltg_gridded[:] = np.nan
densg_gridded = np.empty((len(timeg),len(depthg_gridded)))
densg_gridded[:] = np.nan

for t,tt in enumerate(timeg):
    depthu,oku = np.unique(depthg[t,:],return_index=True)
    tempu = tempg[t,oku]
    saltu = saltg[t,oku]
    densu = densg[t,oku]
    okdd = np.isfinite(depthu)
    depth_fin = depthu[okdd]
    temp_fin = tempu[okdd]
    salt_fin = saltu[okdd]
    dens_fin = densu[okdd]
    ok = np.isfinite(temp_fin)
    
    if np.sum(ok) < 3:
        tempg_gridded[t,:] = np.nan
        saltg_gridded[t,:] = np.nan
        densg_gridded[t,:] = np.nan
    else:
        okd = depthg_gridded < np.max(depth_fin[ok])
        tempg_gridded[t,okd] = np.interp(depthg_gridded[okd],depth_fin[ok],temp_fin[ok])
        saltg_gridded[t,okd] = np.interp(depthg_gridded[okd],depth_fin[ok],salt_fin[ok])
        densg_gridded[t,okd] = np.interp(depthg_gridded[okd],depth_fin[ok],dens_fin[ok])
        
#%% Get rid off of profiles with no data below 10 m

dc = 10
tempg_full = []
saltg_full = []
densg_full = []
timeg_full = []
for t,tt in enumerate(timeg):
    okt = np.isfinite(tempg_gridded[t,:])
    if sum(depthg_gridded[okt] > dc) > 10:
        if tempg_gridded[t,0] != tempg_gridded[t,20]:
            tempg_full.append(tempg_gridded[t,:])
            saltg_full.append(saltg_gridded[t,:])
            densg_full.append(densg_gridded[t,:])
            timeg_full.append(tt)
             
tempg_full = np.asarray(tempg_full)
saltg_full = np.asarray(saltg_full)
densg_full = np.asarray(densg_full)
timeg_full = np.asarray(timeg_full)

#%% Access MOHID files

MOHID_files = sorted(glob.glob(os.path.join(MOHID_folder,'*Water*')))

ncmohid = xr.open_dataset(MOHID_files[0])

#dep_mohid = np.asarray(ncmohid.variables['depth'][:])
dep_mohid = np.flipud(-1*np.array([1.65, 3.45, 5.25, 7.2, 9.15, 11.1, 13.05, 15, 18, 25,\
                      40, 62.5, 87.5, 112.5, 137.5, 175, 225, 275, 350, 450,\
                      550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450,\
                      1625, 1875, 2250, 2750, 3250, 3750, 4250, 4750, 5250, 5750]))
lat_mohid = np.asarray(ncmohid.variables['lat'][:])
lon_mohid = np.asarray(ncmohid.variables['lon'][:])

# getting time
time_moh = []
for f,file in enumerate(MOHID_files):
    ncmohid = xr.open_dataset(file)
    tt = ncmohid.variables['time'][:]
    time_moh.append([mdates.num2date(mdates.date2num(t)) for t in tt])
    
time_mohid = np.asarray([val for sublist in time_moh for val in sublist])    

#%% Getting temperature and salinity from MOHID followig glider track

#oktimem = np.where(np.logical_and(mdates.date2num(time_mohi) >= mdates.date2num(tmin), \
#                                  mdates.date2num(time_mohi) <= mdates.date2num(tmax)))
#time_mohid = time_mohi[oktimem[0]] 

target_temp_moh = np.empty((len(dep_mohid),25,len(MOHID_files)))
target_temp_moh[:] = np.nan

target_salt_moh = np.empty((len(dep_mohid),25,len(MOHID_files)))
target_salt_moh[:] = np.nan

#depth_matrix_moh = np.empty((len(dep_mohid),25,len(MOHID_files)))
#depth_matrix_moh[:] = np.nan   

for f,file in enumerate(MOHID_files):
    ncmohid = xr.open_dataset(file)
    tt_moh = np.asarray(ncmohid.variables['time'][:])
    
    # Changing times to timestamp
    tstamp_model = [mdates.date2num(tt_moh[i]) for i in np.arange(len(tt_moh))]
    
    # interpolating glider lon and lat to lat and lon on model time
    sublonm=np.interp(tstamp_model,tstamp_glider,target_lon)
    sublatm=np.interp(tstamp_model,tstamp_glider,target_lat)

    # getting the model grid positions for sublonm and sublatm
    oklonm=np.round(np.interp(sublonm,lon_mohid[0,:],np.arange(len(lon_mohid[0,:])))).astype(int)
    oklatm=np.round(np.interp(sublatm,lat_mohid[:,0],np.arange(len(lat_mohid[:,0])))).astype(int)
    
    '''
    fig,ax = plt.subplots()
    plt.plot(tstamp_model,lon_mohid[0,oklonm],'.-r')
    plt.plot(tstamp_glider,target_lon,'.-g')
    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
    ax.xaxis.set_major_formatter(xfmt)
    
    fig,ax = plt.subplots()
    plt.plot(tstamp_model,lat_mohid[oklatm,0],'.-r')
    plt.plot(tstamp_glider,target_lat,'.-g')
    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
    ax.xaxis.set_major_formatter(xfmt)
    '''
    
    # Getting glider transect from model    
    for i in range(len(oklonm)):
        print(len(oklonm),' ',i)
        target_temp_moh[:,i,f] = ncmohid.variables['temperature'][i,:,oklatm[i],oklonm[i]]
        target_salt_moh[:,i,f] = ncmohid.variables['salinity'][i,:,oklatm[i],oklonm[i]]
        '''
        # Constructing depth matrix     
        bat_mohid = np.asarray(ncmohid.variables['bathymetry'][oklatm[i],oklonm[i]])
        mask_mohid = np.asarray(ncmohid.variables['mask'][:,oklatm[i],oklonm[i]])
        if np.isnan(bat_mohid):
            depth_matrix_moh[:,i,f] = np.nan
        else:
            dep_vec = mask_mohid * bat_mohid       
            ok = np.where(dep_vec == bat_mohid)
            delta_z = bat_mohid /len(ok[0])
            z_lev = [(bat_mohid  - (2*(n+1)-1)/2*delta_z) for n in ok[0]-ok[0][0]] 
            dep_vec[ok] = -1*np.asarray(z_lev) 
            depth_matrix_moh[:,i,f] = dep_vec
    '''
target_temp_mohid = np.reshape(target_temp_moh,(len(dep_mohid),25*len(MOHID_files)),order='F')
target_salt_mohid = np.reshape(target_salt_moh,(len(dep_mohid),25*len(MOHID_files)),order='F')

time_matrix_mohid = np.tile(mdates.date2num(time_mohid),(len(dep_mohid),1))
depth_matrix_mohid = np.tile(dep_mohid,(len(time_mohid),1)).T

#%% calculating density MOHID

target_dens_mohid = sw.dens(target_salt_mohid,target_temp_mohid,depth_matrix_mohid) 
  
#%% Interpolating lat and lon to glider track
'''            
# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
tstamp_model = [mdates.date2num(time_mohid[i]) for i in np.arange(len(time_mohid))]

# interpolating glider lon and lat to lat and lon on model time
sublonm=np.interp(tstamp_model,tstamp_glider,target_lon)
sublatm=np.interp(tstamp_model,tstamp_glider,target_lat)

# getting the model grid positions for sublonm and sublatm
oklonm=np.round(np.interp(sublonm,lon_mohid[0,:],np.arange(len(lon_mohid[0,:])))).astype(int)
oklatm=np.round(np.interp(sublatm,lat_mohid[:,0],np.arange(len(lat_mohid[:,0])))).astype(int)
    
# Getting glider transect from model
target_temp_mohid = np.empty((len(dep_mohid),len(oktimem[0])))
target_temp_mohid[:] = np.nan
for i in range(len(oktimem[0])):
    print(len(oktimem[0]),' ',i)
    target_temp_mohid[:,i] = ncmohid.variables['temperature'][oktimem[0][i],:,oklatm[i],oklonm[i]] 

'''

#%% Calculate depth from MOHID
'''
depth_matrix_mohid = np.empty((len(dep_mohid),len(oktimem[0])))
depth_matrix_mohid[:] = np.nan

for i in range(len(oktimem[0])):
    print(len(oktimem[0]),' ',i)
    bat_mohid = np.asarray(ncmohid.variables['bathymetry'][oklatm[i],oklonm[i]])
    mask_mohid = np.asarray(ncmohid.variables['mask'][:,oklatm[i],oklonm[i]])
    if np.isnan(bat_mohid):
            depth_matrix_mohid[:,i] = np.nan
    else:
        dep_vec = mask_mohid * bat_mohid       
        ok = np.where(dep_vec == bat_mohid)
        delta_z = bat_mohid /len(ok[0])
        z_lev = [(bat_mohid  - (2*(n+1)-1)/2*delta_z) for n in ok[0]-ok[0][0]] 
        dep_vec[ok] = -1*np.asarray(z_lev) 
        depth_matrix_mohid[:,i] = dep_vec
'''
 
#%% glider profile temperature with less gaps

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),20))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
#plt.contour(timeg_full,-depthg_gridded,tempg_full.T,colors = 'lightgrey',**kw)
plt.contour(timeg_full,-depthg_gridded,tempg_full.T,[26],colors = 'k')
plt.contourf(timeg_full,-depthg_gridded,tempg_full.T,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('RU22',fontsize=20)
ax.set_xlim(tmin,tmax)
ax.set_ylim(-100,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU22_SOULIK_vs_depth2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
                 
#%% Figure MOHID temperature

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),20))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.contourf(time_matrix_mohid,depth_matrix_mohid,target_temp_mohid,colors = 'lightgrey',**kw)
plt.contour(time_matrix_mohid,depth_matrix_mohid,target_temp_mohid,[26],colors = 'k')
plt.contourf(time_matrix_mohid,depth_matrix_mohid,target_temp_mohid,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xlim(tmin,tmax)
ax.set_ylim(-100,0)
plt.title('MOHID', fontsize=20)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/MOHID_RU22_SOULIK_temp.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Time series at 10 m

d = 10
nzm = np.where(dep_mohid >= -1*d)[0][0]
nzg = np.where(depthg_gridded >= d)[0][0]
dt = date2num(datetime(2018,8,23,12)) - date2num(datetime(2018,8,23,9))

fig, ax = plt.subplots(figsize=(8.8, 1.7))
plt.plot(timeg_full,tempg_full[:,nzg],'o-',color='royalblue',label='ng288')
plt.plot(mdates.date2num(time_mohid),target_temp_mohid[nzm,:],'o-g',label='MOHID')
plt.legend(fontsize=14)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(np.arange(17,30,0.1))),\
         np.arange(17,30,0.1),'--k')

ax.set_ylabel('($^\circ$C)',fontsize=16)
ax.set_title('Time Series Temperature at 10 m',fontsize=20)
ax.set_xlim(tmin,tmax)
ax.set_ylim(15,30)

xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/salt_time_series_RU22_MOHID_10m.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% glider profile salinity with less gaps

kw = dict(levels = np.linspace(30,34.6,24))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
#plt.contour(timeg_full,-depthg_gridded,saltg_full.T,colors = 'lightgrey',**kw)
plt.contourf(timeg_full,-depthg_gridded,saltg_full.T,cmap=cmocean.cm.haline,**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Salinity',fontsize=16)
ax.set_title('RU22',fontsize=20)
ax.set_xlim(tmin,tmax)
ax.set_ylim(-100,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU22_SOULIK_vs_depth_salt.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure MOHID salinity

time_matrix_mohid = np.tile(mdates.date2num(time_mohid),(len(dep_mohid),1))
depth_matrix_mohid = np.tile(dep_mohid,(len(time_mohid),1)).T

kw = dict(levels = np.linspace(30,34.6,24))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.contourf(time_matrix_mohid,depth_matrix_mohid,target_salt_mohid,colors = 'lightgrey',**kw)
plt.contourf(time_matrix_mohid,depth_matrix_mohid,target_salt_mohid,cmap=cmocean.cm.haline,**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Salinity',fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xlim(tmin,tmax)
ax.set_ylim(-100,0)
plt.title('MOHID', fontsize=20)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/MOHID_RU22_SOULIK_salt.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Time series salinity at 10 m

d = 10
nzm = np.where(dep_mohid >= -1*d)[0][0]
nzg = np.where(depthg_gridded >= d)[0][0]
dt = date2num(datetime(2018,8,23,12)) - date2num(datetime(2018,8,23,9))

fig, ax = plt.subplots(figsize=(8.8, 1.7))
plt.plot(timeg_full,saltg_full[:,nzg],'o-',color='royalblue',label='ng288')
plt.plot(mdates.date2num(time_mohid),target_salt_mohid[nzm,:],'o-g',label='MOHID')
plt.legend(fontsize=14)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(np.arange(30,34,0.1))),\
         np.arange(30,34,0.1),'--k')

ax.set_ylabel('Salinity',fontsize=16)
ax.set_title('Time Series Salinity at 10 m',fontsize=20)
ax.set_xlim(tmin,tmax)
ax.set_ylim(30,34)

xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/salt_time_series_RU22_MOHID_10m.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% glider profile density with less gaps

#kw = dict(levels = np.linspace(np.floor(np.nanmin(densg_gridded)),\
#                               np.ceil(np.nanmax(densg_gridded)),17))

kw = dict(levels = np.linspace(1018,1027,19))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
#plt.contour(timeg_full,-depthg_gridded,tempg_full.T,colors = 'lightgrey',**kw)
plt.contour(timeg_full,-depthg_gridded,densg_full.T,[26],colors = 'k')
plt.contourf(timeg_full,-depthg_gridded,densg_full.T,cmap=cmocean.cm.dense,**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Density ($Kg/m^3$)',fontsize=16)
ax.set_title('RU22',fontsize=20)
ax.set_xlim(tmin,tmax)
ax.set_ylim(-100,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU22_SOULIK_density.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure MOHID density

time_matrix_mohid = np.tile(mdates.date2num(time_mohid),(len(dep_mohid),1))
depth_matrix_mohid = np.tile(dep_mohid,(len(time_mohid),1)).T

kw = dict(levels = np.linspace(1018,1027,19))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.contourf(time_matrix_mohid,depth_matrix_mohid,target_dens_mohid,colors = 'lightgrey',**kw)
plt.contourf(time_matrix_mohid,depth_matrix_mohid,target_dens_mohid,cmap=cmocean.cm.dense,**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Density ($Kg/m^3$)',fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xlim(tmin,tmax)
ax.set_ylim(-100,0)
plt.title('MOHID', fontsize=20)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/MOHID_RU22_SOULIK_dens.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Time series density at 10 m

d = 10
nzm = np.where(dep_mohid >= -1*d)[0][0]
nzg = np.where(depthg_gridded >= d)[0][0]
dt = date2num(datetime(2018,8,23,12)) - date2num(datetime(2018,8,23,9))

fig, ax = plt.subplots(figsize=(8.8, 1.7))
plt.plot(timeg_full,densg_full[:,nzg],'o-',color='royalblue',label='ng288')
plt.plot(mdates.date2num(time_mohid),target_dens_mohid[nzm,:],'o-g',label='MOHID')
plt.legend(fontsize=14)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(np.arange(1018,1027,0.1))),\
         np.arange(1018,1027,0.1),'--k')

ax.set_ylabel('Density',fontsize=16)
ax.set_title('Time Series Density at 10 m',fontsize=20)
ax.set_xlim(tmin,tmax)
ax.set_ylim(1018,1027)

xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/density_time_series_RU22_MOHID_10m.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure GOFS 3.1 temperature
'''
time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_temp_GOFS = temp31.T

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),21))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_SOULIK.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% Figure GOFS 3.1 temperature with incremental insertion window

time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_temp_GOFS = temp31.T

dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),21))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_temp_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

patch = patches.Rectangle((date2num(datetime(2018,8,17,9)),-270),dt,270,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_SOULIK2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% Figure GOFS 3.1 salinity with incremental insertion window

time_matrix_GOFS = np.tile(timestamp31,(depth31.shape[0],1)).T
z_matrix_GOFS = np.tile(depth31,(time31.shape[0],1))
target_salt_GOFS = salt31.T

dt = date2num(datetime(2018,10,8,12)) - date2num(datetime(2018,10,8,9))

kw = dict(levels = np.linspace(31,34.6,19))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_salt_GOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrix_GOFS,-1*z_matrix_GOFS,target_salt_GOFS,[26],colors = 'k')
plt.contourf(time_matrix_GOFS,-1*z_matrix_GOFS,target_salt_GOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depth31)),-1*z_matrix_GOFS[0,:],'--k')
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Salinity',fontsize=16)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

patch = patches.Rectangle((date2num(datetime(2018,8,17,9)),-100),dt,100,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_salt_SOULIK2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%% glider profile temperature with less gaps

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),21))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
#plt.contour(timeg_full,-depthg_gridded,tempg_full.T,colors = 'lightgrey',**kw)
plt.contour(timeg_full,-depthg_gridded,tempg_full.T,[26],colors = 'k')
plt.contourf(timeg_full,-depthg_gridded,tempg_full.T,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('RU22',fontsize=20)
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU22_SOULIK_vs_depth2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

plt.show()

#%% glider profile salinity with less gaps

kw = dict(levels = np.linspace(31,34.6,19))

fig, ax = plt.subplots(figsize=(11, 1.7))
#fig, ax = plt.subplots(figsize=(6, 3))
plt.contour(timeg_full,-depthg_gridded,saltg_full.T,colors = 'lightgrey',**kw)
plt.contour(timeg_full,-depthg_gridded,saltg_full.T,[26],colors = 'k')
plt.contourf(timeg_full,-depthg_gridded,saltg_full.T,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Salinity',fontsize=16)
ax.set_title('RU22',fontsize=20)
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(-100,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/RU22_SOULIK_vs_depth_salt.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

plt.show()

#%% Time series at 10 m

d = 10
nz31 = np.where(depth31 >= d)[0][0]
nzg = np.where(depthg_gridded >= d)[0][0]
dt = date2num(datetime(2018,8,23,12)) - date2num(datetime(2018,8,23,9))

fig, ax = plt.subplots(figsize=(8.8, 1.7))
plt.plot(timeg_full,saltg_full[:,nzg],'o-',color='royalblue',label='ng288')
plt.plot(time31,target_salt_GOFS[:,nz31],'o-g',label='GOFS 3.1')
plt.legend(fontsize=14)
plt.plot(np.tile(datetime(2018, 8, 23, 0),len(np.arange(17,30,0.1))),\
         np.arange(17,30,0.1),'--k')
patch = patches.Rectangle((date2num(datetime(2018,8,17,9)),31),dt,4,\
                          facecolor = 'lightgray',alpha=0.5)
ax.add_patch(patch)

#ax.set_ylabel('($^\circ$C)',fontsize=16)
ax.set_title('Time Series Salinity at 10 m',fontsize=20)
ax.set_xlim(datetime(2018,8,15),datetime(2018,8,25))
ax.set_ylim(31,34.6)

xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/salt_time_series_RU22_GOFS31_10m.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
'''