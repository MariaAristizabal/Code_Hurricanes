#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:07:35 2020

@author: aristizabal
"""

#%% User input

# RU33 (MAB + SAB)
lon_lim = [-75,-70]
lat_lim = [36,42]

# Folder where to save figure
folder_fig = '/home/aristizabal/Figures/'

# Folder Fay POM
folder_hwrf_pom = '/home/aristizabal/HWRF_POM_Fay/HWRF_POM_06l_2020070918/'

# Folder Fay HWRF
folder_hwrf = '/home/aristizabal/HWRF_POM_Fay/HWRF_POM_06l_2020070918_grib2_to_nc/'

# File Fay ROMS
file_ROMS = '/home/aristizabal/WRF_ROMS_Fay/roms_his_fay.nc'

# File Fay WRF
file_WRF = '/home/aristizabal/WRF_ROMS_Fay/fay_wrf_his_d01_2020-07-09_12_00_00_subset.nc'

# MARACOSS erddap server 
url_buoy = "http://erddap.maracoos.org/erddap/"

#%%
from erddapy import ERDDAP
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
from datetime import datetime
from datetime import timedelta
import cmocean
import os
import os.path
import glob
from matplotlib.dates import date2num, num2date
import matplotlib.dates as mdates

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Time window

tini = datetime(2020, 7, 9, 18)
tend = datetime(2020, 7, 14, 18)

#%%
print('Looking for data sets')
e = ERDDAP(server = url_buoy)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': tini.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'max_time': tend.strftime('%Y-%m-%dT%H:%M:%SZ'),
}

search_url = e.get_search_url(response='csv', **kw)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
datasets = search['Dataset ID'].values

msg = 'Found {} Datasets:\n\n{}'.format
print(msg(len(datasets), '\n'.join(datasets)))

#%% For: AtlanticShores_833d_ebdb_69d0

# Setting constraints
constraints = {
        'time>=': str(tini),
        'time<=': str(tend),
        }

variables = [
        'time',
        'station_name',
        'latitude',
        'longitude',
        'depth',
        'sea_water_velocity_to_direction',
        'sea_water_speed'
        ]

e = ERDDAP(
        server=url_buoy,
        protocol='tabledap',
        response='nc'
        )

e.dataset_id = datasets[0]
e.constraints = constraints
e.variables = variables

df_vel = e.to_pandas(
            index_col='time (UTC)',
            parse_dates=True,
            )

time_vel, ind = np.unique(df_vel.index,return_index=True)
depth_vel = df_vel['depth (m)'].values
water_speed = df_vel['sea_water_speed (cm/s)'].values

# Reshape velocity and depth into array depth x time
zn = ind[1] # 34 vertical levels
depth_levels = depth_vel[0:zn]

water_speed_matrix = np.empty((zn,len(time_vel)))
water_speed_matrix[:] = np.nan
for i,ii in enumerate(ind):
    if i < len(time_vel)-1:
        water_speed_matrix[0:len(water_speed[ind[i]:ind[i+1]]),i] = water_speed[ind[i]:ind[i+1]]
    else:
        water_speed_matrix[0:len(water_speed[ind[i]:len(water_speed)]),i] = water_speed[ind[i]:len(water_speed)]

plt.figure()
plt.contourf(time_vel,-depth_levels,water_speed_matrix)
plt.colorbar()
        
#%% For: AtlanticShores_b9cf_7616_205e

# Setting constraints
constraints = {
        'time>=': str(tini),
        'time<=': str(tend),
        }

variables = [
        'time',
        'station_name',
        'latitude',
        'longitude',
        'air_pressure',
        'air_temperature',
        'relative_humidity',
        'sea_water_temperature_at_1m',
        'sea_water_temperature_at_2m',
        'sea_water_temperature_at_32m',
        'sea_water_salinity',
        'sea_water_pressure',
        'wind_speed_of_gust'
        ]

e = ERDDAP(
        server=url_buoy,
        protocol='tabledap',
        response='nc'
        )

e.dataset_id = datasets[1]
e.constraints = constraints
e.variables = variables

df_air_water_temp = e.to_pandas(
            index_col='time (UTC)',
            parse_dates=True)

time_temp = df_air_water_temp.index.values
stat_name = df_air_water_temp['station_name'].values[0]
lat_buoy = df_air_water_temp['latitude (degrees_north)'].values[0]
lon_buoy = df_air_water_temp['longitude (degrees_east)'].values[0]
air_press = df_air_water_temp['air_pressure (mbar)'].values
air_temp = df_air_water_temp['air_temperature (degree_C)'].values 
relat_hum = df_air_water_temp['relative_humidity (percent)'].values 
water_temp_1m = df_air_water_temp['sea_water_temperature_at_1m (degree_C)'].values
water_temp_2m = df_air_water_temp['sea_water_temperature_at_2m (degree_C)'].values
water_temp_32m = df_air_water_temp['sea_water_temperature_at_32m (degree_C)'].values
water_salt = df_air_water_temp['sea_water_salinity (PSU)'].values
           
#%% For: AtlanticShores_2582_8641_755a

# Setting constraints
constraints = {
        'time>=': str(tini),
        'time<=': str(tend),
        }

variables = [
        'time',
        'station_name',
        'latitude',
        'longitude',
        'altitude',
        'wind_from_direction',
        'wind_speed'
        ]

e = ERDDAP(
        server=url_buoy,
        protocol='tabledap',
        response='nc'
        )

e.dataset_id = datasets[2]
e.constraints = constraints
e.variables = variables

df_wind = e.to_pandas(
            index_col='time (UTC)',
            parse_dates=True,
            )            
        
#%% Reading POM grid file

grid_file = sorted(glob.glob(os.path.join(folder_hwrf_pom,'*grid*.nc')))[0]
pom_grid = xr.open_dataset(grid_file)
lon_pom = np.asarray(pom_grid['east_e'][:])
lat_pom = np.asarray( pom_grid['north_e'][:])
zlevc = np.asarray(pom_grid['zz'][:])
topoz = np.asarray(pom_grid['h'][:])

#%% Getting list of POM files
ncfiles = sorted(glob.glob(os.path.join(folder_hwrf_pom,'*pom.0*.nc')))

# Reading POM time
time_pom = []
for i,file in enumerate(ncfiles):
    print(i)
    pom = xr.open_dataset(file)
    tpom = pom['time'][:]
    timestamp_pom = date2num(tpom)[0]
    time_pom.append(num2date(timestamp_pom))

time_POM = np.asarray(time_pom)

oklon = np.round(np.interp(lon_buoy,lon_pom[0,:],np.arange(len(lon_pom[0,:])))).astype(int)
oklat = np.round(np.interp(lat_buoy,lat_pom[:,0],np.arange(len(lat_pom[:,0])))).astype(int)
topoz_pom = np.asarray(topoz[oklat,oklon])
zmatrix_POM = np.dot(topoz_pom.reshape(-1,1),zlevc.reshape(1,-1)).T

#%% Get POM temperature profile
prof_temp_POM = np.empty((len(time_POM),zmatrix_POM.shape[0]))
prof_temp_POM[:] = np.nan

for i,file in enumerate(ncfiles):
    print(i)
    pom = xr.open_dataset(file)
    tpom = pom['time'][:]
    timestamp_pom = date2num(tpom)[0]
    timePOM = num2date(timestamp_pom)
    print(timePOM)
    prof_temp_POM[i,:] = np.asarray(pom['t'][0,:,oklat,oklon])
    
#%% Getting list of HWRF files
    
ncfiles_hwrf = sorted(glob.glob(os.path.join(folder_hwrf,'*.nc')))

time_hwrf = []
PRES_hwrf = []
TMP_hwrf = []
RH2m_hwrf = []
UGRD_hwrf = []
VGRD_hwrf = []
SHTFL_hwrf = []
LHTFL_hwrf = []
for N,file in enumerate(ncfiles_hwrf):
    print(N)
    HWRF = xr.open_dataset(file)
    lat_hwrf = np.asarray(HWRF.variables['latitude'][:])
    lon_hwrf = np.asarray(HWRF.variables['longitude'][:])
    
    oklon = np.round(np.interp(lon_buoy,lon_hwrf,np.arange(len(lon_hwrf)))).astype(int)
    oklat = np.round(np.interp(lat_buoy,lat_hwrf,np.arange(len(lat_hwrf)))).astype(int)
    
    if np.logical_and(oklon != 0, oklat != 0):  
        time_hwrf.append(HWRF.variables['time'].values)
        PRES_hwrf.append(HWRF.variables['PRES_surface'][0,oklat,oklon].values)
        TMP_hwrf.append(HWRF.variables['TMP_surface'][0,oklat,oklon].values)
        RH2m_hwrf.append(HWRF.variables['RH_2maboveground'][0,oklat,oklon])
        UGRD_hwrf.append(HWRF.variables['UGRD_10maboveground'][0,oklat,oklon])
        VGRD_hwrf.append(HWRF.variables['VGRD_10maboveground'][0,oklat,oklon])
        SHTFL_hwrf.append(HWRF.variables['SHTFL_surface'][0,oklat,oklon])
        LHTFL_hwrf.append(HWRF.variables['LHTFL_surface'][0,oklat,oklon])
        #enth_hwrf = SHTFL_hwrf + LHTFL_hwrf
    
time_hwrf = np.asarray(time_hwrf)
PRES_hwrf = np.asarray(PRES_hwrf)*0.01
TMP_hwrf = np.asarray(TMP_hwrf) - 273.15
RH2m_hwrf = np.asarray(RH2m_hwrf)
UGRD_hwrf = np.asarray(UGRD_hwrf)
VGRD_hwrf = np.asarray(VGRD_hwrf)
SHTFL_hwrf = np.asarray(SHTFL_hwrf)
LHTFL_hwrf = np.asarray(LHTFL_hwrf)

#%% Read ROMS time, lat and lon
print('Retrieving coordinates and time from ROMS ')

ROMS = xr.open_dataset(file_ROMS,decode_times=False)

latrhoROMS = np.asarray(ROMS.variables['lat_rho'][:])
lonrhoROMS = np.asarray(ROMS.variables['lon_rho'][:])
srhoROMS = np.asarray(ROMS.variables['s_rho'][:])
ttROMS = ROMS.variables['ocean_time'][:]
timROMS = netCDF4.num2date(ttROMS[:],ttROMS.attrs['units'])

timeROMS = [timROMS[t]._to_real_datetime() for t in np.arange(len(timROMS))]

target_lonROMS = [lon_buoy,lon_buoy]
target_latROMS = [lat_buoy,lat_buoy]

# getting the model grid positions for target_lonROMS and target_latROMS
oklatROMS = np.empty((len(target_lonROMS)))
oklatROMS[:] = np.nan
oklonROMS= np.empty((len(target_lonROMS)))
oklonROMS[:] = np.nan

# search in xi_rho direction 
oklatmm = []
oklonmm = []
for x in np.arange(len(target_latROMS)):
    print(x)
    for pos_xi in np.arange(latrhoROMS.shape[1]):
        pos_eta = np.round(np.interp(target_latROMS[x],latrhoROMS[:,pos_xi],np.arange(len(latrhoROMS[:,pos_xi])),\
                                     left=np.nan,right=np.nan))
        if np.isfinite(pos_eta):
            oklatmm.append((pos_eta).astype(int))
            oklonmm.append(pos_xi)
        
    pos = np.round(np.interp(target_lonROMS[x],lonrhoROMS[oklatmm,oklonmm],np.arange(len(lonrhoROMS[oklatmm,oklonmm])))).astype(int)    
    oklatROMS1 = oklatmm[pos]
    oklonROMS1 = oklonmm[pos] 
    
    #search in eta-rho direction
    oklatmm = []
    oklonmm = []
    for pos_eta in np.arange(latrhoROMS.shape[0]):
        pos_xi = np.round(np.interp(target_lonROMS[x],lonrhoROMS[pos_eta,:],np.arange(len(lonrhoROMS[pos_eta,:])),\
                                    left=np.nan,right=np.nan))
        if np.isfinite(pos_xi):
            oklatmm.append(pos_eta)
            oklonmm.append(pos_xi.astype(int))
    
    pos_lat = np.round(np.interp(target_latROMS[x],latrhoROMS[oklatmm,oklonmm],np.arange(len(latrhoROMS[oklatmm,oklonmm])))).astype(int)
    oklatROMS2 = oklatmm[pos_lat]
    oklonROMS2 = oklonmm[pos_lat] 
    
    #check for minimum distance
    dist1 = np.sqrt((oklonROMS1-target_lonROMS[x])**2 + (oklatROMS1-target_latROMS[x])**2) 
    dist2 = np.sqrt((oklonROMS2-target_lonROMS[x])**2 + (oklatROMS2-target_latROMS[x])**2) 
    if dist1 >= dist2:
        oklatROMS[x] = oklatROMS1
        oklonROMS[x] = oklonROMS1
    else:
        oklatROMS[x] = oklatROMS2
        oklonROMS[x] = oklonROMS2
        
    oklatROMS = oklatROMS.astype(int)
    oklonROMS = oklonROMS.astype(int)

#%% Read ROMS S-coordinate parameters

Vtransf = np.asarray(ROMS.variables['Vtransform'])
Vstrect = np.asarray(ROMS.variables['Vstretching'])
Cs_r = np.asarray(ROMS.variables['Cs_r'])
Cs_w = np.asarray(ROMS.variables['Cs_w'])
sc_r = np.asarray(ROMS.variables['s_rho'])
sc_w = np.asarray(ROMS.variables['s_w'])

# depth
h = np.asarray(ROMS.variables['h'])
# critical depth parameter
hc = np.asarray(ROMS.variables['hc'])

igrid = 1

#%%
zROMS = np.empty((len(timeROMS),sc_r.shape[0]))
zROMS[:] = np.nan

tempROMS = np.asarray(ROMS.variables['temp'][:,:,oklatROMS[0],oklonROMS[0]])
h = np.asarray(ROMS.variables['h'][oklatROMS[0],oklonROMS[0]])
zeta = np.asarray(ROMS.variables['zeta'][:,oklatROMS[0],oklonROMS[0]])
        
# Calculate ROMS depth as a function of time
if Vtransf ==1:
    if igrid == 1:
        for k in np.arange(sc_r.shape[0]):
            z0 = (sc_r[k]-Cs_r[k])*hc + Cs_r[k]*h
            zROMS[:,k] = z0 + zeta * (1.0 + z0/h)

if Vtransf == 2:
    if igrid == 1:
        for k in np.arange(sc_r.shape[0]):
            z0 = (hc*sc_r[k] + Cs_r[k]*h) / (hc+h)
            zROMS[:,k] = zeta + (zeta+h)*z0

tempROMS_1m = np.empty((len(timeROMS)))     
tempROMS_1m[:] = np.nan       
for t in np.arange(len(timeROMS)):
    okd = np.round(np.interp(1,zROMS[t,:],np.arange(zROMS.shape[1]))).astype(int)
    tempROMS_1m[t] = tempROMS[t,okd]
    
#%% Reading WRF output
print('Retrieving coordinates and time from WRF ')

WRF = xr.open_dataset(file_WRF,decode_times=False)

lat_WRF = np.asarray(WRF.variables['XLAT'][:,:,0])
lon_WRF = np.asarray(WRF.variables['XLONG'][:,:,0])

ti_WRF = WRF.variables['Time']
tim_WRF = netCDF4.num2date(ti_WRF[:],ti_WRF.attrs['units'])

time_WRF = [tim_WRF[t]._to_real_datetime() for t in np.arange(len(tim_WRF))]

#%% Finding grid points in WRF 

target_lonWRF = [lon_buoy,lon_buoy]
target_latWRF = [lat_buoy,lat_buoy]

# getting the model grid positions for target_lonWRF and target_latWRF
oklatWRF = np.empty((len(target_lonWRF)))
oklatWRF[:] = np.nan
oklonWRF= np.empty((len(target_lonWRF)))
oklonWRF[:] = np.nan

# search in xi_rho direction 
oklatmm = []
oklonmm = []
for x in np.arange(len(target_latWRF)):
    print(x)
    for pos_xi in np.arange(lat_WRF.shape[1]):
        pos_eta = np.round(np.interp(target_latWRF[x],lat_WRF[:,pos_xi],np.arange(len(lat_WRF[:,pos_xi])),\
                                     left=np.nan,right=np.nan))
        if np.isfinite(pos_eta):
            oklatmm.append((pos_eta).astype(int))
            oklonmm.append(pos_xi)
        
    pos = np.round(np.interp(target_lonWRF[x],lon_WRF[oklatmm,oklonmm],np.arange(len(lon_WRF[oklatmm,oklonmm])))).astype(int)    
    oklatWRF1 = oklatmm[pos]
    oklonWRF1 = oklonmm[pos] 
    
    #search in eta-rho direction
    oklatmm = []
    oklonmm = []
    for pos_eta in np.arange(lat_WRF.shape[0]):
        pos_xi = np.round(np.interp(target_lonWRF[x],lon_WRF[pos_eta,:],np.arange(len(lon_WRF[pos_eta,:])),\
                                    left=np.nan,right=np.nan))
        if np.isfinite(pos_xi):
            oklatmm.append(pos_eta)
            oklonmm.append(pos_xi.astype(int))
    
    pos_lat = np.round(np.interp(target_latWRF[x],lat_WRF[oklatmm,oklonmm],np.arange(len(lat_WRF[oklatmm,oklonmm])))).astype(int)
    oklatWRF2 = oklatmm[pos_lat]
    oklonWRF2 = oklonmm[pos_lat] 
    
    #check for minimum distance
    dist1 = np.sqrt((oklonWRF1-target_lonWRF[x])**2 + (oklatWRF1-target_latWRF[x])**2) 
    dist2 = np.sqrt((oklonWRF2-target_lonWRF[x])**2 + (oklatWRF2-target_latWRF[x])**2) 
    if dist1 >= dist2:
        oklatWRF[x] = oklatWRF1
        oklonWRF[x] = oklonWRF1
    else:
        oklatWRF[x] = oklatWRF2
        oklonWRF[x] = oklonWRF2
        
    oklatWRF = oklatWRF.astype(int)
    oklonWRF = oklonWRF.astype(int)

#%% Reading WRF variables

T2_WRF = np.asarray(WRF.variables['T2'][:,oklatWRF[0],oklonWRF[0]]) - 273.15
SLP_WRF = np.asarray(WRF.variables['SLP'][:,oklatWRF[0],oklonWRF[0]])
RH2_WRF = np.asarray(WRF.variables['RH2'][:,oklatWRF[0],oklonWRF[0]])
    
#%% ROMS Time series temp at 1m

tfay = datetime(2020,7,10,20)

max_valt = 26
min_valt = 8  
nlevelst = max_valt - min_valt + 1
kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))
  
okd = np.round(np.interp(1.0,np.abs(zmatrix_POM[:,0]),np.arange(zmatrix_POM.shape[0]))).astype(int)
temp_POM_1m = prof_temp_POM[:,okd]
 
fig,ax = plt.subplots(figsize=(10, 4))
plt.plot(time_temp,water_temp_1m,'X-',color='royalblue',label=stat_name + ' Buoy')
plt.plot(time_POM,temp_POM_1m,'-o',color='mediumorchid',label='HWRF-MPIPOM (IC clim.)')
plt.plot(timeROMS,tempROMS_1m,'-s',color='maroon',label='WRF-ROMS')
plt.plot(np.tile(tfay,len(np.arange(21,27))),np.arange(21,27),'--k')
#xfmt = mdates.DateFormatter('%d-%b\n %Y')
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.legend()
plt.title('Temperature at 1 m depth',fontsize=16)
plt.ylabel('$^oC$',fontsize=14)
file = folder_fig + 'temp_time_series_1m_HWRF-POM_Atlac_Shor_'+str(time_POM[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%  Air pressure

fig,ax = plt.subplots(figsize=(10, 4))
plt.plot(time_temp,air_press,'X-',color='royalblue',label=stat_name + ' Buoy')
plt.plot(time_hwrf,PRES_hwrf,'-o',color='mediumorchid',label='HWRF-MPIPOM (IC clim.)')
plt.plot(time_WRF,SLP_WRF,'-s',color='maroon',label='WRF-ROMS')
plt.plot(np.tile(tfay,len(np.arange(1000,1018))),np.arange(1000,1018),'--k')
#xfmt = mdates.DateFormatter('%d-%b\n %Y')
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.legend()
plt.title('Air Pressure at Surface',fontsize=16)
plt.ylabel('mbar',fontsize=14)
file = folder_fig + 'air_pres_surf_HWRF_POM_WRF_ROMS_Atlac_Shor_'+str(time_hwrf[0][0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Air temperature

fig,ax = plt.subplots(figsize=(10, 4))
plt.plot(time_temp,air_temp,'X-',color='royalblue',label=stat_name + ' Buoy')
plt.plot(time_hwrf,TMP_hwrf,'-o',color='mediumorchid',label='HWRF-MPIPOM (IC clim.)')
plt.plot(time_WRF,T2_WRF,'-s',color='maroon',label='WRF-ROMS')
plt.plot(np.tile(tfay,len(np.arange(22,27))),np.arange(22,27),'--k')
#xfmt = mdates.DateFormatter('%d-%b\n %Y')
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.legend()
plt.title('Air Temperature at Surface',fontsize=16)
plt.ylabel('$^oC$',fontsize=14)
file = folder_fig + 'air_temp_surf_HWRF_POM_WRF_ROMS_Atlac_Shor_'+str(time_hwrf[0][0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Relative humidity

fig,ax = plt.subplots(figsize=(10, 4))
plt.plot(time_temp,relat_hum,'X-',color='royalblue',label=stat_name + ' Buoy')
plt.plot(time_hwrf,RH2m_hwrf,'-o',color='mediumorchid',label='HWRF-MPIPOM (IC clim.)')
plt.plot(time_WRF,RH2_WRF,'-s',color='maroon',label='WRF-ROMS')
plt.plot(np.tile(tfay,len(np.arange(60,100))),np.arange(60,100),'--k')
#xfmt = mdates.DateFormatter('%d-%b\n %Y')
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.legend()
plt.title('Relative Humidity',fontsize=16)
plt.ylabel('$\%$',fontsize=14)
file = folder_fig + 'relat_hum_HWRF-POM_WRF_ROMS_Atlac_Shor_'+str(time_hwrf[0][0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

