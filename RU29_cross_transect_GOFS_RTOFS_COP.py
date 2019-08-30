#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 09:56:09 2019

@author: aristizabal
"""

#%% User input

# RU29 (Caribbean)
lon_lim = [-70,-62]
lat_lim = [16,20]
#gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

#Time window
#date_ini = '2019-08-26T00:00:00Z'
#date_end = '2019-08-27T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

# url for GOFS 3.1
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# FTP server RTOFS
ftp_RTOFS = 'ftp.ncep.noaa.gov'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

# COPERNICUS MARINE ENVIRONMENT MONITORING SERVICE (CMEMS)
url_cmems = 'http://nrt.cmems-du.eu/motu-web/Motu'
service_id = 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS'
product_id = 'global-analysis-forecast-phy-001-024'
depth_min = '0.493'
out_dir = '/Users/aristizabal/Desktop'
ncCOP_global = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/global-analysis-forecast-phy-001-024_1565877333169.nc'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
from datetime import datetime, timedelta
import cmocean
import matplotlib.dates as mdates 
from ftplib import FTP
import os
import os.path

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Get time bounds for the previous day
'''
te = datetime.today()
tend = datetime(te.year,te.month,te.day)

ti = datetime.today() - timedelta(1)
tini = datetime(ti.year,ti.month,ti.day)
'''

#%% Get time bounds for current day

te = datetime.today() + timedelta(1)
tend = datetime(te.year,te.month,te.day)

#ti = datetime.today() - timedelta(1)
ti = datetime.today() 
tini = datetime(ti.year,ti.month,ti.day)

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
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

#%% Defining cross transect

x1 = -(66 + 29/60 + 32.59/3600) 
y1 = 17 + 16/60 + 44.30/3600
x2 = -(62 + 42/60 + 23.88/3600)
y2 = 18 + 58/60 + 49.43/3600
# Slope
m = (y1-y2)/(x1-x2)
# Intercept
b = y1 - m*x1

X = np.arange(x1,x2,0.05)
Y = b + m*X

#%%

# Conversion from glider longitude and latitude to GOFS convention
target_lonGOFS = np.empty((len(X),))
target_lonGOFS[:] = np.nan
for i,ii in enumerate(X):
    if ii < 0: 
        target_lonGOFS[i] = 360 + ii
    else:
        target_lonGOFS[i] = ii
target_latGOFS = Y

# Conversion from glider longitude and latitude to RTOFS convention
target_lonRTOFS = X
target_latRTOFS = Y

# Conversion from glider longitude and latitude to Copernicus convention
target_lonCOP = X
target_latCOP = Y

dist = np.sqrt((X-x1)**2 + (Y-y1)**2)*111 # approx along transect distance in km

#%% Read GOFS 3.1 lat, lon, depth and time

print('Retrieving coordinates from GOFS')

GOFS = xr.open_dataset(url_GOFS,decode_times=False)

latGOFS = np.asarray(GOFS['lat'][:])
lonGOFS = np.asarray(GOFS['lon'][:])
depthGOFS = np.asarray(GOFS['depth'][:])
ttGOFS= GOFS['time']
tGOFS = netCDF4.num2date(ttGOFS[:],ttGOFS.units) 

#tini = datetime.datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
#tend = datetime.datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

oktimeGOFS = np.where(np.logical_and(tGOFS >= tini, tGOFS <= tend))
timeGOFS = tGOFS[oktimeGOFS]

# interpolating transect X and Y to lat and lon 
oklonGOFS = np.round(np.interp(target_lonGOFS,lonGOFS,np.arange(0,len(lonGOFS)))).astype(int)
oklatGOFS = np.round(np.interp(target_latGOFS,latGOFS,np.arange(0,len(latGOFS)))).astype(int)

#%% load RTOFS nc files

print('Loading 6 hourly RTOFS nc files from FTP server')
for t in np.arange(len(nc_files_RTOFS)):
    #file = out_dir + '/' + nc_files_RTOFS[t]
    file = nc_files_RTOFS[t]

    # Login to ftp file
    ftp = FTP('ftp.ncep.noaa.gov')
    ftp.login()
    ftp.cwd('pub/data/nccf/com/rtofs/prod/')
    if tend.month < 10:
        if tend.day < 10:
            ftp.cwd('rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day))
        else:
            ftp.cwd('rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day))
    else:
        if tend.day < 10:
            ftp.cwd('rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day))
        else:
            ftp.cwd('rtofs.' + str(tini.year) + str(tini.month) + str(tini.day))

    # Download nc files
    print('loading ' + file)
    ftp.retrbinary('RETR '+file, open(file,'wb').write)

#%% Read RTOFS grid and time
    
print('Retrieving coordinates from RTOFS')
RTOFS = xr.open_dataset(nc_files_RTOFS[0])
latRTOFS = np.asarray(RTOFS.Latitude[:])
lonRTOFS = np.asarray(RTOFS.Longitude[:])
depthRTOFS = np.asarray(RTOFS.Depth[:])

tRTOFS = []
for t in np.arange(len(nc_files_RTOFS)):
    RTOFS = xr.open_dataset(nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(RTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
          for t in np.arange(len(nc_files_RTOFS))])
    
oktimeRTOFS = np.where(np.logical_and(mdates.date2num(tRTOFS) >= mdates.date2num(tini), 
                                     mdates.date2num(tRTOFS) <= mdates.date2num(tend)))
timeRTOFS = tRTOFS[oktimeRTOFS[0]]

# interpolating transect X and Y to lat and lon 
oklonRTOFS = np.round(np.interp(target_lonRTOFS,lonRTOFS[0,:],np.arange(0,len(lonRTOFS[0,:])))).astype(int)
oklatRTOFS = np.round(np.interp(target_latRTOFS,latRTOFS[:,0],np.arange(0,len(latRTOFS[:,0])))).astype(int)    

#%% Downloading and reading Copernicus grid
    
COP_grid = xr.open_dataset(ncCOP_global)

latCOP_glob = np.asarray(COP_grid.latitude[:])
lonCOP_glob = np.asarray(COP_grid.longitude[:])
    
#%%

fig, ax = plt.subplots(figsize=(12, 6)) 
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
#ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)
ax.plot(X,Y,'-k')
ax.plot(x1,y1,'s',color='cyan',label='0 km')
ax.plot(x2,y2,'s',color='blue',label=str(np.round(dist[-1]))+' km')
ax.axis('scaled')
ax.legend(fontsize=14)
plt.title('Transect',fontsize=20)

file = folder + 'Anegada_passage_transect'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%  Figure GOFS 3.1 

max_valt = 31
min_valt = 2   
nlevelst = max_valt - min_valt + 1

max_vals = 34
min_vals = 38  
nlevelss = max_valt - min_valt + 1

target_tempGOFS = np.empty((len(depthGOFS),len(target_lonGOFS)))
target_tempGOFS[:] = np.nan
target_saltGOFS = np.empty((len(depthGOFS),len(target_lonGOFS)))
target_saltGOFS[:] = np.nan

print('Getting glider transect from GOFS')
for t,tind in enumerate(oktimeGOFS[0]):
    print('tind = ',tind)
    for pos in range(len(oklonGOFS)):
        print(len(oklonGOFS),pos)
        target_tempGOFS[:,pos] = GOFS.variables['water_temp'][tind,:,oklatGOFS[pos],oklonGOFS[pos]]
        target_saltGOFS[:,pos] = GOFS.variables['salinity'][tind,:,oklatGOFS[pos],oklonGOFS[pos]]
    
    fig, ax = plt.subplots(figsize=(9, 3))
    kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))
    
    cs = plt.contourf(dist,-depthGOFS,target_tempGOFS,cmap=cmocean.cm.thermal,**kw)
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('Temperature ($^oC$)',size=14)
    plt.contour(dist,-depthGOFS,target_tempGOFS,[26],colors='k')
    plt.title('GOFS Transect on ' + tGOFS[tind].strftime("%Y-%m-%d %H"),size=16)
    ax.set_ylim(-1000,0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Along Transect Distance (km)',fontsize=14)
    
    file = folder + 'Anegada_passage_transect_temp_GOFS'+ \
                        tGOFS[tind].strftime("%Y-%m-%d-%H")
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
    
    fig, ax = plt.subplots(figsize=(9, 3))
    kw = dict(levels = np.linspace(min_vals,max_vals,nlevelss)) 
    cs = plt.contourf(dist,-depthGOFS,target_saltGOFS,cmap=cmocean.cm.haline)#,**kw)
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('Salinity',size=14)
    plt.title('GOFS Transect on ' + tGOFS[tind].strftime("%Y-%m-%d %H"),size=16)
    ax.set_ylim(-1000,0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Along Transect Distance (km)',fontsize=14)
    
    file = folder + 'Anegada_passage_transect_salt_GOFS'+ \
                        tGOFS[tind].strftime("%Y-%m-%d-%H")
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
     
#%% Figures RTOFS  
    
max_valt = 31
min_valt = 2   
nlevelst = max_valt - min_valt + 1

max_vals = 34
min_vals = 38  
nlevelss = max_valt - min_valt + 1

target_tempRTOFS = np.empty((len(depthRTOFS),len(target_lonRTOFS)))
target_tempRTOFS[:] = np.nan
target_saltRTOFS = np.empty((len(depthRTOFS),len(target_lonRTOFS)))
target_saltRTOFS[:] = np.nan

print('Getting glider transect from RTOFS')
for tind in range(len(oktimeRTOFS[0])):
    print(len(oktimeRTOFS[0]),' ',i)
    for pos in range(len(oklonRTOFS)):
        print(len(oklonRTOFS),pos)
        nc_file = nc_files_RTOFS[tind]
        RTOFS = xr.open_dataset(nc_file)
        target_tempRTOFS[:,pos] = RTOFS.variables['temperature'][0,:,oklatRTOFS[pos],oklonRTOFS[pos]]
        target_saltRTOFS[:,pos] = RTOFS.variables['salinity'][0,:,oklatRTOFS[pos],oklonRTOFS[pos]]
    
    '''
    nc_file = nc_files_RTOFS[i]
    ncRTOFS = xr.open_dataset(nc_file)
    target_tempRTOFS[:,i] = ncRTOFS.variables['temperature'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
    target_saltRTOFS[:,i] = ncRTOFS.variables['salinity'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
    '''
    fig, ax = plt.subplots(figsize=(9, 3))
    kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))
    
    cs = plt.contourf(dist,-depthRTOFS,target_tempRTOFS,cmap=cmocean.cm.thermal,**kw)
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('Temperature ($^oC$)',size=14)
    plt.contour(dist,-depthRTOFS,target_tempRTOFS,[26],colors='k')
    plt.title('RTOFS Transect on ' + timeRTOFS[tind].strftime("%Y-%m-%d %H"),size=16)
    ax.set_ylim(-1000,0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Along Transect Distance (km)',fontsize=14)
    
    file = folder + 'Anegada_passage_transect_temp_RTOFS'+ \
                        tRTOFS[tind].strftime("%Y-%m-%d-%H")
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
    
    fig, ax = plt.subplots(figsize=(9, 3))
    kw = dict(levels = np.linspace(min_vals,max_vals,nlevelss)) 
    cs = plt.contourf(dist,-depthRTOFS,target_saltRTOFS,cmap=cmocean.cm.haline)#,**kw)
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('Salinity',size=14)
    plt.title('RTOFS Transect on ' + timeRTOFS[tind].strftime("%Y-%m-%d %H"),size=16)
    ax.set_ylim(-1000,0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Along Transect Distance (km)',fontsize=14)
    
    file = folder + 'Anegada_passage_transect_salt_RTOFS'+ \
                        tRTOFS[tind].strftime("%Y-%m-%d-%H")
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
    
#%% Downloading and reading Copernicus output
    
motuc = 'python -m motuclient --motu ' + url_cmems + \
        ' --service-id ' + service_id + \
        ' --product-id ' + product_id + \
        ' --longitude-min ' + str(np.min(X)-2/12) + \
        ' --longitude-max ' + str(np.max(X)+2/12) + \
        ' --latitude-min ' + str(np.min(Y)-2/12) + \
        ' --latitude-max ' + str(np.max(Y)+2/12) + \
        ' --date-min ' + str(tini-timedelta(0.5)) + \
        ' --date-max ' + str(tend+timedelta(0.5)) + \
        ' --depth-min ' + depth_min + \
        ' --depth-max ' + str(1000) + \
        ' --variable ' + 'thetao' + ' ' + \
        ' --variable ' + 'so'  + ' ' + \
        ' --out-dir ' + out_dir + \
        ' --out-name ' + 'Ane_trans' + '.nc' + ' ' + \
        ' --user ' + 'maristizabalvar' + ' ' + \
        ' --pwd ' +  'MariaCMEMS2018'

os.system(motuc)
#%%
COP_file = out_dir + '/' + 'Ane_trans'  + '.nc'
COP = xr.open_dataset(COP_file)

latCOP = np.asarray(COP.latitude[:])
lonCOP = np.asarray(COP.longitude[:])
depthCOP = np.asarray(COP.depth[:])
tCOP = np.asarray(mdates.num2date(mdates.date2num(COP.time[:])))

oktimeCOP = np.where(np.logical_and(mdates.date2num(tCOP) >= mdates.date2num(tini),\
                                        mdates.date2num(tCOP) <= mdates.date2num(tend)))
timeCOP = tCOP[oktimeCOP]
#%%
# interpolating transect X and Y to lat and lon 
oklonCOP = np.round(np.interp(target_lonCOP,lonCOP,np.arange(0,len(lonCOP)))).astype(int)
oklatCOP = np.round(np.interp(target_latCOP,latCOP,np.arange(0,len(latCOP)))).astype(int) 
    
target_tempCOP = np.empty((len(depthCOP),len(target_lonCOP)))
target_tempCOP[:] = np.nan
target_saltCOP = np.empty((len(depthCOP),len(target_lonCOP)))
target_saltCOP[:] = np.nan

print('Getting glider transect from Copernicus')
for tind in range(len(oktimeCOP[0])):
    print(len(oktimeCOP[0]),' ',tind)
    for pos in range(len(oklonCOP)):
        print(len(oklonCOP),pos)
        target_tempCOP[:,pos] = COP.variables['thetao'][oktimeCOP[0],:,oklatCOP[pos],oklonCOP[pos]]
        target_saltCOP[:,pos] = COP.variables['so'][oktimeCOP[0],:,oklatCOP[pos],oklonCOP[pos]]
    
    fig, ax = plt.subplots(figsize=(9, 3))
    kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))
    
    cs = plt.contourf(dist,-depthCOP,target_tempCOP,cmap=cmocean.cm.thermal,**kw)
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('Temperature ($^oC$)',size=14)
    plt.contour(dist,-depthCOP,target_tempCOP,[26],colors='k')
    plt.title('Copernicus Transect on ' + timeCOP[tind].strftime("%Y-%m-%d %H"),size=16)
    ax.set_ylim(-1000,0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Along Transect Distance (km)',fontsize=14)
    
    file = folder + 'Anegada_passage_transect_temp_COP'+ \
                        timeCOP[tind].strftime("%Y-%m-%d-%H")
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
    
    fig, ax = plt.subplots(figsize=(9, 3))
    kw = dict(levels = np.linspace(min_vals,max_vals,nlevelss)) 
    cs = plt.contourf(dist,-depthCOP,target_saltCOP,cmap=cmocean.cm.haline)#,**kw)
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('Salinity',size=14)
    plt.title('Copernicus Transect on ' + timeCOP[tind].strftime("%Y-%m-%d %H"),size=16)
    ax.set_ylim(-1000,0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xlabel('Along Transect Distance (km)',fontsize=14)
    
    file = folder + 'Anegada_passage_transect_salt_COP'+ \
                        timeCOP[tind].strftime("%Y-%m-%d-%H")
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)     