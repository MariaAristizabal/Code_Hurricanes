#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:46:28 2019

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

#%% Defining points of interest

x1 = -(64 + 16/60 + 18.82/3600)
y1 = 18 + 20/60 + 54.93/3600

x2 = -(64 + 0/60 + 48.43/3600)
y2 = 18 + 16/60 + 31.83/3600

x3 = -(63 + 36/60 + 12.43/3600)
y3 = 18 + 15/60 + 0.77/3600

X = np.array([x1,x2,x3])
Y = np.array([y1,y2,y3])

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
        ' --out-name ' + 'Aneg_pass' + '.nc' + ' ' + \
        ' --user ' + 'maristizabalvar' + ' ' + \
        ' --pwd ' +  'MariaCMEMS2018'

os.system(motuc)

#%%
COP_file = out_dir + '/' + 'Aneg_pass' + '.nc'
COP = xr.open_dataset(COP_file)

latCOP = np.asarray(COP.latitude[:])
lonCOP = np.asarray(COP.longitude[:])
depthCOP = np.asarray(COP.depth[:])
tCOP = np.asarray(mdates.num2date(mdates.date2num(COP.time[:])))

oktimeCOP = np.where(np.logical_and(mdates.date2num(tCOP) >= mdates.date2num(tini),\
                                        mdates.date2num(tCOP) <= mdates.date2num(tend)))
timeCOP = tCOP[oktimeCOP]

# interpolating transect X and Y to lat and lon 
oklonCOP = np.round(np.interp(target_lonCOP,lonCOP,np.arange(0,len(lonCOP)))).astype(int)
oklatCOP = np.round(np.interp(target_latCOP,latCOP,np.arange(0,len(latCOP)))).astype(int) 
    
#%%

fig, ax = plt.subplots(figsize=(12, 6)) 
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
#ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)
plt.plot(X[0],Y[0],'*r',markersize = 10, label=str(np.round(X[0],5))+' '+str(np.round(Y[0],5)))
plt.plot(X[1],Y[1],'*b',markersize = 10,label=str(np.round(X[1],5))+' '+str(np.round(Y[1],5)))
plt.plot(X[2],Y[2],'*g',markersize = 10,label=str(np.round(X[2],5))+' '+str(np.round(Y[2],5)))
plt.axis('scaled')
plt.legend(fontsize=14)
plt.title('Profiles Locations',fontsize=20)

file = folder + 'Anegada_passage_profile_locations'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%  Figure temp and salinity profiles

col = ['r','b','g']

target_tempRTOFS = np.empty((len(timeRTOFS),len(depthRTOFS)))
target_tempRTOFS[:] = np.nan
target_saltRTOFS = np.empty((len(timeRTOFS),len(depthRTOFS)))
target_saltRTOFS[:] = np.nan

for pos in range(len(oklonGOFS)):
    print(len(oklonGOFS),pos)
    
    print('Getting glider profile from GOFS')
    target_tempGOFS = GOFS.variables['water_temp'][oktimeGOFS[0],:,oklatGOFS[pos],oklonGOFS[pos]]
    target_saltGOFS = GOFS.variables['salinity'][oktimeGOFS[0],:,oklatGOFS[pos],oklonGOFS[pos]]
    
    print('Getting glider profile from RTOFS')
    for tind,tt in enumerate(timeRTOFS):
        print(len(oktimeRTOFS[0]),' ',tind)
        nc_file = nc_files_RTOFS[tind]
        RTOFS = xr.open_dataset(nc_file)
        target_tempRTOFS[tind,:] = RTOFS.variables['temperature'][0,:,oklatRTOFS[pos],oklonRTOFS[pos]]
        target_saltRTOFS[tind,:] = RTOFS.variables['salinity'][0,:,oklatRTOFS[pos],oklonRTOFS[pos]]
    
    print('Getting glider profile from Copernicus')    
    target_tempCOP = COP.variables['thetao'][oktimeCOP[0],:,oklatCOP[pos],oklonCOP[pos]]
    target_saltCOP = COP.variables['so'][oktimeCOP[0],:,oklatCOP[pos],oklonCOP[pos]]

    
    fig, ax = plt.subplots(figsize=(5, 12))
    plt.plot(target_tempGOFS.T,-depthGOFS,'.-',color='lightcoral',label='_nolegend_')
    plt.plot(np.nanmean(target_tempGOFS,axis=0),-depthGOFS,'.-r',markersize=12,linewidth=2,\
             label='GOFS 3.1'+' '+str(timeGOFS[0].year)+' '+'['+str(timeGOFS[0])[5:13]+','+str(timeGOFS[-1])[5:13]+']')
    
    plt.plot(target_tempRTOFS.T,-depthRTOFS,'.-',color='mediumseagreen',label='_nolegend_')
    plt.plot(np.nanmean(target_tempRTOFS,axis=0),-depthRTOFS,'.-g',markersize=12,linewidth=2,\
             label='RTOFS'+' '+str(timeRTOFS[0].year)+' '+'['+str(timeRTOFS[0])[5:13]+','+str(timeRTOFS[-1])[5:13]+']')
    plt.plot(target_tempCOP.T,-depthCOP,'.-',color='plum',label='_nolegend_')
    plt.plot(np.nanmean(target_tempCOP,axis=0),-depthCOP,'.-',color='darkorchid',markersize=12,linewidth=2,\
             label='Copernicus'+' '+str(timeCOP[0].year)+' '+'['+str(timeCOP[0])[5:13]+','+str(timeCOP[-1])[5:13]+']')
    
    plt.ylabel('Depth (m)',fontsize=20)
    plt.xlabel('Temperature ($^oC$)',fontsize=20)
    plt.title('Temperature Profile at \n '+'[' + str(np.round(X[pos],5))+' , '+\
               str(np.round(Y[pos],5))+']',fontsize=20, color=col[pos])
    plt.ylim([-1100,0])
    plt.legend(loc='lower left',bbox_to_anchor=(-0.2,0.0),fontsize=14)
    plt.grid('on')
    
    file = folder + 'Anegada_passage_profile_temp_'+str(pos)
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
    
    fig, ax = plt.subplots(figsize=(5, 12))
    plt.plot(target_saltGOFS.T,-depthGOFS,'.-',color='lightcoral',label='_nolegend_')
    plt.plot(np.nanmean(target_saltGOFS,axis=0),-depthGOFS,'.-r',markersize=12,linewidth=2,\
             label='GOFS 3.1'+' '+str(timeGOFS[0].year)+' '+'['+str(timeGOFS[0])[5:13]+','+str(timeGOFS[-1])[5:13]+']')
    
    plt.plot(target_saltRTOFS.T,-depthRTOFS,'.-',color='mediumseagreen',label='_nolegend_')
    plt.plot(np.nanmean(target_saltRTOFS,axis=0),-depthRTOFS,'.-g',markersize=12,linewidth=2,\
             label='RTOFS'+' '+str(timeRTOFS[0].year)+' '+'['+str(timeRTOFS[0])[5:13]+','+str(timeRTOFS[-1])[5:13]+']')
    plt.plot(target_saltCOP.T,-depthCOP,'.-',color='plum',label='_nolegend_')
    plt.plot(np.nanmean(target_saltCOP,axis=0),-depthCOP,'.-',color='darkorchid',markersize=12,linewidth=2,\
             label='Copernicus'+' '+str(timeCOP[0].year)+' '+'['+str(timeCOP[0])[5:13]+','+str(timeCOP[-1])[5:13]+']')
    
    plt.ylabel('Depth (m)',fontsize=20)
    plt.xlabel('Salinity',fontsize=20)
    plt.title('Salinity Profile at \n '+'[' + str(np.round(X[pos],5))+' , '+\
               str(np.round(Y[pos],5))+']',fontsize=20,color=col[pos])
    plt.ylim([-1100,0])
    plt.legend(loc='lower left',bbox_to_anchor=(-0.2,0.0),fontsize=14)
    plt.grid('on')
    
    file = folder + 'Anegada_passage_profile_salt_'+str(pos)
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
    