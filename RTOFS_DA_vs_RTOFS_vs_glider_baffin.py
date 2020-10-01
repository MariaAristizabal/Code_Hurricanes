#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:57:38 2020

@author: aristizabal
"""

#%% User input

cycle = '2020042500'

# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

# urls
url_glider = 'https://data.ioos.us/gliders/erddap'

# folder ab files HYCOM
folder_RTOFS_DA = '/home/aristizabal/RTOFS-DA/data_' + cycle
prefix_RTOFS_DA = 'archv'

# RTOFS grid file name
folder_RTOFS_DA_grid_depth = '/home/aristizabal/RTOFS-DA/GRID_DEPTH/'
RTOFS_DA_grid = folder_RTOFS_DA_grid_depth + 'regional.grid'
RTOFS_DA_depth = folder_RTOFS_DA_grid_depth + 'regional.depth'

folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']
    
# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

folder_figs = '/home/aristizabal/Figures/'

#%% 
from erddapy import ERDDAP
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from datetime import datetime, timedelta
import os
import os.path
import glob

import sys
sys.path.append('/home/aristizabal/NCEP_scripts/')
#from utils4HYCOM_orig import readBinz 
from utils4HYCOM import readgrids, readdepth, readVar

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Time window

year = int(cycle[0:4])
month = int(cycle[4:6])
day = int(cycle[6:8])
hour = int(cycle[9:])
tini = datetime(year, month, day, hour)
tend = tini + timedelta(days=1)

#%% Reading bathymetry data
ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

#%% Look for datasets in IOOS glider dac
print('Looking for glider data sets')
e = ERDDAP(server = url_glider)

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
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

# Setting constraints
constraints = {
        'time>=': str(tini),
        'time<=': str(tend),
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }

variables = [
        'depth',
        'latitude',
        'longitude',
        'time',
        'temperature',
        'salinity'
        ]

e = ERDDAP(
        server=url_glider,
        protocol='tabledap',
        response='nc'
        )

#%% Read RTOFS grid and time
print('Retrieving coordinates from RTOFS')

if tini.month < 10:
    if tini.day < 10:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day)
else:
    if tini.day < 10:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day)
    else:
        fol = 'rtofs.' + str(tini.year) + str(tini.month) + str(tini.day)

ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[0])
latRTOFS = np.asarray(ncRTOFS.Latitude[:])
lonRTOFS = np.asarray(ncRTOFS.Longitude[:])
depthRTOFS = np.asarray(ncRTOFS.Depth[:])

tRTOFS = []
for t in np.arange(len(nc_files_RTOFS)):
    ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
         for t in np.arange(len(nc_files_RTOFS))])

#%% Reading RTOFS-DA  grid
print('Retrieving coordinates from RTOFS')
# Reading lat and lon
lines_grid = [line.rstrip() for line in open(RTOFS_DA_grid+'.b')]
lon_RTOFS_DA = np.array(readgrids(RTOFS_DA_grid,'plon:',[0]))
lat_RTOFS_DA = np.array(readgrids(RTOFS_DA_grid,'plat:',[0]))

depth_RTOFS_DA = np.asarray(readdepth(RTOFS_DA_depth,'depth'))

#%% Looping through all gliders found
for id in gliders[1:]:    
    #id=gliders[1]
    print('Reading ' + id )
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables

    # Converting glider data to data frame
    df = e.to_pandas(
            index_col='time (UTC)',
            parse_dates=True,
            skiprows=(1,)  # units information can be dropped.
            ).dropna()

    # Coverting glider vectors into arrays
    timeg, ind = np.unique(df.index.values,return_index=True)
    latg = df['latitude (degrees_north)'].values[ind]
    long = df['longitude (degrees_east)'].values[ind]

    dg = df['depth (m)'].values
    #vg = df['temperature (degree_Celsius)'].values
    tg = df[df.columns[3]].values
    sg = df[df.columns[4]].values

    delta_z = 0.3
    zn = np.int(np.round(np.max(dg)/delta_z))

    depthg = np.empty((zn,len(timeg)))
    depthg[:] = np.nan
    tempg = np.empty((zn,len(timeg)))
    tempg[:] = np.nan
    saltg = np.empty((zn,len(timeg)))
    saltg[:] = np.nan

    # Grid variables
    depthg_gridded = np.arange(0,np.nanmax(dg),delta_z)
    tempg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    tempg_gridded[:] = np.nan
    saltg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    saltg_gridded[:] = np.nan

    for i,ii in enumerate(ind):
        if i < len(timeg)-1:
            depthg[0:len(dg[ind[i]:ind[i+1]]),i] = dg[ind[i]:ind[i+1]]
            tempg[0:len(tg[ind[i]:ind[i+1]]),i] = tg[ind[i]:ind[i+1]]
            saltg[0:len(sg[ind[i]:ind[i+1]]),i] = sg[ind[i]:ind[i+1]]
        else:
            depthg[0:len(dg[ind[i]:len(dg)]),i] = dg[ind[i]:len(dg)]
            tempg[0:len(tg[ind[i]:len(tg)]),i] = tg[ind[i]:len(tg)]
            saltg[0:len(sg[ind[i]:len(sg)]),i] = sg[ind[i]:len(sg)]

    for t,tt in enumerate(timeg):
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
            okd = np.logical_and(depthg_gridded >= np.min(depthf[oks]),\
                                 depthg_gridded < np.max(depthf[oks]))
            saltg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[oks],saltf[oks])

    '''
    # Conversion from glider longitude and latitude to GOFS convention
    target_lonGOFS = np.empty((len(long),))
    target_lonGOFS[:] = np.nan
    for i,ii in enumerate(long):
        if ii < 0:
            target_lonGOFS[i] = 360 + ii
        else:
            target_lonGOFS[i] = ii
    target_latGOFS = latg
    '''
    
    # Conversion from glider longitude and latitude to RTOFS convention
    target_lonRTOFS = long
    target_latRTOFS = latg

    # Narrowing time window of RTOFS to coincide with glider time window
    #tmin = tini
    #tmax = tend
    tmin = mdates.num2date(mdates.date2num(timeg[0]))
    tmax = mdates.num2date(mdates.date2num(timeg[-1]))
    oktimeRTOFS = np.where(np.logical_and(tRTOFS >= tmin, tRTOFS <= tmax))
    timeRTOFS = mdates.num2date(mdates.date2num(tRTOFS[oktimeRTOFS]))

    # Changing times to timestamp
    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
    tstamp_RTOFS = [mdates.date2num(timeRTOFS[i]) for i in np.arange(len(timeRTOFS))]

    # interpolating glider lon and lat to lat and lon on RTOFS time
    sublonRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,target_lonRTOFS)
    sublatRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,target_latRTOFS)

    # getting the model grid positions for sublonm and sublatm
    oklonRTOFS = np.round(np.interp(sublonRTOFS,lonRTOFS[0,:],np.arange(len(lonRTOFS[0,:])))).astype(int)
    oklatRTOFS = np.round(np.interp(sublatRTOFS,latRTOFS[:,0],np.arange(len(latRTOFS[:,0])))).astype(int)

    # Getting glider transect from RTOFS
    print('Getting glider transect from RTOFS')
    target_tempRTOFS = np.empty((len(depthRTOFS),len(oktimeRTOFS[0])))
    target_tempRTOFS[:] = np.nan
    for i in range(len(oktimeRTOFS[0])):
        print(len(oktimeRTOFS[0]),' ',i)
        nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[i]
        ncRTOFS = xr.open_dataset(nc_file)
        target_tempRTOFS[:,i] = ncRTOFS.variables['temperature'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
    target_tempRTOFS[target_tempRTOFS < -100] = np.nan

    target_saltRTOFS = np.empty((len(depthRTOFS),len(oktimeRTOFS[0])))
    target_saltRTOFS[:] = np.nan
    for i in range(len(oktimeRTOFS[0])):
        print(len(oktimeRTOFS[0]),' ',i)
        nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[i]
        ncRTOFS = xr.open_dataset(nc_file)
        target_saltRTOFS[:,i] = ncRTOFS.variables['salinity'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
    target_saltRTOFS[target_saltRTOFS < -100] = np.nan

    # Reading RTOFS-DA ab files    
    afiles = sorted(glob.glob(os.path.join(folder_RTOFS_DA,prefix_RTOFS_DA+'*.a')))
    nz = 41
    layers = np.arange(0,nz)
    target_temp_RTOFS_DA = np.empty((len(afiles[0:8]),nz))
    target_temp_RTOFS_DA[:] = np.nan
    target_salt_RTOFS_DA = np.empty((len(afiles[0:8]),nz))
    target_salt_RTOFS_DA[:] = np.nan
    target_zRTOFS_DA = np.empty((len(afiles[0:8]),nz))
    target_zRTOFS_DA[:] = np.nan
    time_RTOFS_DA = []
    oklonRTOFS_DA = []
    oklatRTOFS_DA = []
    
    target_lon = long + 360
    target_lat = latg
    
    for tt,file in enumerate(afiles[0:8]):
        print(file)
        #file = afiles[0]
        lines = [line.rstrip() for line in open(file[:-2]+'.b')]
        time_stamp = lines[-1].split()[2]
        hycom_days = lines[-1].split()[3]
        tzero=datetime(1901,1,1,0,0)
        timeRT = tzero+timedelta(float(hycom_days)-1)
        time_RTOFS_DA.append(timeRT)
        timestampRTOFS = mdates.date2num(timeRT)
        
        sublonRTOFS = np.interp(timestampRTOFS,tstamp_glider,target_lon)
        sublatRTOFS = np.interp(timestampRTOFS,tstamp_glider,target_lat)
        oklonRTOFS_da = np.int(np.round(np.interp(sublonRTOFS,lon_RTOFS_DA[0,:],np.arange(len(lon_RTOFS_DA[0,:])))))
        oklatRTOFS_da = np.int(np.round(np.interp(sublatRTOFS,lat_RTOFS_DA[:,0],np.arange(len(lat_RTOFS_DA[:,0])))))
        oklonRTOFS_DA.append(oklonRTOFS_da)
        oklatRTOFS_DA.append(oklatRTOFS_da)
        
        #ztmp=readVar(file[:-2],'archive','srfhgt',[0])*0.01 # converts [cm] to [m]
        #target_srfhgt = ztmp[oklatRTOFS_DA,oklonRTOFS_DA]  
        target_ztmp = [0]
        for lyr in tuple(layers):
            print(lyr)
            temp_RTOFS = readVar(file[:-2],'archive','temp',[lyr+1])
            target_temp_RTOFS_DA[tt,lyr] = temp_RTOFS[oklatRTOFS_da,oklonRTOFS_da]
            
            salt_RTOFS = readVar(file[:-2],'archive','salin',[lyr+1])
            target_salt_RTOFS_DA[tt,lyr] = salt_RTOFS[oklatRTOFS_da,oklonRTOFS_da]
            
            dp = readVar(file[:-2],'archive','thknss',[lyr+1])/9806
            target_ztmp = np.append(target_ztmp,dp[oklatRTOFS_da,oklonRTOFS_da])
            
        target_zRTOFS_DA[tt,:] = np.cumsum(target_ztmp[0:-1]) + np.diff(np.cumsum(target_ztmp))/2    

    time_RTOFS_DA = np.asarray(time_RTOFS_DA)
    oklonRTOFS_DA = np.array(oklonRTOFS_DA)
    oklatRTOFS_DA = np.array(oklatRTOFS_DA)
    
    # Convertion from RTOFS to glider convention
    lonRTOFSg = lonRTOFS[0,:]
    latRTOFSg = latRTOFS[:,0]

    meshlonRTOFSg = np.meshgrid(lonRTOFSg,latRTOFSg)
    meshlatRTOFSg = np.meshgrid(latRTOFSg,lonRTOFSg)
    
    # Convertion from RTOFS-DA to glider convention
    lon_RTOFS_DAg = np.empty((len(lon_RTOFS_DA[0,:]),))
    lon_RTOFS_DAg[:] = np.nan
    for i,ii in enumerate(lon_RTOFS_DA[0,:]):
        if ii >= 180.0:
            lon_RTOFS_DAg[i] = ii - 360
        else:
            lon_RTOFS_DAg[i] = ii
    lat_RTOFS_DAg = lat_RTOFS_DA[:,0]

    meshlonRTOFS_DAg = np.meshgrid(lon_RTOFS_DAg,lat_RTOFS_DAg)
    meshlatRTOFS_DAg = np.meshgrid(lat_RTOFS_DAg,lon_RTOFS_DAg)
    
    # min and max values for plotting
    min_temp = np.floor(np.min([np.nanmin(df[df.columns[3]]),np.nanmin(target_tempRTOFS)]))
    max_temp = np.ceil(np.max([np.nanmax(df[df.columns[3]]),np.nanmax(target_tempRTOFS)]))

    min_salt = np.floor(np.min([np.nanmin(df[df.columns[4]]),np.nanmin(target_saltRTOFS)]))
    max_salt = np.ceil(np.max([np.nanmax(df[df.columns[4]]),np.nanmax(target_saltRTOFS)]))

    # Temperature profile
    fig, ax = plt.subplots(figsize=(14, 12))
    grid = plt.GridSpec(5, 2, wspace=0.4, hspace=0.3)

    ax = plt.subplot(grid[:,0])
    plt.plot(tempg,-depthg,'.',color='cyan',label='_nolegend_')
    plt.plot(np.nanmean(tempg_gridded,axis=1),-depthg_gridded,'.-b',\
             label=id[:-14]+' '+str(timeg[0])[0:4]+' '+'['+str(timeg[0])[5:19]+','+str(timeg[-1])[5:19]+']')
    plt.plot(target_tempRTOFS,-depthRTOFS,'.-',color='mediumseagreen',label='_nolegend_')
    plt.plot(np.nanmean(target_tempRTOFS,axis=1),-depthRTOFS,'.-g',markersize=12,linewidth=2,\
             label='RTOFS'+' '+str(timeRTOFS[0].year)+' '+'['+str(timeRTOFS[0])[5:13]+','+str(timeRTOFS[-1])[5:13]+']')
    plt.plot(target_temp_RTOFS_DA.T,-target_zRTOFS_DA.T,'.-',color='bisque',label='_nolegend_')
    plt.plot(np.nanmean(target_temp_RTOFS_DA.T,axis=1),-np.nanmean(target_zRTOFS_DA.T,axis=1),'.-',\
             color='sandybrown',markersize=12,linewidth=2,\
             label='RTOFS-DA '+str(time_RTOFS_DA[0].year)+' '+'['+str(time_RTOFS_DA[0])[5:13]+','+str(time_RTOFS_DA[-1])[5:13]+']')
    plt.ylabel('Depth (m)',fontsize=20)
    plt.xlabel('Temperature ($^oC$)',fontsize=20)
    plt.title('Temperature Profile ' + id,fontsize=20)
    plt.ylim([-np.nanmax(depthg)-100,0.1])
    plt.legend(loc='lower left',bbox_to_anchor=(-0.2,0.0),fontsize=14)
    plt.grid('on')

    # lat and lon bounds of box to draw
    minlonRTOFS = np.min(meshlonRTOFSg[0][oklatRTOFS.min()-2:oklatRTOFS.max()+2,oklonRTOFS.min()-2:oklonRTOFS.max()+2])
    maxlonRTOFS = np.max(meshlonRTOFSg[0][oklatRTOFS.min()-2:oklatRTOFS.max()+2,oklonRTOFS.min()-2:oklonRTOFS.max()+2])
    minlatRTOFS = np.min(meshlatRTOFSg[0][oklonRTOFS.min()-2:oklonRTOFS.max()+2,oklatRTOFS.min()-2:oklatRTOFS.max()+2])
    maxlatRTOFS = np.max(meshlatRTOFSg[0][oklonRTOFS.min()-2:oklonRTOFS.max()+2,oklatRTOFS.min()-2:oklatRTOFS.max()+2])

    # Getting subdomain for plotting glider track on bathymetry
    oklatbath = np.logical_and(bath_lat >= np.min(latg)-5,bath_lat <= np.max(latg)+5)
    oklonbath = np.logical_and(bath_lon >= np.min(long)-5,bath_lon <= np.max(long)+5)

    bath_latsub = bath_lat[oklatbath]
    bath_lonsub = bath_lon[oklonbath]
    bath_elevs = bath_elev[oklatbath,:]
    bath_elevsub = bath_elevs[:,oklonbath]

    ax = plt.subplot(grid[0:2,1])
    plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    #plt.axis([np.min(long)-5,np.max(long)+5,np.min(latg)-5,np.max(latg)+5])
    plt.plot(long,latg,'.-',color='orange',label='Glider track',\
             markersize=3)
    rect = patches.Rectangle((minlonRTOFS,minlatRTOFS),\
                             maxlonRTOFS-minlonRTOFS,maxlatRTOFS-minlatRTOFS,\
                             linewidth=1,edgecolor='k',facecolor='none')
    ax.add_patch(rect)
    plt.title('Glider track and model grid positions',fontsize=20)
    plt.axis('scaled')
    #plt.legend(loc='upper center',bbox_to_anchor=(1.1,1))
    plt.legend(loc='upper left')

    ax = plt.subplot(grid[2,1])
    ax.plot(long,latg,'.-',color='orange',label='Glider track')
    ax.plot(long[0],latg[0],'sc',label='Initial profile time '+str(timeg[0])[0:16])
    ax.plot(long[-1],latg[-1],'sb',label='final profile time '+str(timeg[-1])[0:16])
    ax.plot(meshlonRTOFSg[0][oklatRTOFS.min()-2:oklatRTOFS.max()+2,oklonRTOFS.min()-2:oklonRTOFS.max()+2],\
        meshlatRTOFSg[0][oklonRTOFS.min()-2:oklonRTOFS.max()+2,oklatRTOFS.min()-2:oklatRTOFS.max()+2].T,\
        '.',color='green')
    ax.plot(lonRTOFSg[oklonRTOFS],latRTOFSg[oklatRTOFS],'og',\
            markersize=8,markeredgecolor='k',\
            label='RTOFS grid points \n nx = ' + str(oklonRTOFS) \
            + '\n ny = ' + str(oklatRTOFS) \
            + '\n [day][hour] = ' + str(np.unique([str(i)[8:10] for i in timeRTOFS])) \
               + str([str(i)[11:13]for i in timeRTOFS]))
    ax.plot(meshlonRTOFS_DAg[0][oklatRTOFS_DA.min()-2:oklatRTOFS_DA.max()+2,oklonRTOFS_DA.min()-2:oklonRTOFS_DA.max()+2],\
        meshlatRTOFS_DAg[0][oklonRTOFS_DA.min()-2:oklonRTOFS_DA.max()+2,oklatRTOFS_DA.min()-2:oklatRTOFS_DA.max()+2].T,\
        '.',color='sandybrown')
    ax.plot(lon_RTOFS_DAg[oklonRTOFS_DA],lat_RTOFS_DAg[oklatRTOFS_DA],'o',color='sandybrown',\
            markersize=8,markeredgecolor='k',\
            label='RTOFS-DA grid points \n nx = ' + str(oklonRTOFS_DA) \
            + '\n ny = ' + str(oklatRTOFS_DA) \
            + '\n [day][hour] = ' + str(np.unique([str(i)[8:10] for i in time_RTOFS_DA])) \
               + str([str(i)[11:13]for i in time_RTOFS_DA]))
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.3))

    file = folder_figs +'temp_profile_' + id + '_' + str(tini).split()[0] + '_' + str(tend).split()[0]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

    # Salinity profile
    fig, ax = plt.subplots(figsize=(14, 12))
    grid = plt.GridSpec(5, 2, wspace=0.4, hspace=0.3)

    ax = plt.subplot(grid[:,0])
    plt.plot(saltg,-depthg,'.',color='cyan')
    plt.plot(np.nanmean(saltg_gridded,axis=1),-depthg_gridded,'.-b',\
             label=id[:-14]+' '+str(timeg[0])[0:4]+' '+'['+str(timeg[0])[5:19]+','+str(timeg[-1])[5:19]+']')
    plt.plot(target_saltRTOFS,-depthRTOFS,'.-',color='mediumseagreen',label='_nolegend_')
    plt.plot(np.nanmean(target_saltRTOFS,axis=1),-depthRTOFS,'.-g',markersize=12,linewidth=2,\
             label='RTOFS'+' '+str(timeRTOFS[0].year)+' '+'['+str(timeRTOFS[0])[5:13]+','+str(timeRTOFS[-1])[5:13]+']')
    plt.plot(target_salt_RTOFS_DA.T,-target_zRTOFS_DA.T,'.-',color='bisque',label='_nolegend_')
    plt.plot(np.nanmean(target_salt_RTOFS_DA.T,axis=1),-np.nanmean(target_zRTOFS_DA.T,axis=1),'.-',\
             color='sandybrown',markersize=12,linewidth=2,\
             label='RTOFS-DA '+str(time_RTOFS_DA[0].year)+' '+'['+str(time_RTOFS_DA[0])[5:13]+','+str(time_RTOFS_DA[-1])[5:13]+']')
    plt.ylabel('Depth (m)',fontsize=20)
    plt.xlabel('Salinity',fontsize=20)
    plt.title('Salinity Profile ' + id,fontsize=20)
    plt.ylim([-np.nanmax(depthg)-100,0.1])
    #plt.ylim([-200,0.1])
    plt.legend(loc='lower left',bbox_to_anchor=(-0.2,0.0),fontsize=14)
    plt.grid('on')

    # lat and lon bounds of box to draw
    minlonRTOFS = np.min(meshlonRTOFSg[0][oklatRTOFS.min()-2:oklatRTOFS.max()+2,oklonRTOFS.min()-2:oklonRTOFS.max()+2])
    maxlonRTOFS = np.max(meshlonRTOFSg[0][oklatRTOFS.min()-2:oklatRTOFS.max()+2,oklonRTOFS.min()-2:oklonRTOFS.max()+2])
    minlatRTOFS = np.min(meshlatRTOFSg[0][oklonRTOFS.min()-2:oklonRTOFS.max()+2,oklatRTOFS.min()-2:oklatRTOFS.max()+2])
    maxlatRTOFS = np.max(meshlatRTOFSg[0][oklonRTOFS.min()-2:oklonRTOFS.max()+2,oklatRTOFS.min()-2:oklatRTOFS.max()+2])

    # Getting subdomain for plotting glider track on bathymetry
    oklatbath = np.logical_and(bath_lat >= np.min(latg)-5,bath_lat <= np.max(latg)+5)
    oklonbath = np.logical_and(bath_lon >= np.min(long)-5,bath_lon <= np.max(long)+5)

    ax = plt.subplot(grid[0:2,1])
    bath_latsub = bath_lat[oklatbath]
    bath_lonsub = bath_lon[oklonbath]
    bath_elevs = bath_elev[oklatbath,:]
    bath_elevsub = bath_elevs[:,oklonbath]

    plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    plt.axis([np.min(long)-5,np.max(long)+5,np.min(latg)-5,np.max(latg)+5])
    plt.plot(long,latg,'.-',color='orange',label='Glider track',\
             markersize=3)
    rect = patches.Rectangle((minlonRTOFS,minlatRTOFS),\
                             maxlonRTOFS-minlonRTOFS,maxlatRTOFS-minlatRTOFS,\
                             linewidth=1,edgecolor='k',facecolor='none')
    ax.add_patch(rect)
    plt.title('Glider track and model grid positions',fontsize=20)
    plt.axis('scaled')
    #plt.legend(loc='upper center',bbox_to_anchor=(1.1,1))
    plt.legend(loc='upper left')

    ax = plt.subplot(grid[2,1])
    ax.plot(long,latg,'.-',color='orange',label='Glider track')
    ax.plot(long[0],latg[0],'sc',label='Initial profile time '+str(timeg[0])[0:16])
    ax.plot(long[-1],latg[-1],'sb',label='final profile time '+str(timeg[-1])[0:16])
    ax.plot(meshlonRTOFSg[0][oklatRTOFS.min()-2:oklatRTOFS.max()+2,oklonRTOFS.min()-2:oklonRTOFS.max()+2],\
        meshlatRTOFSg[0][oklonRTOFS.min()-2:oklonRTOFS.max()+2,oklatRTOFS.min()-2:oklatRTOFS.max()+2].T,\
        '.',color='green')
    ax.plot(lonRTOFSg[oklonRTOFS],latRTOFSg[oklatRTOFS],'og',\
            markersize=8,markeredgecolor='k',\
            label='RTOFS grid points \n nx = ' + str(oklonRTOFS) \
            + '\n ny = ' + str(oklatRTOFS) \
            + '\n [day][hour] = ' + str(np.unique([str(i)[8:10] for i in timeRTOFS])) \
               + str([str(i)[11:13]for i in timeRTOFS]))
    ax.plot(meshlonRTOFS_DAg[0][oklatRTOFS_DA.min()-2:oklatRTOFS_DA.max()+2,oklonRTOFS_DA.min()-2:oklonRTOFS_DA.max()+2],\
        meshlatRTOFS_DAg[0][oklonRTOFS_DA.min()-2:oklonRTOFS_DA.max()+2,oklatRTOFS_DA.min()-2:oklatRTOFS_DA.max()+2].T,\
        '.',color='sandybrown')
    ax.plot(lon_RTOFS_DAg[oklonRTOFS_DA],lat_RTOFS_DAg[oklatRTOFS_DA],'o',color='sandybrown',\
            markersize=8,markeredgecolor='k',\
            label='RTOFS-DA grid points \n nx = ' + str(oklonRTOFS_DA) \
            + '\n ny = ' + str(oklatRTOFS_DA) \
            + '\n [day][hour] = ' + str(np.unique([str(i)[8:10] for i in time_RTOFS_DA])) \
               + str([str(i)[11:13]for i in time_RTOFS_DA]))
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.3))

    file = folder_figs + 'salt_profile_' + id + '_' + str(tini).split()[0] + '_' + str(tend).split()[0]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)