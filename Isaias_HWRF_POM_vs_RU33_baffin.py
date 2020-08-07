#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:53:52 2020

@author: aristizabal
"""

#%% User input
cycle ='2020080118'
name = 'isaias'
storm_id = '09l'

# GoMex
lon_lim = [-87,-60]
lat_lim = [8,45]

folder_pom = '/home/aristizabal/HWRF_POM_09l_2020/HWRF_POM_09l_' + cycle + '/'

# POM grid file name
grid_file = folder_pom + name + storm_id + '.' + cycle + '.pom.grid.nc'

# POM files
prefix = name + storm_id + '.' + cycle + '.pom.'

# Glider data 
url_glider = 'https://data.ioos.us/gliders/erddap'
#gdata = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20200715T1558/ru33-20200715T1558.nc3.nc'

folder_fig = '/home/aristizabal/Figures/'

#%% 
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.dates import date2num, num2date
import os
import os.path
import glob
import cmocean
import netCDF4
from netCDF4 import Dataset
from erddapy import ERDDAP
import pandas as pd
import datetime as datetime, timedelta
import matplotlib.dates as mdates

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)


#%%

tini = datetime.datetime(2020, 8, 1, 18, 0)
tend = datetime.datetime(2020, 8, 6, 6, 0)

#%% Reading glider data
'''
ncglider = xr.open_dataset(gdata,decode_times=False)
#ncglider = Dataset(gdata,decode_times=False)
inst_id = ncglider.id.split('-')[0]
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)
tempglider = np.array(ncglider.temperature[0,:,:])
saltglider = np.array(ncglider.salinity[0,:,:])
depthglider = np.array(ncglider.depth[0,:,:])

timestamp_glider = date2num(time_glider)[0]

# Conversion from glider longitude and latitude to RTOFS convention
target_lon = []
for lon in longlider[0,:]:
    if lon < 0: 
        target_lon.append(360 + lon)
    else:
        target_lon.append(lon)
target_lon = np.array(target_lon)
target_lat = np.array(latglider[0,:])

#%%
tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

okg = np.where(np.logical_and(time_glider.T >= tmin, time_glider.T <= tmax))

timeg = time_glider[0,okg[0]]
timestampg = timestamp_glider[okg[0]]
latg = np.asarray(latglider[0,okg[0]])
long = np.asarray(longlider[0,okg[0]])
target_latg = target_lat[okg[0]]
target_long = target_lon[okg[0]]
depthg = depthglider[okg[0],:]
tempg = tempglider[okg[0],:]
saltg = saltglider[okg[0],:]
'''
#%%
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

#%% RU33
e.dataset_id = 'ru33-20200715T1558'
e.constraints = constraints
e.variables = variables

# checking data frame is not empty
df = e.to_pandas()
if len(df.index) != 0 :

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

timestampg = date2num(timeg)

#%% Reading POM grid files
pom_grid = xr.open_dataset(grid_file)
lon_POM = np.asarray(pom_grid['east_e'][:])
lat_POM = np.asarray( pom_grid['north_e'][:])
zlevc = np.asarray(pom_grid['zz'][:])
topoz = np.asarray(pom_grid['h'][:])


#%% Reading POM temperature

ncfiles = sorted(glob.glob(os.path.join(folder_pom,prefix+'*0*.nc')))

target_temp_pom = np.empty((len(ncfiles),len(zlevc),))
target_temp_pom[:] = np.nan
target_topoz_pom = np.empty((len(ncfiles),))
target_topoz_pom[:] = np.nan
time_pom = []
for x,file in enumerate(ncfiles):
    print(x)
    pom = xr.open_dataset(file)

    tpom = pom['time'][:]
    timestamp_pom = date2num(tpom)[0]
    time_pom.append(num2date(timestamp_pom))

    # Interpolating latg and longlider into RTOFS grid
    sublonpom = np.interp(timestamp_pom,timestampg,long)
    sublatpom = np.interp(timestamp_pom,timestampg,latg)
    oklonpom = np.int(np.round(np.interp(sublonpom,lon_POM[0,:],np.arange(len(lon_POM[0,:])))))
    oklatpom = np.int(np.round(np.interp(sublatpom,lat_POM[:,0],np.arange(len(lat_POM[:,0])))))

    target_temp_pom[x,:] = np.asarray(pom['t'][0,:,oklatpom,oklonpom])
    target_topoz_pom[x] = np.asarray(topoz[oklatpom,oklonpom])

timestamp_pom = date2num(time_pom)

z_matrix_pom = np.dot(target_topoz_pom.reshape(-1,1),zlevc.reshape(1,-1))


#%% RU33

kw = dict(levels = np.arange(6,31,3))
fig, ax=plt.subplots(figsize=(10, 3))
plt.contour(timeg,-depthg_gridded,tempg_gridded,levels=[26],colors = 'k')
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=cmocean.cm.thermal,**kw)

plt.title(e.dataset_id,fontsize=16)
plt.ylim(-70,0)
plt.xlim(timeg[0],timeg[-1])
time_vec = [datetime(2020,8,2,0),datetime(2020,8,2,12),datetime(2020,8,3,0),datetime(2020,8,3,12)]
plt.xticks(time_vec)
xfmt = mdates.DateFormatter('%b-%d-%H')
ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('$^oC$',fontsize=14)
ax.set_ylabel('Depth (m)',fontsize=14)

file = folder_fig + 'Temp_' + e.dataset_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Temp transect POM
time_matrix_pom = np.tile(time_pom,(z_matrix_pom.shape[1],1)).T

kw = dict(levels = np.arange(6,31,3))
fig, ax = plt.subplots(figsize=(10, 3))
cs = plt.contourf(time_matrix_pom,z_matrix_pom,target_temp_pom,cmap=cmocean.cm.thermal,**kw)
plt.contour(time_matrix_pom,z_matrix_pom,target_temp_pom,[26],color='k')

plt.title('HWRF-MPIPOM Cycle ' + cycle + ' Along Track ' + e.dataset_id[0:4],fontsize=14)
plt.ylim(-70,0)
plt.xlim(timeg[0],timeg[-1])
time_vec = [datetime(2020,8,2,0),datetime(2020,8,2,12),datetime(2020,8,3,0),datetime(2020,8,3,12)]
plt.xticks(time_vec)
xfmt = mdates.DateFormatter('%b-%d-%H')
ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('$^oC$',fontsize=14)
ax.set_ylabel('Depth (m)',fontsize=14)

file = folder_fig + 'HWRF_POM_vs_RU33_' + storm_id + '_' + cycle
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

timeg[0] + timedelta(hours=6)