#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:39:32 2020

@author: aristizabal
"""

#%% User input
cycle ='2020080118'
name = 'isaias'
storm_id = '09l'

# GoMex
lon_lim = [-87,-60]
lat_lim = [8,45]

# files for HMON-HYCOM output
folder_hycom = '/home/aristizabal/HMON_HYCOM_09l_2020/HMON_HYCOM_09l_' + cycle + '/'
prefix = 'hmon_rtofs_hat10_3z'
gridfile = folder_hycom + 'hwrf_rtofs_hat10.basin.regional.grid'
depthfile = folder_hycom + 'hwrf_rtofs_hat10.basin.regional.depth'

# Glider data 
url_glider = 'https://data.ioos.us/gliders/erddap'
#gdata = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20200715T1558/ru33-20200715T1558.nc3.nc'

folder_fig = '/home/aristizabal/Figures/'

#%% 
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.dates import date2num, num2date
import os
import os.path
import glob
import cmocean
from erddapy import ERDDAP
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates

import sys
sys.path.append('/home/aristizabal/NCEP_scripts/')
from utils4HYCOM import readgrids
#from utils4HYCOM import readdepth, readVar
#from utils4HYCOM2 import readBinz
from utils4HYCOM import readBinz

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%%
tini = datetime.datetime(2020, 8, 1, 18, 0)
tend = datetime.datetime(2020, 8, 6, 6, 0)

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

#%% Reading RTOFS grid
print('Retrieving coordinates from RTOFS')
# Reading lat and lon
lines_grid = [line.rstrip() for line in open(gridfile+'.b')]
lon_hycom = np.array(readgrids(gridfile,'plon:',[0]))
lat_hycom = np.array(readgrids(gridfile,'plat:',[0]))

#depth_HMON_HYCOM = np.asarray(readdepth(HMON_HYCOM_depth,'depth'))

# Reading depths
afiles = sorted(glob.glob(os.path.join(folder_hycom,'*'+prefix+'*.a')))
lines=[line.rstrip() for line in open(afiles[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]=='temp':
        #print(line.split()[1])
        z.append(float(line.split()[1]))
z_hycom = np.asarray(z) 

#%%
afiles = sorted(glob.glob(os.path.join(folder_hycom,'*'+prefix+'*.a')))
time_hycom = []
for x, file in enumerate(afiles):
    print(x)
    #lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    year = int(file.split('/')[-1].split('.')[1][0:4])
    month = int(file.split('/')[-1].split('.')[1][4:6])
    day = int(file.split('/')[-1].split('.')[1][6:8])
    hour = int(file.split('/')[-1].split('.')[1][8:10])
    dt = int(file.split('/')[-1].split('.')[-2][1:])
    timestamp_hycom = date2num(datetime(year,month,day,hour)) + dt/24
    time_hycom.append(num2date(timestamp_hycom))

# Reading 3D variable from binary file
#N = 0
#temp_hycom = readBinz(afiles[N][:-2],'3z','temp')

#%% Reading HMON-HYCOM ab files

nz = len(z_hycom) 

target_temp_HMON_HYCOM = np.empty((len(afiles),nz,))
target_temp_HMON_HYCOM[:] = np.nan
time_HMON_HYCOM = []
for x, file in enumerate(afiles):
    print(x)
    lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    year = int(file.split('.')[1][0:4])
    month = int(file.split('.')[1][4:6])
    day = int(file.split('.')[1][6:8])
    hour = int(file.split('.')[1][8:10])
    dt = int(file.split('.')[3][1:])
    timestamp_HMON_HYCOM = date2num(datetime(year,month,day,hour)) + dt/24
    time_HMON_HYCOM.append(num2date(timestamp_HMON_HYCOM))
    
    # Interpolating latg and longlider into RTOFS grid
    sublonHMON_HYCOM = np.interp(timestamp_HMON_HYCOM,timestampg,long+360)
    sublatHMON_HYCOM = np.interp(timestamp_HMON_HYCOM,timestampg,latg)
    oklonHMON_HYCOM = np.int(np.round(np.interp(sublonHMON_HYCOM,lon_hycom[0,:],np.arange(len(lon_hycom[0,:])))))
    oklatHMON_HYCOM = np.int(np.round(np.interp(sublatHMON_HYCOM,lat_hycom[:,0],np.arange(len(lat_hycom[:,0])))))
    
    # Reading 3D variable from binary file 
    temp_HMON_HYCOM = readBinz(file[:-2],'3z','temp')
    #ts=readBin(afile,'archive','temp')
    target_temp_HMON_HYCOM[x,:] = temp_HMON_HYCOM[oklatHMON_HYCOM,oklonHMON_HYCOM,:]
    
    # Extracting list of variables
    #count=0
    #for line in lines:
    #    count+=1
    #    if line[0:5] == 'field':
    #        break

    #lines=lines[count:]
    #vars=[line.split()[0] for line in lines]
    
time_HMON_HYCOM = np.asarray(time_HMON_HYCOM)
timestamp_HMON_HYCOM = date2num(time_HMON_HYCOM) 

#%% RU33

kw = dict(levels = np.arange(6,31,3))
fig, ax=plt.subplots(figsize=(10, 3))
plt.contour(timeg,-depthg_gridded,tempg_gridded,levels=[26],colors = 'k')
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=cmocean.cm.thermal,**kw)

plt.title(e.dataset_id,fontsize=16)
plt.ylim(-70,0)
plt.xlim(timeg[0],timeg[-1])
plt.xticks()
xfmt = mdates.DateFormatter('%d-%H')
ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('$^oC$',fontsize=14)
ax.set_ylabel('Depth (m)',fontsize=14)

file = folder_fig + 'Temp_' + e.dataset_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Temp transect HMON-HYCOM

kw = dict(levels = np.arange(6,31,3))
fig, ax = plt.subplots(figsize=(10, 3))
cs = plt.contourf(time_HMON_HYCOM,-z_hycom,target_temp_HMON_HYCOM.T,cmap=cmocean.cm.thermal,**kw)
plt.contour(time_HMON_HYCOM,-z_hycom,target_temp_HMON_HYCOM.T,[26],color='k')

plt.title('HMON-HYCOM Cycle ' + cycle + ' Along Track ' + e.dataset_id[0:4],fontsize=14)
plt.ylim(-70,0)
plt.xlim(timeg[0],timeg[-1])
time_vec = [datetime(2020,8,2,0),datetime(2020,8,2,12),datetime(2020,8,3,0),datetime(2020,8,3,12)]
plt.xticks(time_vec)
xfmt = mdates.DateFormatter('%b-%d-%H')
ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('$^oC$',fontsize=14)
ax.set_ylabel('Depth (m)',fontsize=14)

file = folder_fig + 'HMON_HYCOM_vs_RU33_' + storm_id + '_' + cycle
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

