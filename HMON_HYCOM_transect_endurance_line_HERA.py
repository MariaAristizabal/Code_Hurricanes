#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:37:19 2020

@author: aristizabal
"""
#%%
storm_id = '94l'
cycle = '2020080312'

# MAB
lon_lim = [-77,-52]
lat_lim = [35,46]

#%% 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import os.path
import glob
import cmocean
from matplotlib.dates import date2num, num2date
import xarray as xr

import sys
sys.path.append('/home/Maria.Aristizabal/NCEP_scripts/')
from utils4HYCOM import readgrids
#from utils4HYCOM import readdepth, readVar
from utils4HYCOM2 import readBinz

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%%

ti = datetime.today()
ffig = '/home/Maria.Aristizabal/Figures/'+ str(ti.year) + '/' + ti.strftime('%b-%d') 
folder_fig =  ffig + '/' + storm_id + '_' + cycle + '/'

os.system('mkdir ' +  ffig)
os.system('mkdir ' +  folder_fig)

#%% Bathymetry file

bath_file = '/scratch2/NOS/nosofs/Maria.Aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

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

    #%% folder and file names    
ti = datetime.today() - timedelta(1)

folder_hmon_hycom = '/scratch2/NOS/nosofs/Maria.Aristizabal/HMON_HYCOM_' + storm_id + '_' + str(ti.year) + '/' + 'HMON_HYCOM_' + storm_id + '_' + cycle + '/'

#%% Reading RTOFS grid    
grid_file = sorted(glob.glob(os.path.join(folder_hmon_hycom,'*regional.grid.*')))[0][:-2]

#%% Reading RTOFS grid
print('Retrieving coordinates from RTOFS')
# Reading lat and lon
#lines_grid = [line.rstrip() for line in open(grid_file+'.b')]
lon_hycom = np.array(readgrids(grid_file,'plon:',[0]))
lat_hycom = np.array(readgrids(grid_file,'plat:',[0]))

#depth_HMON_HYCOM = np.asarray(readdepth(HMON_HYCOM_depth,'depth'))

# Reading depths
afiles = sorted(glob.glob(os.path.join(folder_hmon_hycom,'*hat10_3z'+'*.a')))
lines=[line.rstrip() for line in open(afiles[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]=='temp':
        #print(line.split()[1])
        z.append(float(line.split()[1]))
depth_HYCOM = np.asarray(z)

time_HYCOM = []
for x, file in enumerate(afiles):
    print(x)
    #lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    year = int(file.split('/')[-1].split('.')[1][0:4])
    month = int(file.split('/')[-1].split('.')[1][4:6])
    day = int(file.split('/')[-1].split('.')[1][6:8])
    hour = int(file.split('/')[-1].split('.')[1][8:10])
    dt = int(file.split('/')[-1].split('.')[-2][1:])
    timestamp_HYCOM = date2num(datetime(year,month,day,hour)) + dt/24
    time_HYCOM.append(num2date(timestamp_HYCOM))

# Reading 3D variable from binary file
oktime = 0 # first file 
temp_HMON_HYCOM = readBinz(afiles[oktime][:-2],'3z','temp')
#salt_HMON_HYCOM = readBinz(afiles[oktime][:-2],'3z','salinity')
#uvel_HMON_HYCOM = readBinz(afiles[oktime][:-2],'3z','u-veloc.')
#vvel_HMON_HYCOM = readBinz(afiles[oktime][:-2],'3z','v-veloc.')

#%% 
oklon_HYCOM = np.where(np.logical_and(lon_hycom[0,:] >= lon_lim[0]+360,lon_hycom[0,:] <= lon_lim[1]+360))[0]
oklat_HYCOM = np.where(np.logical_and(lat_hycom[:,0] >= lat_lim[0],lat_hycom[:,0] <= lat_lim[1]))[0]

lon_HYCOM =lon_hycom[0,oklon_HYCOM]-360
lat_HYCOM =lat_hycom[oklat_HYCOM,0]

#%%

x1 = -74.1
y1 = 39.4
x2 = -73.0
y2 = 38.6
# Slope
m = (y1-y2)/(x1-x2)
# Intercept
b = y1 - m*x1

X = np.arange(x1,-72,0.05)
Y = b + m*X

dist = np.sqrt((X-x1)**2 + (Y-y1)**2)*111 # approx along transect distance in km

oklon = np.round(np.interp(X,lon_hycom[0,:]-360,np.arange(len(lon_hycom[0,:])))).astype(int)
oklat = np.round(np.interp(Y,lat_hycom[:,0],np.arange(len(lat_hycom[:,0])))).astype(int)

trans_temp_HYCOM = temp_HMON_HYCOM[oklat,oklon,:]

min_valt = 8
max_valt = 27
nlevelst = max_valt - min_valt + 1
kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))

fig, ax = plt.subplots(figsize=(9, 3))
plt.contourf(dist,-depth_HYCOM,trans_temp_HYCOM.T,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(dist,-depth_HYCOM,trans_temp_HYCOM.T,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Along Transect Distance (km)',fontsize=14)
plt.title('HMON-HYCOM Endurance Line ' + 'Storm ' + storm_id + ' Cycle ' + cycle,fontsize=16)
plt.ylim([-100,0])
plt.xlim([0,200])

file = folder_fig + 'HMON_HYCOM_temp_MAB_endurance_line_cycle_'+cycle
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%

x1 = -90
y1 = 20 + 52/60
x2 = -90
y2 = 30

Y = np.arange(y1,y2,0.05)
X = np.tile(x1,len(Y))

dist = np.sqrt((X-x1)**2 + (Y-y1)**2)*111 # approx along transect distance in km

oklon = np.round(np.interp(X,lon_hycom[0,:]-360,np.arange(len(lon_hycom[0,:])))).astype(int)
oklat = np.round(np.interp(Y,lat_hycom[:,0],np.arange(len(lat_hycom[:,0])))).astype(int)

trans_temp_HYCOM = temp_HMON_HYCOM[oklat,oklon,:]

min_valt = 12
max_valt = 32
nlevelst = max_valt - min_valt + 1
kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))

fig, ax = plt.subplots(figsize=(9, 3))
plt.contourf(dist,-depth_HYCOM,trans_temp_HYCOM.T,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(dist,-depth_HYCOM,trans_temp_HYCOM.T,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Along Transect Distance (km)',fontsize=14)
plt.title('HMON-HYCOM Endurance Line ' + 'Storm ' + storm_id + ' Cycle ' + cycle,fontsize=16)
plt.ylim([-300,0])
#plt.xlim([0,200])

file = folder_fig + 'HMON_HYCOM_temp_GoMex_across_cycle_'+cycle
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
