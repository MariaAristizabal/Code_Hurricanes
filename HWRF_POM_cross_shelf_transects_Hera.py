#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:41:55 2020

@author: aristizabal
"""

#%%
storm_id = '94l'
cycle = '2020080312'

# MAB
lon_lim = [-77,-52]
lat_lim = [35,46]


#%%
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.dates import date2num, num2date
from datetime import datetime, timedelta
import os
import os.path
import glob
import cmocean

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

folder_hwrf_pom = '/scratch2/NOS/nosofs/Maria.Aristizabal/HWRF_POM_' + storm_id + '_' + str(ti.year) + '/' + 'HWRF_POM_' + storm_id + '_' + cycle + '/'

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

oktime_POM = np.where(time_POM == time_POM[1])[0][0] #first file

#%%
'''
oklon_POM = np.where(np.logical_and(lon_pom[0,:] >= lon_lim[0],lon_pom[0,:] <= lon_lim[1]))[0]
oklat_POM = np.where(np.logical_and(lat_pom[:,0] >= lat_lim[0],lat_pom[:,0] <= lat_lim[1]))[0]
#oktime_POM = np.where(time_POM == time_POM[0])[0][0]

lon_POM =lon_pom[0,oklon_POM]
lat_POM =lat_pom[oklat_POM,0]
'''

#%% Getting POM variables
#pom = xr.open_dataset(ncfiles[0]) #firts file is cycle
'''
pom = xr.open_dataset(ncfiles[oktime_POM]) #second file in cycle
sst_POM = np.asarray(pom['t'][0,0,oklat_POM,oklon_POM])
sst_POM[sst_POM==0] = np.nan
sss_POM = np.asarray(pom['s'][0,0,oklat_POM,oklon_POM])
sss_POM[sss_POM==0] = np.nan
ssh_POM = np.asarray(pom['elb'][0,oklat_POM,oklon_POM])
ssh_POM[ssh_POM==0] = np.nan
su_POM = np.asarray(pom['u'][0,0,oklat_POM,oklon_POM])
su_POM[su_POM==0] = np.nan
sv_POM = np.asarray(pom['v'][0,0,oklat_POM,oklon_POM])
sv_POM[sv_POM==0] = np.nan
'''

#%% Figure temp transect along Endurance line

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

oklon = np.round(np.interp(X,lon_pom[0,:],np.arange(len(lon_pom[0,:])))).astype(int)
oklat = np.round(np.interp(Y,lat_pom[:,0],np.arange(len(lat_pom[:,0])))).astype(int)
topoz_pom = np.asarray(topoz[oklat,oklon])
zmatrix_POM = np.dot(topoz_pom.reshape(-1,1),zlevc.reshape(1,-1)).T
dist_matrix = np.tile(dist,(zmatrix_POM.shape[0],1))

trans_temp_POM = np.empty((zmatrix_POM.shape[0],zmatrix_POM.shape[1]))
trans_temp_POM[:] = np.nan
pom = xr.open_dataset(ncfiles[oktime_POM])
for x in np.arange(len(X)):
    print(x)
    trans_temp_POM[:,x] = np.asarray(pom['t'][0,:,oklat[x],oklon[x]])

min_valt = 8
max_valt = 27
nlevelst = max_valt - min_valt + 1
kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))

fig, ax = plt.subplots(figsize=(9, 3))
plt.contourf(dist_matrix,zmatrix_POM,trans_temp_POM,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(dist_matrix,zmatrix_POM,trans_temp_POM,[26],color='k')
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.title('HWRF-POM Endurance Line MAB ' + 'Storm ' + storm_id + ' Cycle ' + cycle ,fontsize=16)
plt.ylim([-100,0])
plt.xlim([0,200])
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xlabel('Along Transect Distance (km)',fontsize=14)

file = folder_fig + 'HWRF_POM_temp_MAB_endurance_line_cycle_'+cycle
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Bathymetry GEBCO HYCOM domain
kw = dict(levels =  np.arange(-5000,1,200))

plt.figure()
plt.contour(bath_lon,bath_lat,bath_elev,levels=[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,cmap=cmocean.cm.topo,**kw)
plt.plot(X,Y,'-k')
plt.colorbar()
plt.axis('scaled')
plt.title('GEBCO Bathymetry')
plt.xlim(-76,-70)
plt.ylim(35,42)

#%% Figure temp transect across GoMex

x1 = -90
y1 = 20 + 52/60
x2 = -90
y2 = 30

Y = np.arange(y1,y2,0.05)
X = np.tile(x1,len(Y))

dist = np.sqrt((X-x1)**2 + (Y-y1)**2)*111 # approx along transect distance in km

oklon = np.round(np.interp(X,lon_pom[0,:],np.arange(len(lon_pom[0,:])))).astype(int)
oklat = np.round(np.interp(Y,lat_pom[:,0],np.arange(len(lat_pom[:,0])))).astype(int)
topoz_pom = np.asarray(topoz[oklat,oklon])
zmatrix_POM = np.dot(topoz_pom.reshape(-1,1),zlevc.reshape(1,-1)).T
dist_matrix = np.tile(dist,(zmatrix_POM.shape[0],1))

trans_temp_POM = np.empty((zmatrix_POM.shape[0],zmatrix_POM.shape[1]))
trans_temp_POM[:] = np.nan
pom = xr.open_dataset(ncfiles[oktime_POM])
for x in np.arange(len(X)):
    print(x)
    trans_temp_POM[:,x] = np.asarray(pom['t'][0,:,oklat[x],oklon[x]])

min_valt = 12
max_valt = 32
nlevelst = max_valt - min_valt + 1
kw = dict(levels = np.linspace(min_valt,max_valt,nlevelst))

fig, ax = plt.subplots(figsize=(9, 3))
plt.contourf(dist_matrix,zmatrix_POM,trans_temp_POM,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(dist_matrix,zmatrix_POM,trans_temp_POM,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.title('HWRF-POM  Across GoMex ' + 'Storm ' + storm_id + ' Cycle ' + cycle,fontsize=16)
plt.ylim([-300,0])
#plt.xlim([0,200])
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xlabel('Along Transect Distance (km)',fontsize=14)

file = folder_fig + 'HWRF_POM_temp_GoMex_across_cycle_'+cycle
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Bathymetry GEBCO HYCOM domain
kw = dict(levels =  np.arange(-5000,1,200))

plt.figure()
plt.contour(bath_lon,bath_lat,bath_elev,levels=[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,cmap=cmocean.cm.topo,**kw)
plt.plot(X,Y,'-k')
plt.colorbar()
plt.axis('scaled')
plt.title('GEBCO Bathymetry')
plt.xlim(-98,-80)
plt.ylim(18,32)