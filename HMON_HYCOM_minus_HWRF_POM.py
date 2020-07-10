#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:27:05 2020

@author: aristizabal
"""

#%% User input
cycle = '2020060406'

# GoMex
lon_lim = [-98,-79]
lat_lim = [15,32.5]

# folder ab files HYCOM
folder_HMON_HYCOM = '/scratch2/NOS/nosofs/Maria.Aristizabal/HMON_HYCOM_Cristobal/HMON_HYCOM_Cristobal_' + cycle +'_oper/'
prefix = 'hmon_rtofs_hat10_3z'

# RTOFS grid file name
HMON_HYCOM_grid = folder_HMON_HYCOM + 'hwrf_rtofs_hat10.basin.regional.grid'
HMON_HYCOM_depth = folder_HMON_HYCOM + 'hwrf_rtofs_hat10.basin.regional.depth'

folder_pom19 = '/scratch2/NOS/nosofs/Maria.Aristizabal/HWRF2019_POM_Cristobal/HWRF2019_POM_Cristobal_' + cycle + '_oper/'

# POM grid file name
grid_file = folder_pom19 + 'cristobal03l.' + cycle + '.pom.grid.nc'

# POM files
prefix = 'cristobal03l.' + cycle + '.pom.'

folder_fig = '/home/Maria.Aristizabal/Cristobal_2020/Figures/'

#%% 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import os
import os.path
import glob
import cmocean
from mpl_toolkits.basemap import Basemap
from matplotlib.dates import date2num, num2date
import scipy.interpolate as interp

import sys
sys.path.append('/home/Maria.Aristizabal/NCEP_scripts/')
from utils4HYCOM import readgrids
#from utils4HYCOM import readdepth, readVar
from utils4HYCOM2 import readBinz

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading RTOFS grid
print('Retrieving coordinates from RTOFS')
# Reading lat and lon
lines_grid = [line.rstrip() for line in open(HMON_HYCOM_grid+'.b')]
lon_HMON_hycom = np.array(readgrids(HMON_HYCOM_grid,'plon:',[0]))
lat_HMON_hycom = np.array(readgrids(HMON_HYCOM_grid,'plat:',[0]))

#depth_HMON_HYCOM = np.asarray(readdepth(HMON_HYCOM_depth,'depth'))

# Reading depths
afiles = sorted(glob.glob(os.path.join(folder_HMON_HYCOM,'*'+prefix+'*.a')))
lines=[line.rstrip() for line in open(afiles[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]=='temp':
        #print(line.split()[1])
        z.append(float(line.split()[1]))
z_HMON_HYCOM = np.asarray(z)

#%%
afiles = sorted(glob.glob(os.path.join(folder_HMON_HYCOM,'*'+prefix+'*.a')))
time_HMON_HYCOM = []
for x, file in enumerate(afiles):
    print(x)
    #lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    year = int(file.split('/')[-1].split('.')[1][0:4])
    month = int(file.split('/')[-1].split('.')[1][4:6])
    day = int(file.split('/')[-1].split('.')[1][6:8])
    hour = int(file.split('/')[-1].split('.')[1][8:10])
    dt = int(file.split('/')[-1].split('.')[-2][1:])
    timestamp_HMON_HYCOM = date2num(datetime(year,month,day,hour)) + dt/24
    time_HMON_HYCOM.append(num2date(timestamp_HMON_HYCOM))

# Reading 3D variable from binary file
N = 0
var_HMON_HYCOM = readBinz(afiles[N][:-2],'3z','temp')

#%% Reading POM grid files
pom_grid = xr.open_dataset(grid_file)
lon_pom = np.asarray(pom_grid['east_e'][:])
lat_pom = np.asarray( pom_grid['north_e'][:])
zlevc = np.asarray(pom_grid['zz'][:])
topoz = np.asarray(pom_grid['h'][:])

#%% Getting list of POM files
ncfiles = sorted(glob.glob(os.path.join(folder_pom19,prefix+'*0*.nc')))

# Reading POM time
time_pom = []
for i,file in enumerate(ncfiles):
    print(i)
    pom = xr.open_dataset(file)
    tpom = pom['time'][:]
    timestamp_pom = date2num(tpom)[0]
    time_pom.append(num2date(timestamp_pom))

time_POM = np.asarray(time_pom)
timestamp_POM = date2num(time_POM)

#%%

oklon_hh = np.where(np.logical_and(lon_HMON_hycom[0,:] >= lon_lim[0]+360, lon_HMON_hycom[0,:] <= lon_lim[1]+360))[0]
oklat_hh = np.where(np.logical_and(lat_HMON_hycom[:,0] >= lat_lim[0], lat_HMON_hycom[:,0] <= lat_lim[1]))[0]

temp_HMON_HYCOM = np.asarray(var_HMON_HYCOM[oklat_hh,:,0][:,oklon_hh])
temp_HMON_HYCOM[temp_HMON_HYCOM > 100] = np.nan

lon_HMON_HYCOM = lon_HMON_hycom[0,oklon_hh]
lat_HMON_HYCOM = lat_HMON_hycom[oklat_hh,0]

m = Basemap(projection='merc',llcrnrlat=15,urcrnrlat=32.5,llcrnrlon=-98,urcrnrlon=-80,resolution='l')
x, y = m(*np.meshgrid(lon_HMON_HYCOM-360,lat_HMON_HYCOM))

plt.figure()
plt.ion()
m.drawcoastlines()
m.fillcontinents(color='seashell')
m.drawmapboundary()

kw = dict(levels = np.linspace(24,30,16))
plt.contourf(x,y,temp_HMON_HYCOM,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
plt.title('HMON_HYCOM SST on '+str(time_HMON_HYCOM[N])[0:13],fontsize=16)
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)

yticks, xticks = m(np.arange(-97.5,-79,2.5),np.arange(15,33,2.5))
plt.yticks(yticks,labels=np.arange(15,33,2.5),fontsize=12)
plt.xticks(xticks,np.arange(-97.5,-79,2.5),fontsize=12)
xq,yq = m(np.tile(-90,len(lat_HMON_HYCOM[:,0])),lat_HMON_HYCOM[:,0])
plt.plot(xq,yq,'-',color='k')

#q = plt.quiver(x[::7,::7], y[::7,::7],u_POM[::7,::7],v_POM[::7,::7])
#xq,yq = m(-74,33.5)
#plt.quiverkey(q,xq,yq,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})

file_name = folder_fig + 'HMON_HYCOM_Cristobal_SST_' + str(time_HMON_HYCOM[0])[0:10]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1)

#%%

oklon_hp = np.where(np.logical_and(lon_pom[0,:] >= lon_lim[0], lon_pom[0,:] <= lon_lim[1]))[0]
oklat_hp = np.where(np.logical_and(lat_pom[:,0] >= lat_lim[0], lat_pom[:,0] <= lat_lim[1]))[0]

Npom = 1
POM = xr.open_dataset(ncfiles[Npom])
temp_POM = np.asarray(pom['t'][0,0,oklat_hp,oklon_hp])
temp_POM[temp_POM==0] = np.nan
u_POM = np.asarray(pom['u'][0,0,oklat_hp,oklon_hp])
u_POM[u_POM==0] = np.nan
v_POM = np.asarray(pom['v'][0,0,oklat_hp,oklon_hp])
v_POM[v_POM==0] = np.nan

lon_POM = lon_pom[0,oklon_hp]
lat_POM = lat_pom[oklat_hp,0]

m = Basemap(projection='merc',llcrnrlat=15,urcrnrlat=32.5,llcrnrlon=-98,urcrnrlon=-80,resolution='l')
x, y = m(*np.meshgrid(lon_POM,lat_POM))

plt.figure()
plt.ion()
m.drawcoastlines()
m.fillcontinents(color='seashell')
m.drawmapboundary()

kw = dict(levels=np.linspace(24,30,16))
plt.contourf(x,y,temp_POM,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
cbar.ax.tick_params(labelsize=14)
plt.title('HWRF-POM SST on '+str(time_POM[Npom])[0:13],fontsize=16)

yticks, xticks = m(np.arange(-97.5,-79,2.5),np.arange(15,33,2.5))
plt.yticks(yticks,labels=np.arange(15,33,2.5))
plt.xticks(xticks,np.arange(-97.5,-79,2.5))
xq,yq = m(np.tile(-90,len(lat_POM)),lat_POM)
plt.plot(xq,yq,'-',color='k')

#q = plt.quiver(x[::7,::7], y[::7,::7],u_POM[::7,::7],v_POM[::7,::7])
#xq,yq = m(-74,33.5)
#plt.quiverkey(q,xq,yq,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})

file_name = folder_fig + 'Cristobal_SST_' + str(time_POM[0])[0:10]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1)

#%% 2D interpolation 

x = lon_HMON_HYCOM-360
y = lat_HMON_HYCOM
z = temp_HMON_HYCOM
z[np.isnan(z)] = 0
#f = interp.RectBivariateSpline(x, y, z,kx=1,ky=1)
f = interp.interp2d(x, y, z)

xnew = lon_POM
ynew = lat_POM
znew = f(xnew,ynew)
znew[znew==0] = np.nan

#%% HMON_HYCOM - HWRF_POM

m = Basemap(projection='merc',llcrnrlat=15,urcrnrlat=32.5,llcrnrlon=-98,urcrnrlon=-80,resolution='l')
x, y = m(*np.meshgrid(xnew,ynew))

#plt.figure(figsize=(10, 8))
plt.figure()
plt.ion()
m.drawcoastlines()
m.fillcontinents(color='seashell')
m.drawmapboundary()

kw = dict(levels=np.linspace(-3,3,13))
plt.contourf(x,y,temp_POM-znew,cmap=cmocean.cm.curl,**kw)
cbar = plt.colorbar()
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
cbar.ax.tick_params(labelsize=14)
plt.title('HWRF-POM minus HMON-HYCOM SST on '+str(time_HMON_HYCOM[0])[0:13],fontsize=14)
yticks, xticks = m(np.arange(-97.5,-79,2.5),np.arange(15,33,2.5))
plt.yticks(yticks,labels=np.arange(15,33,2.5))
plt.xticks(xticks,np.arange(-97.5,-79,2.5))

#%% HWRF-POM tem transect
POM = xr.open_dataset(ncfiles[Npom])
oklon = np.where(lon_pom[0,:]>=-90)[0][0]
oklat_hp = np.where(np.logical_and(lat_pom[:,0] >= lat_lim[0], lat_pom[:,0] <= lat_lim[1]))[0]
temp_trans_POM = np.asarray(POM['t'][0,:,oklat_hp,oklon]).T

z_matrix_pom = np.dot(topoz[oklat_hp,oklon].reshape(-1,1),zlevc.reshape(1,-1))
lat_matrix = np.tile(lat_pom[oklat_hp,0],(zlevc.shape[0],1)).T

kw = dict(levels = np.linspace(12,32,21))
plt.figure()
plt.contourf(lat_matrix,z_matrix_pom,temp_trans_POM,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
plt.contour(lat_matrix,z_matrix_pom,temp_trans_POM,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.title('Temperature along Cristobal Path on '+str(time_POM[Npom])[0:13],fontsize=14)
plt.ylim([-200,0])
plt.xlim([20,30])
plt.ylabel('Depth (m)',fontsize=14)

file = folder_fig + 'GOFS_temp_along_Cristobal_' + str(time_POM[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% HMON-HYCOM temp transect
oklon2 = np.where(lon_HMON_hycom[0,:] >= -90+360)[0][0]
oklat = np.where(np.logical_and(lat_HMON_hycom[:,0] >= lat_lim[0], lat_HMON_hycom[:,0] <= lat_lim[1]))[0]

trans_temp_HMON_HYCOM = np.asarray(var_HMON_HYCOM[oklat,oklon2,:].T)

kw = dict(levels = np.linspace(12,32,21))
plt.figure()
plt.contourf(lat_HMON_hycom[oklat,0],-z_HMON_HYCOM,trans_temp_HMON_HYCOM,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
plt.contour(lat_HMON_hycom[oklat,0],-z_HMON_HYCOM,trans_temp_HMON_HYCOM,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.title('Temperature along Cristobal Path on '+str(time_HMON_HYCOM[0])[0:13],fontsize=14)
plt.ylim([-200,0])
plt.xlim([20,30])
plt.ylabel('Depth (m)',fontsize=14)

file = folder_fig + 'HMON_HYCOM_temp_along_Cristobal_' + str(time_HMON_HYCOM[N])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% 2D interpolation 
'''
x = lat_HMON_hycom[oklat,0]
y = -z_HMON_HYCOM
z = trans_temp_HMON_HYCOM
#z[np.isnan(z)] = 0
#f = interp.RectBivariateSpline(x, y, z,kx=1,ky=1)
f = interp.interp2d(x, y, z)

xnew = lat_matrix
ynew = z_matrix_pom
znew = np.empty((xnew.shape[0],xnew.shape[1]))
znew[:] = np.nan
for y,lat in enumerate(lat_matrix):
    print(y)
    znew[y,:] = f(xnew[:,0],ynew[y,:])[:,y]
#znew[znew==0] = np.nan
''' 

x = lat_matrix
y = z_matrix_pom
z = temp_trans_POM 
f = interp.interp2d(x, y, z)  

x = lat_HMON_hycom[oklat,0]
y = -z_HMON_HYCOM
z = trans_temp_HMON_HYCOM
#z[np.isnan(z)] = 0
#f = interp.RectBivariateSpline(x, y, z,kx=1,ky=1)
f = interp.interp2d(x, y, z)

xnew = lat_matrix
ynew = z_matrix_pom
znew = np.empty((xnew.shape[0],xnew.shape[1]))
znew[:] = np.nan
for y,lat in enumerate(lat_matrix):
    print(y)
    znew[y,:] = f(xnew[:,0],ynew[y,:])[:,y]
#znew[znew==0] = np.nan    
    
#%%
kw = dict(levels = np.linspace(5,35,21))
plt.figure()
plt.contourf(xnew,ynew,znew,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
#plt.contour(xnew,ynew,znew,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.title('Temperature along Cristobal Path on '+str(time_POM[Npom])[0:13],fontsize=14)
plt.ylim([-200,0])
plt.xlim([20,30])
plt.ylabel('Depth (m)',fontsize=14)

    

