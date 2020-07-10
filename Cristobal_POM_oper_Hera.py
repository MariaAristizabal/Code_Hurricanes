#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:15:36 2020

@author: aristizabal
"""

#%% User input
cycle ='2020060406'

# GoMex
lon_lim = [-98,80]
lat_lim = [15,32.5]

folder_pom19 = '/scratch2/NOS/nosofs/Maria.Aristizabal/HWRF2019_POM_Cristobal/HWRF2019_POM_Cristobal_' + cycle + '_oper/'

# POM grid file name
grid_file = folder_pom19 + 'cristobal03l.' + cycle + '.pom.grid.nc'

# POM files
prefix = 'cristobal03l.' + cycle + '.pom.'

# Name of 3D variable
var_name = 't'

folder_fig = '/home/aristizabal/Cristobal_2020/Figures/'

#%% 
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.dates import date2num, num2date
import os
import os.path
import glob
import cmocean
from mpl_toolkits.basemap import Basemap

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading POM grid files
pom_grid = xr.open_dataset(grid_file)
lon_POM = np.asarray(pom_grid['east_e'][:])
lat_POM = np.asarray( pom_grid['north_e'][:])
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
pom = xr.open_dataset(ncfiles[0])
temp_POM = np.asarray(pom['t'][0,0,:,:])
temp_POM[temp_POM==0] = np.nan
u_POM = np.asarray(pom['u'][0,0,:,:])
u_POM[u_POM==0] = np.nan
v_POM = np.asarray(pom['v'][0,0,:,:])
v_POM[v_POM==0] = np.nan

m = Basemap(projection='merc',llcrnrlat=15,urcrnrlat=32.5,llcrnrlon=-98,urcrnrlon=-80,resolution='l')
x, y = m(*np.meshgrid(lon_POM[0,:],lat_POM[:,0]))

#plt.figure(figsize=(10, 8))
plt.figure()
plt.ion()
m.drawcoastlines()
m.fillcontinents(color='seashell')
m.drawmapboundary()
#m.drawparallels(np.arange(15,33,2.5),labels=[15. , 17.5, 20. , 22.5, 25. , 27.5, 30. ,32.5])
#m.drawmeridians()

kw = dict(levels=np.linspace(24,30,16))
plt.contourf(x,y,temp_POM,cmap=cmocean.cm.thermal,**kw)
xq,yq = m(np.tile(-90,len(lat_POM[:,0])),lat_POM[:,0])
plt.plot(xq,yq,'-',color='k')
cbar = plt.colorbar()
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
cbar.ax.tick_params(labelsize=14)
#plt.title('HWRF-POM SST and Surface Velocity on '+str(time_POM[0])[0:13],fontsize=14)
plt.title('HWRF-POM SST on '+str(time_POM[0])[0:13],fontsize=16)
#c.set_label('($^oC$)',rotation=90, labelpad=15, fontsize=16)

yticks, xticks = m(np.arange(-97.5,-79,2.5),np.arange(15,33,2.5))
plt.yticks(yticks,labels=np.arange(15,33,2.5),fontsize=12)
plt.xticks(xticks,np.arange(-97.5,-79,2.5),fontsize=12)

q = plt.quiver(x[::7,::7], y[::7,::7],u_POM[::7,::7],v_POM[::7,::7])
xq,yq = m(-74,33.5)
plt.quiverkey(q,xq,yq,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})

file_name = folder_fig + 'Cristobal_SST_' + str(time_POM[0])[0:10]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1)

#%%
POM = xr.open_dataset(ncfiles[0])
oklon = np.where(lon_POM[0,:]>=-90)[0][0]
temp_trans_POM = np.asarray(POM['t'][0,:,:,oklon]).T

z_matrix_pom = np.dot(topoz[:,oklon].reshape(-1,1),zlevc.reshape(1,-1))
lat_matrix = np.tile(lat_POM[:,0],(zlevc.shape[0],1)).T

kw = dict(levels = np.linspace(12,32,21))
plt.figure()
plt.contourf(lat_matrix,z_matrix_pom,temp_trans_POM,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
plt.contour(lat_matrix,z_matrix_pom,temp_trans_POM,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.title('Temperature along Cristobal Path',fontsize=16)
plt.ylim([-200,0])
plt.xlim([20,30])
plt.ylabel('Depth (m)',fontsize=14)

file = folder_fig + 'GOFS_temp_along_Cristobal_' + str(time_POM[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

