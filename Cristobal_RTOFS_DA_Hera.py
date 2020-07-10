#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:36:33 2020

@author: aristizabal
"""

#%% User input
cycle = '2020060300'

# GoMex
lon_lim = [-98,-79]
lat_lim = [15,32.5]

# folder ab files HYCOM
folder_RTOFS_DA = '/scratch2/NOS/nosofs/Maria.Aristizabal/RTOFS-DA/data_' + cycle
prefix_RTOFS_DA = 'archv'

# RTOFS grid file name
folder_RTOFS_DA_grid_depth = '/scratch2/NOS/nosofs/Maria.Aristizabal/RTOFS-DA/GRID_DEPTH/'
RTOFS_DA_grid = folder_RTOFS_DA_grid_depth + 'regional.grid'
RTOFS_DA_depth = folder_RTOFS_DA_grid_depth + 'regional.depth'

folder_fig = '/home/aristizabal/Cristobal_2020/Figures/'

#%% 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import os
import os.path
import glob
import cmocean
from mpl_toolkits.basemap import Basemap

import sys
sys.path.append('/home/Maria.Aristizabal/NCEP_scripts/')
from utils4HYCOM import readBinz, readgrids
from utils4HYCOM import readdepth, readVar

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading RTOFS grid
print('Retrieving coordinates from RTOFS')
# Reading lat and lon
lines_grid = [line.rstrip() for line in open(RTOFS_DA_grid+'.b')]
lon_RTOFS_DA = np.array(readgrids(RTOFS_DA_grid,'plon:',[0]))
lat_RTOFS_DA = np.array(readgrids(RTOFS_DA_grid,'plat:',[0]))

depth_RTOFS_DA = np.asarray(readdepth(RTOFS_DA_depth,'depth'))

#%%
N = 0
var = 'temp'

afiles = sorted(glob.glob(os.path.join(folder_RTOFS_DA,prefix_RTOFS_DA+'*.a')))
file = afiles[N]
lines = [line.rstrip() for line in open(file[:-2]+'.b')]
time_stamp = lines[-1].split()[2]
hycom_days = lines[-1].split()[3]
tzero=datetime(1901,1,1,0,0)
time_RTOFS_DA = tzero+timedelta(float(hycom_days)-1)

# Reading 3D variable from binary file 
var_rtofs = readBinz(file[:-2],'3z',var)

#%%

oklon = np.where(np.logical_and(lon_RTOFS_DA[0,:] >= lon_lim[0]+360, lon_RTOFS_DA[0,:] <= lon_lim[1]+360))[0]
oklat = np.where(np.logical_and(lat_RTOFS_DA[:,0] >= lat_lim[0], lat_RTOFS_DA[:,0] <= lat_lim[1]))[0]

temp_RTOFS_DA = np.asarray(var_rtofs[oklat,:,0][:,oklon])

m = Basemap(projection='merc',llcrnrlat=15,urcrnrlat=32.5,llcrnrlon=-98,urcrnrlon=-80,resolution='l')
x, y = m(*np.meshgrid(lon_RTOFS_DA[0,oklon]-360,lat_RTOFS_DA[oklat,0]))

plt.figure()
plt.ion()
m.drawcoastlines()
m.fillcontinents(color='seashell')
m.drawmapboundary()

kw = dict(levels = np.linspace(24,30,16))
plt.contourf(x,y,temp_RTOFS_DA,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
plt.title('RTOFS-DA SST on '+str(time_RTOFS_DA)[0:13],fontsize=16)
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)

yticks, xticks = m(np.arange(-97.5,-79,2.5),np.arange(15,33,2.5))
plt.yticks(yticks,labels=np.arange(15,33,2.5),fontsize=12)
plt.xticks(xticks,np.arange(-97.5,-79,2.5),fontsize=12)
xq,yq = m(np.tile(-90,len(lat_RTOFS_DA[:,0])),lat_RTOFS_DA[:,0])
plt.plot(xq,yq,'-',color='k')

#q = plt.quiver(x[::7,::7], y[::7,::7],u_POM[::7,::7],v_POM[::7,::7])
#xq,yq = m(-74,33.5)
#plt.quiverkey(q,xq,yq,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})

file_name = folder_fig + 'Cristobal_SST_' + str(time_RTOFS_DA[0])[0:10]
plt.savefig(file_name,bbox_inches = 'tight',pad_inches = 0.1)

#%%
oklon = np.where(lon_RTOFS_DA[0,:] >= -90+360)[0][0]
oklat = np.where(np.logical_and(lat_RTOFS_DA[:,0] >= lat_lim[0], lat_RTOFS_DA[:,0] <= lat_lim[1]))[0]

N = 0
nz = 41
layers = np.arange(0,nz)
target_temp_RTOFS_DA = np.empty((oklat.shape[0],nz))
target_temp_RTOFS_DA[:] = np.nan
target_zRTOFS_DA = np.empty((oklat.shape[0] ,nz))
target_zRTOFS_DA[:] = np.nan
time_RTOFS_DA = []
oklonRTOFS_DA = []
oklatRTOFS_DA = []
    
#target_lon = long + 360
#target_lat = latg
target_ztmp = np.tile(0,len(oklat))
for lyr in tuple(layers):
    print(lyr)
    temp_RTOFS = readVar(file[:-2],'archive','temp',[lyr+1])
    target_temp_RTOFS_DA[:,lyr] = temp_RTOFS[oklat,oklon].data
    
    dp = readVar(file[:-2],'archive','thknss',[lyr+1])/9806
    dpsub = dp[oklat,oklon].data
    dpsub[dpsub>10**10] = np.nan
    target_ztmp = np.vstack([target_ztmp,dpsub])

target_temp_RTOFS_DA[target_temp_RTOFS_DA>100] = np.nan    
target_zRTOFS_DA = (np.cumsum(target_ztmp[0:-1,:],axis=0) + np.diff(np.cumsum(target_ztmp,axis=0),axis=0)/2).T 
lat_matrix = np.tile(lat_RTOFS_DA[oklat,0],[nz,1]).T

#temp_trans_RTOFS_DA = np.asarray(var_rtofs[:,oklon,:][oklat,:])

kw = dict(levels = np.linspace(12,32,21))
plt.figure()
plt.contourf(lat_matrix,-target_zRTOFS_DA,target_temp_RTOFS_DA,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
plt.contour(lat_matrix,-target_zRTOFS_DA,target_temp_RTOFS_DA,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.title('Temperature along Cristobal Path',fontsize=16)
plt.ylim([-200,0])
plt.xlim([20,30])
plt.ylabel('Depth (m)',fontsize=14)

file = folder_fig + 'RTOFS_DA_temp_along_Cristobal_' + str(time_RTOFS_DA[0])[0:10]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)