#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:14:51 2020

@author: aristizabal
"""

#%% User input

# date limits
date_ini = '2020/09/22/00'
date_end = '2020/09/22/00'

# MAB
#lon_lim = [-110.0,-10.0]
#lat_lim = [15.0,45.0]
lon_lim = [-98.0,-50.0]
lat_lim = [15.0,45.0]

# Server location
url_GOFS_HYCOM = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'
url_GOFS_oceanmap = 'http://data.oceansmap.com/eds_thredds/dodsC/EDS/HYCOM_GLOBAL_NAVY/hycomglobalnavy_2020092200.nc'

folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/'

#%%
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import xarray as xr
import netCDF4

#%% Function Conversion from glider longitude and latitude to GOFS convention

def glider_coor_to_GOFS_coord(long,latg):
    
    target_lon = np.empty((len(long),))
    target_lon[:] = np.nan
    for i,ii in enumerate(long):
        if ii < 0: 
            target_lon[i] = 360 + ii
        else:
            target_lon[i] = ii
    target_lat = latg
    
    return target_lon, target_lat

#%%  Function Conversion from GOFS convention to glider longitude and latitude
    
def GOFS_coor_to_glider_coord(lon_GOFS,lat_GOFS):
    
    lon_GOFSg = np.empty((len(lon_GOFS),))
    lon_GOFSg[:] = np.nan
    for i in range(len(lon_GOFS)):
        if lon_GOFS[i] > 180: 
            lon_GOFSg[i] = lon_GOFS[i] - 360 
        else:
            lon_GOFSg[i] = lon_GOFS[i]
    lat_GOFSg = lat_GOFS
    
    return lon_GOFSg, lat_GOFSg

#%% GOFS 3.1 OceanMap

GOFS_oceanmap = xr.open_dataset(url_GOFS_oceanmap,decode_times=False)

lat_GOFS_oceanmap = np.asarray(GOFS_oceanmap['lat'][:])
lon_GOFS_oceanmap = np.asarray(GOFS_oceanmap['lon'][:])
tt = GOFS_oceanmap['time']
t_GOFS_oceanmap = netCDF4.num2date(tt[:],tt.units) 
depth_GOFS_oceanmap = np.asarray(GOFS_oceanmap['depth'][:])

lon_limG, lat_limG = glider_coor_to_GOFS_coord(lon_lim,lat_lim)

oklon_GO = np.where(np.logical_and(lon_GOFS_oceanmap>=lon_limG[0],lon_GOFS_oceanmap<=lon_limG[1]))[0]
oklat_GO = np.where(np.logical_and(lat_GOFS_oceanmap>=lat_limG[0],lat_GOFS_oceanmap<=lat_limG[1]))[0]

u_surf_GOFS_oceanmap = np.asarray(GOFS_oceanmap['water_u'][0,0,oklat_GO,oklon_GO])
v_surf_GOFS_oceanmap = np.asarray(GOFS_oceanmap['water_v'][0,0,oklat_GO,oklon_GO])

#%% GOFS 3.1 HYCOM

GOFS_HYCOM = xr.open_dataset(url_GOFS_HYCOM,decode_times=False)

lat_GOFS_HYCOM = np.asarray(GOFS_HYCOM['lat'][:])
lon_GOFS_HYCOM = np.asarray(GOFS_HYCOM['lon'][:])
tt = GOFS_HYCOM['time']
t_GOFS_HYCOM = netCDF4.num2date(tt[:],tt.units) 
depth_GOFS_HYCOM = np.asarray(GOFS_HYCOM['depth'][:])

lon_limG, lat_limG = glider_coor_to_GOFS_coord(lon_lim,lat_lim)

oklon_GH = np.where(np.logical_and(lon_GOFS_HYCOM>=lon_limG[0],lon_GOFS_HYCOM<=lon_limG[1]))[0]
oklat_GH = np.where(np.logical_and(lat_GOFS_HYCOM>=lat_limG[0],lat_GOFS_HYCOM<=lat_limG[1]))[0]
okt = np.where(t_GOFS_HYCOM == t_GOFS_oceanmap)[0][0]

u_surf_GOFS_HYCOM = np.asarray(GOFS_HYCOM['water_u'][okt,0,oklat_GH,oklon_GH])
v_surf_GOFS_HYCOM = np.asarray(GOFS_HYCOM['water_v'][okt,0,oklat_GH,oklon_GH])

#%%
kw = dict(levels = np.arange(-2.5,2.6,0.5))

plt.figure(figsize=(10,5))
plt.contour(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],u_surf_GOFS_oceanmap,\
             cmap = cmocean.cm.balance,**kw)
plt.contourf(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],u_surf_GOFS_oceanmap,\
              cmap = cmocean.cm.balance,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
plt.axis('scaled')
plt.title('data.oceansmap.com/eds_thredds/dodsC/EDS/HYCOM_GLOBAL_NAVY/hycomglobalnavy_2020092200.nc \n'+
          'U surface velocity  on ' + str(t_GOFS_oceanmap[0]))
file = folder_fig + 'oceanmap_u_surf_'+str(t_GOFS_oceanmap[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
kw = dict(levels = np.arange(-2.5,2.6,0.5))

plt.figure(figsize=(10,5))
plt.contour(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],u_surf_GOFS_HYCOM,\
             cmap = cmocean.cm.balance,**kw)
plt.contourf(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],u_surf_GOFS_HYCOM,\
             cmap = cmocean.cm.balance,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
plt.axis('scaled')
plt.title('tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z \n'+
          'U surface velocity  on ' + str(t_GOFS_HYCOM[okt]))
file = folder_fig + 'HYCOM_u_surf_'+str(t_GOFS_HYCOM[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
kw = dict(levels = np.arange(-2.5,2.6,0.5))

plt.figure(figsize=(10,5))
plt.contour(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],v_surf_GOFS_oceanmap,\
             cmap = cmocean.cm.balance,**kw)
plt.contourf(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],v_surf_GOFS_oceanmap,\
              cmap = cmocean.cm.balance,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
plt.axis('scaled')
plt.title('data.oceansmap.com/eds_thredds/dodsC/EDS/HYCOM_GLOBAL_NAVY/hycomglobalnavy_2020092200.nc \n'+
          'V surface velocity  on ' + str(t_GOFS_oceanmap[0]))
file = folder_fig + 'oceanmap_v_surf_'+str(t_GOFS_oceanmap[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
kw = dict(levels = np.arange(-2.5,2.6,0.5))

plt.figure(figsize=(10,5))
plt.contour(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],v_surf_GOFS_HYCOM,\
             cmap = cmocean.cm.balance,**kw)
plt.contourf(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],v_surf_GOFS_HYCOM,\
             cmap = cmocean.cm.balance,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
plt.axis('scaled')
plt.title('tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z \n'+
          'V surface velocity  on ' + str(t_GOFS_HYCOM[okt]))
file = folder_fig + 'HYCOM_v_surf_'+str(t_GOFS_HYCOM[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
kw = dict(levels = np.arange(0,2.6,0.5))

Vel_surf_GOFS_oceanmap = np.sqrt(v_surf_GOFS_oceanmap**2 + u_surf_GOFS_oceanmap**2)

plt.figure(figsize=(10,5))
plt.contour(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],Vel_surf_GOFS_oceanmap,\
             cmap = cmocean.cm.speed,**kw)
plt.contourf(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],Vel_surf_GOFS_oceanmap,\
              cmap = cmocean.cm.speed,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
plt.axis('scaled')
plt.title('data.oceansmap.com/eds_thredds/dodsC/EDS/HYCOM_GLOBAL_NAVY/hycomglobalnavy_2020092200.nc \n'+
          'Vel magnitude surface velocity  on ' + str(t_GOFS_oceanmap[0]))
file = folder_fig + 'oceanmap_Vel_surf_'+str(t_GOFS_oceanmap[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
kw = dict(levels = np.arange(0,2.6,0.5))

Vel_surf_GOFS_HYCOM = np.sqrt(v_surf_GOFS_HYCOM**2 + u_surf_GOFS_HYCOM**2)

plt.figure(figsize=(10,5))
plt.contour(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],Vel_surf_GOFS_HYCOM,\
             cmap = cmocean.cm.speed,**kw)
plt.contourf(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],Vel_surf_GOFS_HYCOM,\
             cmap = cmocean.cm.speed,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
plt.axis('scaled')
plt.title('tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z \n'+
          'Vel magnitude surface velocity  on ' + str(t_GOFS_HYCOM[okt]))
file = folder_fig + 'HYCOM_Vel_surf_'+str(t_GOFS_HYCOM[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
kw = dict(levels = np.arange(0,2.6,0.5))

Vel_surf_GOFS_oceanmap = np.sqrt(v_surf_GOFS_oceanmap**2 + u_surf_GOFS_oceanmap**2)

plt.figure(figsize=(10,5))
#plt.contour(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],Vel_surf_GOFS_oceanmap,\
#             cmap = cmocean.cm.speed,**kw)
plt.contourf(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],Vel_surf_GOFS_oceanmap,\
              cmap = cmocean.cm.speed,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
q=plt.quiver(lon_GOFS_oceanmap[oklon_GO][::7]-360,lat_GOFS_oceanmap[oklat_GO][::7],\
             u_surf_GOFS_oceanmap[::7,::7],v_surf_GOFS_oceanmap[::7,::7] ,scale=3,scale_units='inches',\
          alpha=0.7)
plt.quiverkey(q,-95,40,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})

plt.axis('scaled')
plt.title('data.oceansmap.com/eds_thredds/dodsC/EDS/HYCOM_GLOBAL_NAVY/hycomglobalnavy_2020092200.nc \n'+
          'Surface velocity  on ' + str(t_GOFS_oceanmap[0]))
file = folder_fig + 'oceanmap_Vel_surf_2_'+str(t_GOFS_oceanmap[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
kw = dict(levels = np.arange(0,2.6,0.5))

Vel_surf_GOFS_HYCOM = np.sqrt(v_surf_GOFS_HYCOM**2 + u_surf_GOFS_HYCOM**2)

plt.figure(figsize=(10,5))
#plt.contour(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],Vel_surf_GOFS_HYCOM,\
#             cmap = cmocean.cm.speed,**kw)
plt.contourf(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],Vel_surf_GOFS_HYCOM,\
             cmap = cmocean.cm.speed,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
q=plt.quiver(lon_GOFS_HYCOM[oklon_GH][::7]-360,lat_GOFS_HYCOM[oklat_GH][::7],\
             u_surf_GOFS_HYCOM[::7,::7],v_surf_GOFS_HYCOM[::7,::7] ,scale=3,scale_units='inches',\
          alpha=0.7)
plt.quiverkey(q,-95,40,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})
plt.axis('scaled')
plt.title('tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z \n'+
          'Vel magnitude surface velocity  on ' + str(t_GOFS_HYCOM[okt]))
file = folder_fig + 'HYCOM_Vel_surf_2_'+str(t_GOFS_HYCOM[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
kw = dict(levels = np.arange(0,2.6,0.5))

Vel_surf_GOFS_oceanmap = np.sqrt(v_surf_GOFS_oceanmap**2 + u_surf_GOFS_oceanmap**2)

plt.figure(figsize=(10,5))
#plt.contour(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],Vel_surf_GOFS_oceanmap,\
#             cmap = cmocean.cm.speed,**kw)
plt.contourf(lon_GOFS_oceanmap[oklon_GO]-360,lat_GOFS_oceanmap[oklat_GO],Vel_surf_GOFS_oceanmap,\
              cmap = cmocean.cm.speed,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
q=plt.quiver(lon_GOFS_oceanmap[oklon_GO][::7]-360,lat_GOFS_oceanmap[oklat_GO][::7],\
             u_surf_GOFS_oceanmap[::7,::7],v_surf_GOFS_oceanmap[::7,::7] ,scale=3,scale_units='inches',\
          alpha=0.7)
plt.quiverkey(q,-96.5,29,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})

plt.title('data.oceansmap.com/eds_thredds/dodsC/EDS/HYCOM_GLOBAL_NAVY/hycomglobalnavy_2020092200.nc \n'+
          'Surface velocity  on ' + str(t_GOFS_oceanmap[0]))
plt.axis('scaled')
plt.xlim(-98,-80)
plt.ylim(15,31)
file = folder_fig + 'oceanmap_Vel_surf_3_'+str(t_GOFS_oceanmap[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
kw = dict(levels = np.arange(0,2.6,0.5))

Vel_surf_GOFS_HYCOM = np.sqrt(v_surf_GOFS_HYCOM**2 + u_surf_GOFS_HYCOM**2)

plt.figure(figsize=(10,5))
#plt.contour(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],Vel_surf_GOFS_HYCOM,\
#             cmap = cmocean.cm.speed,**kw)
plt.contourf(lon_GOFS_HYCOM[oklon_GH]-360,lat_GOFS_HYCOM[oklat_GH],Vel_surf_GOFS_HYCOM,\
             cmap = cmocean.cm.speed,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('m/s',fontsize=14) 
q=plt.quiver(lon_GOFS_HYCOM[oklon_GH][::7]-360,lat_GOFS_HYCOM[oklat_GH][::7],\
             u_surf_GOFS_HYCOM[::7,::7],v_surf_GOFS_HYCOM[::7,::7] ,scale=3,scale_units='inches',\
          alpha=0.7)
plt.quiverkey(q,-96.5,29,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})
plt.axis('scaled')
plt.title('tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z \n'+
          'Vel magnitude surface velocity  on ' + str(t_GOFS_HYCOM[okt]))
plt.xlim(-98,-80)
plt.ylim(15,31)
file = folder_fig + 'HYCOM_Vel_surf_3_'+str(t_GOFS_HYCOM[0])[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%



