#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:22:03 2019

@author: aristizabal
"""

#%% User input

# Directories where RTOFS files reside 
Dir= '/Volumes/aristizabal/ncep_model//HMON-HYCOM_Michael/'
Dir_graph = '/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts/Figures/'

# RTOFS grid file name
gridfile = 'hwrf_rtofs_hat10.basin.regional.grid'

# RTOFS a/b file name
prefix_ab = 'michael14l.2018100718.hmon_rtofs_hat10_3z'

# Name of 3D variable
var_name = 'temp'

# Glider data 

# ng288
gdata = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc'

# date limits
date_ini = '2018-10-07T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'

#%% Modules to read HYCOM output 

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')

from utils4HYCOM import readBinz, readgrids

import os
import os.path
import glob
from datetime import datetime
from matplotlib.dates import date2num, num2date
import matplotlib.pyplot as plt
import numpy as np

import netCDF4
import matplotlib.dates as mdates

import xarray as xr

#%% Functio to calculate density

def dens0(s, t):
    s, t = list(map(np.asanyarray, (s, t)))
    T68 = T68conv(t)
    # UNESCO 1983 Eqn.(13) p17.
    b = (8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9)
    c = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
    d = 4.8314e-4
    return (smow(t) + (b[0] + (b[1] + (b[2] + (b[3] + b[4] * T68) * T68) *
            T68) * T68) * s + (c[0] + (c[1] + c[2] * T68) * T68) * s * s ** 0.5 + d * s ** 2)

def smow(t):
    t = np.asanyarray(t)
    a = (999.842594, 6.793952e-2, -9.095290e-3, 1.001685e-4, -1.120083e-6, 6.536332e-9)
    T68 = T68conv(t)
    return (a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * T68) * T68) * T68) * T68) * T68)
    
def T68conv(T90):
    T90 = np.asanyarray(T90)
    return T90 * 1.00024

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
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
latg = latglider[0,okg[0]]
long = longlider[0,okg[0]]
target_latg = target_lat[okg[0]]
target_long = target_lon[okg[0]]
depthg = depthglider[okg[0],:]
tempg = tempglider[okg[0],:]
saltg = saltglider[okg[0],:]

#%%  Calculate density
        
densg = dens0(saltg, tempg)        
        
#%% Grid glider variables according to depth

depthg_gridded = np.arange(0,np.nanmax(depthg),0.5)
tempg_gridded = np.empty((len(timeg),len(depthg_gridded)))
tempg_gridded[:] = np.nan

for t,tt in enumerate(timeg):
    depthu,oku = np.unique(depthg[t,:],return_index=True)
    tempu = tempg[t,oku]
    okdd = np.isfinite(depthu)
    depth_fin = depthu[okdd]
    temp_fin = tempu[okdd]
    ok = np.isfinite(temp_fin)
    
    if np.sum(ok) < 3:
        tempg_gridded[t,:] = np.nan
    else:
        okd = depthg_gridded < np.max(depth_fin[ok])
        tempg_gridded[t,okd] = np.interp(depthg_gridded[okd],depth_fin[ok],temp_fin[ok]) 
        
#%% Get rid off of profiles with no data below 100 m

tempg_full = []
timeg_full = []
for t,tt in enumerate(timeg):
    okt = np.isfinite(tempg_gridded[t,:])
    if sum(depthg_gridded[okt] > 100) > 10:
        if tempg_gridded[t,0] != tempg_gridded[t,20]:
            tempg_full.append(tempg_gridded[t,:]) 
            timeg_full.append(tt) 
       
tempg_full = np.asarray(tempg_full)
timeg_full = np.asarray(timeg_full)

#%% Grid glider variables according to density

densg_gridded = np.arange(np.nanmin(densg),np.nanmax(densg),0.05)
tempg_gridded2 = np.empty((len(timeg),len(densg_gridded)))
tempg_gridded2[:] = np.nan

for t,tt in enumerate(timeg):
    densu,oku = np.unique(densg[t,:],return_index=True)
    tempu = tempg[t,oku]
    okdd = np.isfinite(densu)
    dens_fin = densu[okdd]
    temp_fin = tempu[okdd]
    ok = np.isfinite(temp_fin)
    
    if np.sum(ok) < 3:
        tempg_gridded2[t,:] = np.nan
    else:
        okd = densg_gridded < np.max(dens_fin[ok])
        tempg_gridded2[t,okd] = np.interp(densg_gridded[okd],dens_fin[ok],temp_fin[ok]) 
       

#%% Reading RTOFS ab files

# Reading lat and lon
lines_grid=[line.rstrip() for line in open(Dir+gridfile+'.b')]
hlon = np.array(readgrids(Dir+gridfile,'plon:',[0]))
hlat = np.array(readgrids(Dir+gridfile,'plat:',[0]))

# Extracting the longitudinal and latitudinal size array
idm=int([line.split() for line in lines_grid if 'longitudinal' in line][0][0])
jdm=int([line.split() for line in lines_grid if 'latitudinal' in line][0][0])

afiles = sorted(glob.glob(os.path.join(Dir,prefix_ab+'*.a')))

# Reading depths
lines=[line.rstrip() for line in open(afiles[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]==var_name:
        #print(line.split()[1])
        z.append(float(line.split()[1]))
z_RTOFS = np.asarray(z) 

nz = len(z_RTOFS) 

target_temp_RTOFS = np.empty((len(afiles),nz,))
target_temp_RTOFS[:] = np.nan
time_RTOFS = []
for x, file in enumerate(afiles):
    print(x)
    lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    year = int(file.split('.')[1][0:4])
    month = int(file.split('.')[1][4:6])
    day = int(file.split('.')[1][6:8])
    hour = int(file.split('.')[1][8:10])
    dt = int(file.split('.')[3][1:])
    timestamp_RTOFS = date2num(datetime(year,month,day,hour)) + dt/24
    time_RTOFS.append(num2date(timestamp_RTOFS))
    
    # Interpolating latg and longlider into RTOFS grid
    sublonRTOFS = np.interp(timestamp_RTOFS,timestampg,target_long)
    sublatRTOFS = np.interp(timestamp_RTOFS,timestampg,target_latg)
    oklonRTOFS = np.int(np.round(np.interp(sublonRTOFS,hlon[0,:],np.arange(len(hlon[0,:])))))
    oklatRTOFS = np.int(np.round(np.interp(sublatRTOFS,hlat[:,0],np.arange(len(hlat[:,0])))))
    
    # Reading 3D variable from binary file 
    temp_RTOFS = readBinz(file[:-2],'3z',var_name)
    #ts=readBin(afile,'archive','temp')
    target_temp_RTOFS[x,:] = temp_RTOFS[oklatRTOFS,oklonRTOFS,:]
    
    # Extracting list of variables
    #count=0
    #for line in lines:
    #    count+=1
    #    if line[0:5] == 'field':
    #        break

    #lines=lines[count:]
    #vars=[line.split()[0] for line in lines]
    
time_RTOFS = np.asarray(time_RTOFS)
timestamp_RTOFS = date2num(time_RTOFS)

#%% Figure

time_matrixR = np.tile(timestamp_RTOFS,(z_RTOFS.shape[0],1)).T
z_matrixR = np.tile(z_RTOFS,(time_RTOFS.shape[0],1))

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.contour(time_matrixR,-1*z_matrixR,target_temp_RTOFS,colors = 'lightgrey',**kw)
plt.contour(time_matrixR,-1*z_matrixR,target_temp_RTOFS,[26],colors = 'k')
plt.contourf(time_matrixR,-1*z_matrixR,target_temp_RTOFS,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(z_RTOFS)),-1*z_RTOFS,'--k')
#ax.set_ylim(36,22.5)
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
ax.set_ylim(-260,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('HMON-HYCOM',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/HMON-HYCOM_temp_Michael.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% glider profile

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(10, 3))
plt.contour(timeg,-depthg_gridded,tempg_gridded.T,colors = 'lightgrey',**kw)
plt.contour(timeg,-depthg_gridded,tempg_gridded.T,[26],colors = 'k')
plt.contourf(timeg,-depthg_gridded,tempg_gridded.T,cmap='RdYlBu_r',**kw)
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('ng288',fontsize=20)
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
ax.set_ylim(-260,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng288_Michael_vs_depth.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% glider profile with less gaps

kw = dict(levels = np.linspace(np.floor(np.nanmin(tempg_gridded)),\
                               np.ceil(np.nanmax(tempg_gridded)),17))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.contour(timeg_full,-depthg_gridded,tempg_full.T,colors = 'lightgrey',**kw)
plt.contour(timeg_full,-depthg_gridded,tempg_full.T,[26],colors = 'k')
plt.contourf(timeg_full,-depthg_gridded,tempg_full.T,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 6),len(depthg_gridded)),-depthg_gridded,'--k')
ax.set_ylabel('Depth (m)',fontsize=16)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('ng288',fontsize=20)
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
ax.set_ylim(-260,0)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng288_Michael_vs_depth2.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

plt.show()

