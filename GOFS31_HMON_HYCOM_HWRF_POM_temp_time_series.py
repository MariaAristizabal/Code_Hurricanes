#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:18:24 2019

@author: aristizabal
"""

#%%
#GOFS3.1 output model location
catalog31_ts = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# files for HMON-HYCOM output
Dir_HMON_HYCOM= '/Volumes/aristizabal/ncep_model/HMON-HYCOM_Michael/'
# RTOFS grid file name
gridfile = 'hwrf_rtofs_hat10.basin.regional.grid'
# RTOFS a/b file name
prefix_ab = 'michael14l.2018100718.hmon_rtofs_hat10_3z'
# Name of 3D variable
var_name = 'temp'

# files for HWRF-POM
ncfolder = '/Volumes/aristizabal/ncep_model/HWRF-POM_Michael/'
# POM grid file name
grid_file = 'michael14l.2018100718.pom.grid.nc'
# POM file name
prefix = 'michael14l.2018100718.pom.'
# Name of 3D variable
var_name = 'temp'

date_ini = '2018/10/08/00/00'
date_end = '2018/10/13/00/00'

target_lon = -86.27
target_lat = 28.94

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 
from datetime import datetime
from matplotlib.dates import date2num, num2date
import matplotlib.dates as mdates

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')
from utils4HYCOM import readBinz, readgrids

import os
import os.path
import glob

#%% Conversion from glider longitude and latitude to GOFS convention

if target_lon < 0: 
    target_lonG = 360 + target_lon
else:
    target_lonG = target_lon
target_latG = target_lat

#%% GOGF 3.1

GOFS31_ts = xr.open_dataset(catalog31_ts,decode_times=False)

latt31 = GOFS31_ts['lat'][:]
lonn31 = GOFS31_ts['lon'][:]
tt31 = GOFS31_ts['time']
t31 = netCDF4.num2date(tt31[:],tt31.units) 
depth31 = GOFS31_ts['depth'][:]

#%%
'''
# Conversion from GOFS convention to glider longitude and latitude
lon31g= np.empty((len(lonn31),))
lon31g[:] = np.nan
for i in range(len(lonn31)):
    if lonn31[i] > 180: 
        lon31g[i] = lonn31[i] - 360 
    else:
        lon31g[i] = lonn31[i]
lat31g = latt31
'''

#%%
### Choosing the lat and lon of profile

oklon31 = np.int(np.round(np.interp(target_lonG,lonn31,np.arange(len(lonn31)))))
oklat31 = np.int(np.round(np.interp(target_latG,latt31,np.arange(len(latt31)))))

lat31= latt31[oklat31]
lon31= lonn31[oklon31]

#%% GOFS 3.1
ti = datetime.strptime(date_ini,'%Y/%m/%d/%H/%M')
te = datetime.strptime(date_end,'%Y/%m/%d/%H/%M')

oktime31 = np.where(np.logical_and(t31 >= ti,t31 <= te))[0]
time31 = t31[oktime31]

# loading surface temperature and salinity
temp31 = GOFS31_ts['water_temp'][oktime31,:,oklat31,oklon31]

#%% Reading HMON-HYCOM ab files

# Reading lat and lon
lines_grid=[line.rstrip() for line in open(Dir_HMON_HYCOM+gridfile+'.b')]
hlon = np.array(readgrids(Dir_HMON_HYCOM+gridfile,'plon:',[0]))
hlat = np.array(readgrids(Dir_HMON_HYCOM+gridfile,'plat:',[0]))

# Extracting the longitudinal and latitudinal size array
idm=int([line.split() for line in lines_grid if 'longitudinal' in line][0][0])
jdm=int([line.split() for line in lines_grid if 'latitudinal' in line][0][0])

afiles = sorted(glob.glob(os.path.join(Dir_HMON_HYCOM,prefix_ab+'*.a')))

# Reading depths
lines=[line.rstrip() for line in open(afiles[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]==var_name:
        #print(line.split()[1])
        z.append(float(line.split()[1]))
z_HMON_HYCOM = np.asarray(z) 

nz = len(z_HMON_HYCOM) 

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
    #sublonHMON_HYCOM = np.interp(timestamp_HMON_HYCOM,timestampg,target_long)
    #sublatHMON_HYCOM = np.interp(timestamp_HMON_HYCOM,timestampg,target_latg)
    oklonHMON_HYCOM = np.int(np.round(np.interp(target_lonG,hlon[0,:],np.arange(len(hlon[0,:])))))
    oklatHMON_HYCOM = np.int(np.round(np.interp(target_latG,hlat[:,0],np.arange(len(hlat[:,0])))))
    
    # Reading 3D variable from binary file 
    temp_HMON_HYCOM = readBinz(file[:-2],'3z',var_name)
    #ts=readBin(afile,'archive','temp')
    target_temp_HMON_HYCOM[x,:] = temp_HMON_HYCOM[oklatHMON_HYCOM,oklonHMON_HYCOM,:]
    
time_HMON_HYCOM = np.asarray(time_HMON_HYCOM)
timestamp_HMON_HYCOM = date2num(time_HMON_HYCOM) 

#%% Reading POM grid files

pom_grid = xr.open_dataset(ncfolder + grid_file)

lonc = np.asarray(pom_grid['east_e'][:])
latc = np.asarray( pom_grid['north_e'][:])
#lonu, latu = pom_grid['east_u'][:], pom_grid['north_u'][:]
#lonv, latv = pom_grid['east_v'][:], pom_grid['north_v'][:]
zlevc = np.asarray(pom_grid['zz'][:])
topoz = np.asarray(pom_grid['h'][:])

#%% Reading POM temperature

ncfiles = sorted(glob.glob(os.path.join(ncfolder,prefix+'*0*.nc')))

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
    #sublonpom = np.interp(timestamp_pom,timestampg,long)
    #sublatpom = np.interp(timestamp_pom,timestampg,latg)
    oklonpom = np.int(np.round(np.interp(target_lon,lonc[0,:],np.arange(len(lonc[0,:])))))
    oklatpom = np.int(np.round(np.interp(target_lat,latc[:,0],np.arange(len(latc[:,0])))))

    target_temp_pom[x,:] = np.asarray(pom['t'][0,:,oklatpom,oklonpom])
    target_topoz_pom[x] = np.asarray(topoz[oklatpom,oklonpom])

timestamp_pom = date2num(time_pom)

z_matrix_pom = np.dot(target_topoz_pom.reshape(-1,1),zlevc.reshape(1,-1))

#%% Figure

time_matrix31 = np.tile(time31,(depth31.shape[0],1)).T
depth_matrix31 = np.tile(depth31,(time31.shape[0],1))

#kw = dict(levels = np.linspace(np.floor(np.nanmin(temp31)),\
#                               np.ceil(np.nanmax(temp31)),18))

kw = dict(levels = np.linspace(14,30,17))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.contour(time_matrix31,-1*depth_matrix31,temp31,colors = 'lightgrey',**kw)
plt.contour(time_matrix31,-1*depth_matrix31,temp31,[26],colors = 'k')
plt.contourf(time_matrix31,-1*depth_matrix31,temp31,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 12),len(depth31)),-1*depth31,'--k')
#ax.set_ylim(36,22.5)
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
ax.set_ylim(-260,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('GOFS 3.1',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_Michael_at_Oct_10_12.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure

time_matrixpom = np.tile(date2num(time_pom),(z_matrix_pom.shape[1],1)).T
#depth_matrixpom = np.tile(z_matrix_pom,(time_pom.shape[0],1))

#kw = dict(levels = np.linspace(11,30,20))
kw = dict(levels = np.linspace(14,30,17))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.contour(time_matrixpom,z_matrix_pom,target_temp_pom,colors = 'lightgrey',**kw)
plt.contour(time_matrixpom,z_matrix_pom,target_temp_pom,[26],colors = 'k')
plt.contourf(time_matrixpom,z_matrix_pom,target_temp_pom,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 12),len(depth31)),-1*depth31,'--k')
#ax.set_ylim(36,22.5)
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
ax.set_ylim(-260,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('HWRF-POM',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/HWRF_pom_temp_Michael_at_Oct_10_12.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Figure

time_matrixHH = np.tile(date2num(time_HMON_HYCOM),(z_HMON_HYCOM.shape[0],1)).T
depth_matrixHH = np.tile(z_HMON_HYCOM,(time_HMON_HYCOM.shape[0],1))

kw = dict(levels = np.linspace(14,30,17))

fig, ax = plt.subplots(figsize=(11, 1.7))
plt.contour(time_matrixHH,-1*depth_matrixHH,target_temp_HMON_HYCOM,colors = 'lightgrey',**kw)
plt.contour(time_matrixHH,-1*depth_matrixHH,target_temp_HMON_HYCOM,[26],colors = 'k')
plt.contourf(time_matrixHH,-1*depth_matrixHH,target_temp_HMON_HYCOM,cmap='RdYlBu_r',**kw)
plt.plot(np.tile(datetime(2018, 10, 10, 12),len(depth31)),-1*depth31,'--k')
#ax.set_ylim(36,22.5)
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
ax.set_ylim(-260,0)
yl = ax.set_ylabel('Depth (m)',fontsize=16) #,labelpad=20)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Temperature ($^\circ$C)',fontsize=16)
ax.set_title('HMON-HYCOM',fontsize=20)
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/HMON_HYCOM_temp_Michael_at_Oct_10_12.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Time series at 10 m depth

zz = 10.0
nz31 = np.where(depth31 >= zz)[0][0]
nzH = np.where(z_HMON_HYCOM >= zz)[0][0]
nzp = np.where(z_matrix_pom[0,:] <= -zz)[0][0]

fig, ax = plt.subplots(figsize=(8.8, 1.7))
plt.plot(time31,temp31[:,nz31],'o-',color='royalblue',\
         linewidth=3,label='GOFS 3.1')
plt.plot(date2num(time_HMON_HYCOM),target_temp_HMON_HYCOM[:,nzH],'o-',color='red',\
         linewidth=3,label='HMON-HYCOM')
plt.plot(date2num(time_pom),target_temp_pom[:,nzp],'o-',color='darkorchid',\
         linewidth=3,label='HWRF-POM')
plt.legend()
plt.plot(np.tile(datetime(2018, 10, 10, 12),len(np.arange(27,30,0.1))),np.arange(27,30,0.1),'--k')
ax.set_title('Temperature at 10 m depth',fontsize=20)
plt.ylabel('($^oC$)',size=20)
ax.set_xlim(datetime(2018,10,8),datetime(2018,10,13))
ax.set_ylim(27,29.5)
plt.grid(True)

xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_Michael_at_Oct_10_12_10meters.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%  Vertical profile
'''
ttmic = datetime(2018,10,9,0)
nt31 = np.where(time31 == ttmic)[0][0]
ntH = np.where(date2num(time_HMON_HYCOM) == date2num(ttmic))[0][0]
ntp = np.where(date2num(time_pom) >= date2num(ttmic))[0][0]

plt.figure(figsize=(3, 6.5))
plt.plot(temp31[nt31,:],-1*depth31,'o-',color='royalblue',\
         linewidth=3,label='GOFS 3.1')
plt.plot(target_temp_HMON_HYCOM[ntH,:],-1*z_HMON_HYCOM,'o-',color='red',\
         linewidth=3,label='HMON-HYCOM')
plt.plot(target_temp_pom[ntp,:],z_matrix_pom[ntp,:],'o-',color='darkorchid',\
         linewidth=3,label='HWRF-POM')
plt.plot(np.tile(26,220),np.arange(-220,0),'--k')
plt.legend(fontsize=12)
plt.ylim(-220,0)
plt.xlim(10,30)
plt.xlabel('Temperature ($^oC$)',size=18)
plt.ylabel('Depth (m)',size=18)
plt.title(str(ttmic),size=18)


file = folder + 'vert_prof_temp_' + '_GOFS31_HYCOM_POM_'+ str(ttmic) +'.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()
'''











