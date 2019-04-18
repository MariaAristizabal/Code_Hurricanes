#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:28:15 2019

@author: aristizabal
"""

#%% User input

# Glider data 
# ng288
gdata = 'https://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc'

# mat file for GOFS 3.1
mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng288.mat'

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

# Directories where HWRF-HYCOM files reside
Dir_HWRF_Hycom = '/Volumes/aristizabal/ncep_model/HWRF-Hycom_exp_Michael/'
Dir_HWRF_Hycom_WW3 = '/Volumes/aristizabal/ncep_model/HWRF-Hycom-WW3_exp_Michael2/'
# HYCOM a/b file name
prefix_ab_exp = 'michael14l.2018100718.hwrf_rtofs_hat10_3z'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# date limits
date_ini = '2018-10-08T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'


#%%

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from matplotlib.dates import date2num, num2date
import xarray as xr
import netCDF4

import scipy.io as sio
ng288 = sio.loadmat(mat_file)

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')
from utils4HYCOM import readBinz, readgrids

import os
import os.path
import glob

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
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


#%% load GOFS 3.1 

tstamp31 =  ng288['timeGOFS31'][:,0]
depth31 = ng288['depthGOFS31'][:,0]
tem31 = ng288['tempGOFS31'][:]

#%% Changing timestamps to datenum

tim31 = []
for i in np.arange(len(tstamp31)):
    tim31.append(datetime.fromordinal(int(tstamp31[i])) + \
        timedelta(days=tstamp31[i]%1) - timedelta(days = 366))
tt31 = np.asarray(tim31)

tti = datetime.strptime(date_ini, '%Y-%m-%dT%H:%M:%SZ')
tte = datetime.strptime(date_end, '%Y-%m-%dT%H:%M:%SZ')

oktime31 = np.logical_and(tt31>=tti, tt31<=tte)

time31 = tt31[oktime31]
timestamp31 = date2num(time31)
temp31 = tem31[:,oktime31] 

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
    sublonHMON_HYCOM = np.interp(timestamp_HMON_HYCOM,timestampg,target_long)
    sublatHMON_HYCOM = np.interp(timestamp_HMON_HYCOM,timestampg,target_latg)
    oklonHMON_HYCOM = np.int(np.round(np.interp(sublonHMON_HYCOM,hlon[0,:],np.arange(len(hlon[0,:])))))
    oklatHMON_HYCOM = np.int(np.round(np.interp(sublatHMON_HYCOM,hlat[:,0],np.arange(len(hlat[:,0])))))
    
    # Reading 3D variable from binary file 
    temp_HMON_HYCOM = readBinz(file[:-2],'3z',var_name)
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
    sublonpom = np.interp(timestamp_pom,timestampg,long)
    sublatpom = np.interp(timestamp_pom,timestampg,latg)
    oklonpom = np.int(np.round(np.interp(sublonpom,lonc[0,:],np.arange(len(lonc[0,:])))))
    oklatpom = np.int(np.round(np.interp(sublatpom,latc[:,0],np.arange(len(latc[:,0])))))

    target_temp_pom[x,:] = np.asarray(pom['t'][0,:,oklatpom,oklonpom])
    target_topoz_pom[x] = np.asarray(topoz[oklatpom,oklonpom])

timestamp_pom = date2num(time_pom)

z_matrix_pom = np.dot(target_topoz_pom.reshape(-1,1),zlevc.reshape(1,-1))


#%% Reading HWRF-Hycom experimental ab files

# Reading lat and lon
lines_grid=[line.rstrip() for line in open(Dir_HWRF_Hycom+gridfile+'.b')]
hlon = np.array(readgrids(Dir_HWRF_Hycom+gridfile,'plon:',[0]))
hlat = np.array(readgrids(Dir_HWRF_Hycom+gridfile,'plat:',[0]))

# Extracting the longitudinal and latitudinal size array
idm=int([line.split() for line in lines_grid if 'longitudinal' in line][0][0])
jdm=int([line.split() for line in lines_grid if 'latitudinal' in line][0][0])

afiles_exp = sorted(glob.glob(os.path.join(Dir_HWRF_Hycom,prefix_ab_exp+'*.a')))

# Reading depths
lines=[line.rstrip() for line in open(afiles_exp[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]==var_name:
        #print(line.split()[1])
        z.append(float(line.split()[1]))
z_HWRF_Hycom = np.asarray(z)

nz = len(z_HWRF_Hycom)

target_temp_HWRF_Hycom = np.empty((len(afiles_exp),nz,))
target_temp_HWRF_Hycom[:] = np.nan
time_HWRF_Hycom = []
for x, file in enumerate(afiles_exp):
    print(x)
    lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    year = int(file.split('.')[1][0:4])
    month = int(file.split('.')[1][4:6])
    day = int(file.split('.')[1][6:8])
    hour = int(file.split('.')[1][8:10])
    dt = int(file.split('.')[3][1:])
    timestamp_HWRF_Hycom = date2num(datetime(year,month,day,hour)) + dt/24
    time_HWRF_Hycom.append(num2date(timestamp_HWRF_Hycom))

    # Interpolating latg and longlider into RTOFS grid
    sublonHWRF_Hycom = np.interp(timestamp_HWRF_Hycom,timestampg,target_long)
    sublatHWRF_Hycom = np.interp(timestamp_HWRF_Hycom,timestampg,target_latg)
    oklonHWRF_Hycom = np.int(np.round(np.interp(sublonHWRF_Hycom,hlon[0,:],np.arange(len(hlon[0,:])))))
    oklatHWRF_Hycom = np.int(np.round(np.interp(sublatHWRF_Hycom,hlat[:,0],np.arange(len(hlat[:,0])))))

    # Reading 3D variable from binary file
    temp_HWRF_Hycom = readBinz(file[:-2],'3z',var_name)
    #ts=readBin(afile,'archive','temp')
    target_temp_HWRF_Hycom[x,:] = temp_HWRF_Hycom[oklatHWRF_Hycom,oklonHWRF_Hycom,:]

    # Extracting list of variables
    #count=0
    #for line in lines:
    #    count+=1
    #    if line[0:5] == 'field':
    #        break

    #lines=lines[count:]
    #vars=[line.split()[0] for line in lines]

time_HWRF_Hycom = np.asarray(time_HWRF_Hycom)
timestamp_HWRF_Hycom = date2num(time_HWRF_Hycom)

#%% Reading HWRF-Hycom_WW3 experimental ab files

# Reading lat and lon
lines_grid=[line.rstrip() for line in open(Dir_HWRF_Hycom_WW3+gridfile+'.b')]
hlon = np.array(readgrids(Dir_HWRF_Hycom_WW3+gridfile,'plon:',[0]))
hlat = np.array(readgrids(Dir_HWRF_Hycom_WW3+gridfile,'plat:',[0]))

# Extracting the longitudinal and latitudinal size array
idm=int([line.split() for line in lines_grid if 'longitudinal' in line][0][0])
jdm=int([line.split() for line in lines_grid if 'latitudinal' in line][0][0])

afiles_exp = sorted(glob.glob(os.path.join(Dir_HWRF_Hycom_WW3,prefix_ab_exp+'*.a')))

# Reading depths
lines=[line.rstrip() for line in open(afiles_exp[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]==var_name:
        #print(line.split()[1])
        z.append(float(line.split()[1]))
z_HWRF_Hycom_WW3 = np.asarray(z)

nz = len(z_HWRF_Hycom_WW3)

target_temp_HWRF_Hycom_WW3 = np.empty((len(afiles_exp),nz,))
target_temp_HWRF_Hycom_WW3[:] = np.nan
time_HWRF_Hycom_WW3 = []
for x, file in enumerate(afiles_exp):
    print(x)
    lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    year = int(file.split('.')[1][0:4])
    month = int(file.split('.')[1][4:6])
    day = int(file.split('.')[1][6:8])
    hour = int(file.split('.')[1][8:10])
    dt = int(file.split('.')[3][1:])
    timestamp_HWRF_Hycom_WW3 = date2num(datetime(year,month,day,hour)) + dt/24
    time_HWRF_Hycom_WW3.append(num2date(timestamp_HWRF_Hycom_WW3))

    # Interpolating latg and longlider into RTOFS grid
    sublonHWRF_Hycom_WW3 = np.interp(timestamp_HWRF_Hycom_WW3,timestampg,target_long)
    sublatHWRF_Hycom_WW3 = np.interp(timestamp_HWRF_Hycom_WW3,timestampg,target_latg)
    oklonHWRF_Hycom_WW3 = np.int(np.round(np.interp(sublonHWRF_Hycom_WW3,hlon[0,:],np.arange(len(hlon[0,:])))))
    oklatHWRF_Hycom_WW3 = np.int(np.round(np.interp(sublatHWRF_Hycom_WW3,hlat[:,0],np.arange(len(hlat[:,0])))))

    # Reading 3D variable from binary file
    temp_HWRF_Hycom_WW3 = readBinz(file[:-2],'3z',var_name)
    #ts=readBin(afile,'archive','temp')
    target_temp_HWRF_Hycom_WW3[x,:] = temp_HWRF_Hycom_WW3[oklatHWRF_Hycom_WW3,oklonHWRF_Hycom_WW3,:]

    # Extracting list of variables
    #count=0
    #for line in lines:
    #    count+=1
    #    if line[0:5] == 'field':
    #        break

    #lines=lines[count:]
    #vars=[line.split()[0] for line in lines]

time_HWRF_Hycom_WW3 = np.asarray(time_HWRF_Hycom_WW3)
timestamp_HWRF_Hycom_WW3 = date2num(time_HWRF_Hycom_WW3)

#%%  Vertical profile 1 days before Michael

ttmic = datetime(2018,10,9,6)
nt31 = np.where(time31 == ttmic)[0][0]
ntg = np.where(timeg > ttmic)[0][0]
ntH = np.where(date2num(time_HMON_HYCOM) == date2num(ttmic))[0][0]
ntp = np.where(date2num(time_pom) > date2num(ttmic))[0][0]

plt.figure(figsize=(3, 6.5))
plt.plot(tempg_gridded[ntg,:],-depthg_gridded,'o-g',linewidth=3,label='ng288')
plt.plot(temp31[:,nt31],-1*depth31,'o-',color='royalblue',\
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

file = folder + 'vert_prof_temp_' + inst_id + '_GOFS31_HYCOM_POM_'+ str(ttmic) +'.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()
     
#%%  Vertical profile during Michael

ttmic = datetime(2018,10,10,6)
nt31 = np.where(time31 == ttmic)[0][0]
ntg = np.where(timeg > ttmic)[0][0]
ntH = np.where(date2num(time_HMON_HYCOM) == date2num(ttmic))[0][0]
ntp = np.where(date2num(time_pom) > date2num(ttmic))[0][0]

plt.figure(figsize=(3, 6.5))
plt.plot(tempg_gridded[ntg,:],-depthg_gridded,'o-g',linewidth=3,label='ng288')
plt.plot(temp31[:,nt31],-1*depth31,'o-',color='royalblue',\
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

file = folder + 'vert_prof_temp_' + inst_id + '_GOFS31_HYCOM_POM_'+ str(ttmic) +'.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%%  Vertical profile 1 day after Michael

ttmic = datetime(2018,10,11,6)
nt31 = np.where(time31 == ttmic)[0][0]
ntg = np.where(timeg > ttmic)[0][0]
ntH = np.where(date2num(time_HMON_HYCOM) == date2num(ttmic))[0][0]
ntp = np.where(date2num(time_pom) > date2num(ttmic))[0][0]

plt.figure(figsize=(3, 6.5))
plt.plot(tempg_gridded[ntg,:],-depthg_gridded,'o-g',linewidth=3,label='ng288')
plt.plot(temp31[:,nt31],-1*depth31,'o-',color='royalblue',\
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

file = folder + 'vert_prof_temp_' + inst_id + '_GOFS31_HYCOM_POM_'+ str(ttmic) +'.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%%  Vertical profile 1 days before Michael

ttmic = datetime(2018,10,9,6)
nt31 = np.where(time31 == ttmic)[0][0]
ntg = np.where(timeg > ttmic)[0][0]
ntHH_exp = np.where(date2num(time_HWRF_Hycom) == date2num(ttmic))[0][0]
ntHHW_exp = np.where(date2num(time_HWRF_Hycom_WW3) > date2num(ttmic))[0][0]

plt.figure(figsize=(3, 6.5))
plt.plot(tempg_gridded[ntg,:],-depthg_gridded,'o-g',linewidth=3,label='ng288')
plt.plot(temp31[:,nt31],-1*depth31,'o-',color='royalblue',\
         linewidth=3,label='GOFS 3.1')
plt.plot(target_temp_HWRF_Hycom[ntHH_exp,:],-1*z_HWRF_Hycom,'o-',color='red',\
         linewidth=3,label='HWRF-HYCOM Exp')
plt.plot(target_temp_HWRF_Hycom_WW3[ntHHW_exp,:],-1*z_HWRF_Hycom_WW3,'o-',color='darkorchid',\
         linewidth=3,label='HWRF-HYCOM-WW3 Exp')
plt.plot(np.tile(26,220),np.arange(-220,0),'--k')
plt.legend(loc='upper left',fontsize=12,bbox_to_anchor=(-0.35,1))
plt.ylim(-220,0)
plt.xlim(10,30)
plt.xlabel('Temperature ($^oC$)',size=18)
plt.ylabel('Depth (m)',size=18)
plt.title(str(ttmic),size=18)

file = folder + 'vert_prof_temp_' + inst_id + '_GOFS31_HWRF_HYCOM_WW3_exp_'+ str(ttmic) +'.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%%  Vertical profile during Michael

ttmic = datetime(2018,10,10,6)
nt31 = np.where(time31 == ttmic)[0][0]
ntg = np.where(timeg > ttmic)[0][0]
ntHH_exp = np.where(date2num(time_HWRF_Hycom) == date2num(ttmic))[0][0]
ntHHW_exp = np.where(date2num(time_HWRF_Hycom_WW3) > date2num(ttmic))[0][0]

plt.figure(figsize=(3, 6.5))
plt.plot(tempg_gridded[ntg,:],-depthg_gridded,'o-g',linewidth=3,label='ng288')
plt.plot(temp31[:,nt31],-1*depth31,'o-',color='royalblue',\
         linewidth=3,label='GOFS 3.1')
plt.plot(target_temp_HWRF_Hycom[ntHH_exp,:],-1*z_HWRF_Hycom,'o-',color='red',\
         linewidth=3,label='HWRF-HYCOM Exp')
plt.plot(target_temp_HWRF_Hycom_WW3[ntHHW_exp,:],-1*z_HWRF_Hycom_WW3,'o-',color='darkorchid',\
         linewidth=3,label='HWRF-HYCOM-WW3 Exp')
plt.plot(np.tile(26,220),np.arange(-220,0),'--k')
#plt.legend(loc='upper left',fontsize=12,bbox_to_anchor=(-0.35,1))
plt.ylim(-220,0)
plt.xlim(10,30)
plt.xlabel('Temperature ($^oC$)',size=18)
plt.ylabel('Depth (m)',size=18)
plt.title(str(ttmic),size=18)

file = folder + 'vert_prof_temp_' + inst_id + '_GOFS31_HWRF_HYCOM_WW3_exp_'+ str(ttmic) +'.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()

#%%  Vertical profile 1 day after Michael

ttmic = datetime(2018,10,11,6)
nt31 = np.where(time31 == ttmic)[0][0]
ntg = np.where(timeg > ttmic)[0][0]
ntHH_exp = np.where(date2num(time_HWRF_Hycom) == date2num(ttmic))[0][0]
ntHHW_exp = np.where(date2num(time_HWRF_Hycom_WW3) > date2num(ttmic))[0][0]

plt.figure(figsize=(3, 6.5))
plt.plot(tempg_gridded[ntg,:],-depthg_gridded,'o-g',linewidth=3,label='ng288')
plt.plot(temp31[:,nt31],-1*depth31,'o-',color='royalblue',\
         linewidth=3,label='GOFS 3.1')
plt.plot(target_temp_HWRF_Hycom[ntHH_exp,:],-1*z_HWRF_Hycom,'o-',color='red',\
         linewidth=3,label='HWRF-HYCOM Exp')
plt.plot(target_temp_HWRF_Hycom_WW3[ntHHW_exp,:],-1*z_HWRF_Hycom_WW3,'o-',color='darkorchid',\
         linewidth=3,label='HWRF-HYCOM-WW3 Exp')
plt.plot(np.tile(26,220),np.arange(-220,0),'--k')
#plt.legend(loc='upper left',fontsize=12,bbox_to_anchor=(-0.35,1))
plt.ylim(-220,0)
plt.xlim(10,30)
plt.xlabel('Temperature ($^oC$)',size=18)
plt.ylabel('Depth (m)',size=18)
plt.title(str(ttmic),size=18)

file = folder + 'vert_prof_temp_' + inst_id + '_GOFS31_HWRF_HYCOM_WW3_exp_'+ str(ttmic) +'.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 
plt.show()








