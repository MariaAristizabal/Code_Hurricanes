#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:22:14 2020

@author: aristizabal
"""
#%% ng288 thredds address

gdata = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc'
         
# date limits
date_ini = '2018-10-07T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'

#%%
import xarray as xr
import netCDF4
import numpy as np
from datetime import datetime

#%% Reading glider data

ncglider = xr.open_dataset(gdata+'#fillmismatch',decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
timeglider = ncglider.time
timeglider = netCDF4.num2date(timeglider[:],timeglider.units)
timestampglider = np.asarray(ncglider.time[:])
tempglider = np.array(ncglider.temperature[0,:,:])
depthglider = np.array(ncglider.depth[0,:,:])

#%% Subsetting the glider data to the desired time window

tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

okg = np.where(np.logical_and(timeglider.T >= tmin, timeglider.T <= tmax))[0]

timeg = timeglider[0,okg]
timestampg = timestampglider[0,okg]
latg = latglider[0,okg]
long = longlider[0,okg]
depthg = depthglider[okg,:]
tempg = tempglider[okg,:]

#%% Conversion from glider longitude and latitude to HYCOM convention
# NOTE: in the HYCOM that I read, the longitude variable has a diferent 
# convention than the glider variable. 
# glider convention: -180 to 180
# HYCOM convention: 0 to 360 

target_lon = []
for lon in long:
    if lon < 0: 
        target_lon.append(360 + lon)
    else:
        target_lon.append(lon)
target_lon = np.array(target_lon)
target_lat = np.array(latg)

#%%

def Extract_glider_transect_model(timestamp_model,lon_model,lat_model,temp_model,\
                                  timestamp_glider,target_lon,target_lat):

    sublon = np.interp(timestamp_model,timestamp_glider,target_lon)
    sublat = np.interp(timestamp_model,timestampg,target_lat)
    oklon = np.int(np.round(np.interp(sublon,lon_model,np.arange(len(lon_model)))))
    oklat = np.int(np.round(np.interp(sublat,lat_model,np.arange(len(lat_model)))))
    
    # dimension temp_model are [lat,lon,depth]
    target_temp_model = temp_model[oklat,oklon,:]
    
    return target_temp_model


#%% Example of how I read temperature from HYCOM files 

# Reading lat and lon
lines_grid=[line.rstrip() for line in open(Dir+gridfile+'.b')]
hlon = np.array(readgrids(Dir+gridfile,'plon:',[0]))
hlat = np.array(readgrids(Dir+gridfile,'plat:',[0]))

# Extracting the longitudinal and latitudinal size array
idm=int([line.split() for line in lines_grid if 'longitudinal' in line][0][0])
jdm=int([line.split() for line in lines_grid if 'latitudinal' in line][0][0])

afiles = sorted(glob.glob(os.path.join(Dir,prefix_ab+'*.a')))
# Note: the 3D output is 6 hourly. therefore half of the a files only contain 3D output
afiles_6h = afiles[::2] 
nz = 41

target_temp_HYCOM = np.empty((len(afiles_6h),nz,))
target_temp_HYCOM[:] = np.nan
target_thknss_HYCOM = np.empty((len(afiles_6h),nz,))
target_thknss_HYCOM[:] = np.nan
time_HYCOM = []
dens = np.empty((len(afiles_6h),nz,))
dens[:] = np.nan

for x, file in enumerate(afiles_6h):
    print(x)
    lines=[line.rstrip() for line in open(file[:-2]+'.b')]

    #Reading time stamp
    time_stamp = lines[-1].split()[2]
    hycom_days = lines[-1].split()[3]
    tzero = datetime(1901,1,1,0,0)
    time_RT = tzero+timedelta(float(hycom_days))
    time_HYCOM.append(time_RT)
    timestamp_HYCOM = time.mktime(time_RT.timetuple())
    depths=[1,3,5,7.5,]
    depths=[]
    # Reading layer density
    for line in lines:
        if line[0:5] == 'field':
            print(line)
            for i in range(len(line.split())):
                if line.split()[i] == 'dens':
                    pos_dens = i
 
    den = []              
    for line in lines:
        if line.split()[0] == 'temp':
            den.append(line.split()[pos_dens-1])
        
    dens[x,:] = den
    
    # Reading 3D variable from binary file 
    temp_HYCOM = readBinz(file[:-2],'3z','temp')
    
    target_temp_HYCOM[x,:] = Extract_glider_transect_model(timestamp_HYCOM,hlon[0,:],hlat[:,0],temp_HYCOM,\
                                  timestampg,target_lon,target_lat)